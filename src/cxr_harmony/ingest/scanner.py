"""Discovery, validation and indexing of an incoming multi-site delivery.

Ingest answers one question per object: does this belong in the cohort, and if
not, why not. Objects that fail are recorded in a quarantine file with a reason
code rather than skipped, because "we received 40,000 and kept 38,112" is a
statement a partner site will eventually ask you to account for, and the answer
has to be reconstructible months later.

Nothing here writes pixel data. See :mod:`cxr_harmony.workspace` for why.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError

from ..schema.vocab import ACCEPTED_BODY_PARTS, ACCEPTED_MODALITIES, QuarantineReason
from ..workspace import Workspace, write_jsonl

#: Header fields without which an object cannot be placed in the cohort at all.
REQUIRED_TAGS = ("SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID", "PatientID")

_CHUNK = 1 << 20


@dataclass
class IngestRecord:
    """One accepted object, indexed in place."""

    source_path: str
    site_id: str
    sha256: str
    size_bytes: int
    sop_uid: str
    study_uid: str
    series_uid: str
    modality: str
    body_part: str
    report_path: str | None

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "site_id": self.site_id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "sop_uid": self.sop_uid,
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "modality": self.modality,
            "body_part": self.body_part,
            "report_path": self.report_path,
        }


@dataclass
class IngestResult:
    """Counts and records for one ingest run."""

    accepted: list[IngestRecord] = field(default_factory=list)
    quarantined: list[dict] = field(default_factory=list)

    @property
    def n_accepted(self) -> int:
        return len(self.accepted)

    @property
    def n_quarantined(self) -> int:
        return len(self.quarantined)

    def reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.quarantined:
            counts[record["reason"]] = counts.get(record["reason"], 0) + 1
        return dict(sorted(counts.items()))


def sha256_file(path: Path) -> str:
    """Streaming digest, so a multi-gigabyte archive does not have to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def discover_sites(src: Path) -> list[str]:
    """Return site directory names, in stable order.

    A site is any immediate subdirectory containing an ``images`` folder. Files at
    the delivery root — a covering letter, a ground-truth file — are ignored.
    """
    return sorted(p.name for p in Path(src).iterdir() if p.is_dir() and (p / "images").is_dir())


def _iter_candidates(src: Path, site_id: str) -> Iterator[Path]:
    yield from sorted((src / site_id / "images").rglob("*.dcm"))


def _find_report(src: Path, site_id: str, image_path: Path) -> Path | None:
    candidate = src / site_id / "reports" / f"{image_path.stem}.txt"
    return candidate if candidate.exists() else None


def _classify(ds: pydicom.Dataset) -> tuple[QuarantineReason, str] | None:
    """Return a rejection reason, or ``None`` if the object is acceptable."""
    for tag in REQUIRED_TAGS:
        if not getattr(ds, tag, None):
            return QuarantineReason.MISSING_REQUIRED_TAG, tag

    modality = str(getattr(ds, "Modality", "") or "").upper()
    if modality not in ACCEPTED_MODALITIES:
        return QuarantineReason.WRONG_MODALITY, modality or "<empty>"

    body_part = str(getattr(ds, "BodyPartExamined", "") or "").upper()
    # An empty BodyPartExamined is common and not disqualifying on its own; a
    # populated value naming a different region is.
    if body_part and body_part not in ACCEPTED_BODY_PARTS:
        return QuarantineReason.WRONG_BODY_PART, body_part

    if "PixelData" not in ds:
        return QuarantineReason.NO_PIXEL_DATA, ""

    return None


def ingest(src: Path, workspace: Workspace) -> IngestResult:
    """Index an incoming delivery into ``workspace``.

    Idempotent: running twice over an unchanged delivery rewrites byte-identical
    manifests. Content-level duplicates within a run are quarantined rather than
    indexed twice, which matters because sites re-send overlapping batches after
    a failed transfer far more often than they admit.
    """
    src = Path(src)
    workspace.ensure()

    result = IngestResult()
    seen_digests: dict[str, str] = {}

    for site_id in discover_sites(src):
        for path in _iter_candidates(src, site_id):
            relative = path.relative_to(src).as_posix()
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=False)
            except InvalidDicomError:
                result.quarantined.append(
                    _quarantine(relative, site_id, QuarantineReason.NOT_DICOM, "")
                )
                continue
            except Exception as exc:  # noqa: BLE001 - any read failure is a quarantine
                result.quarantined.append(
                    _quarantine(
                        relative, site_id, QuarantineReason.UNREADABLE, type(exc).__name__
                    )
                )
                continue

            rejection = _classify(ds)
            if rejection is not None:
                reason, detail = rejection
                result.quarantined.append(_quarantine(relative, site_id, reason, detail))
                continue

            digest = sha256_file(path)
            if digest in seen_digests:
                result.quarantined.append(
                    _quarantine(
                        relative,
                        site_id,
                        QuarantineReason.DUPLICATE_CONTENT,
                        f"identical to {seen_digests[digest]}",
                    )
                )
                continue
            seen_digests[digest] = relative

            report = _find_report(src, site_id, path)
            result.accepted.append(
                IngestRecord(
                    source_path=relative,
                    site_id=site_id,
                    sha256=digest,
                    size_bytes=path.stat().st_size,
                    sop_uid=str(ds.SOPInstanceUID),
                    study_uid=str(ds.StudyInstanceUID),
                    series_uid=str(ds.SeriesInstanceUID),
                    modality=str(ds.Modality).upper(),
                    body_part=str(getattr(ds, "BodyPartExamined", "") or "").upper(),
                    report_path=(
                        report.relative_to(src).as_posix() if report is not None else None
                    ),
                )
            )

    result.accepted.sort(key=lambda r: (r.site_id, r.source_path))
    result.quarantined.sort(key=lambda r: (r["site_id"], r["source_path"]))

    write_jsonl(workspace.raw_manifest, (r.to_dict() for r in result.accepted))
    write_jsonl(workspace.quarantine, result.quarantined)
    return result


def _quarantine(source_path: str, site_id: str, reason: QuarantineReason, detail: str) -> dict:
    return {
        "source_path": source_path,
        "site_id": site_id,
        "reason": reason.value,
        "detail": detail,
    }
