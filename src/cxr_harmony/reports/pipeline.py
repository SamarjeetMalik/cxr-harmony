"""Report stage: parse, scrub, extract, and write alongside the de-identified images."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pydicom

from ..deid.pseudonym import Pseudonymiser, load_or_create_key
from ..schema.vocab import ReportSection
from ..workspace import Workspace, read_jsonl, write_jsonl
from .labels import extract_labels
from .parser import clinical_text, parse_sections
from .scrub import scrub_report


@dataclass
class ReportRecord:
    study_uid: str
    site_id: str
    relative_path: str
    sections: dict[str, str]
    findings: list[str]
    denied_findings: list[str]
    redaction_count: int
    source_path: str

    def to_dict(self) -> dict:
        return {
            "study_uid": self.study_uid,
            "site_id": self.site_id,
            "relative_path": self.relative_path,
            "sections": self.sections,
            "findings": self.findings,
            "denied_findings": self.denied_findings,
            "redaction_count": self.redaction_count,
            "source_path": self.source_path,
        }


@dataclass
class ReportResult:
    records: list[ReportRecord] = field(default_factory=list)

    @property
    def n_reports(self) -> int:
        return len(self.records)

    @property
    def total_redactions(self) -> int:
        return sum(r.redaction_count for r in self.records)


def _identifiers_from_source(ds: pydicom.Dataset) -> tuple[list[str], list[str], list[str]]:
    """Harvest the identifier literals the header knows about.

    This is why the report stage reads the *original* object rather than the
    de-identified one: the de-identified header no longer contains the name we
    need to search the prose for.
    """
    names = [
        str(getattr(ds, "PatientName", "") or ""),
        str(getattr(ds, "ReferringPhysicianName", "") or ""),
        str(getattr(ds, "PerformingPhysicianName", "") or ""),
        str(getattr(ds, "NameOfPhysiciansReadingStudy", "") or ""),
    ]
    ids = [
        str(getattr(ds, "PatientID", "") or ""),
        str(getattr(ds, "OtherPatientIDs", "") or ""),
        str(getattr(ds, "AccessionNumber", "") or ""),
        str(getattr(ds, "PatientTelephoneNumbers", "") or ""),
    ]
    institutions = [
        str(getattr(ds, "InstitutionName", "") or ""),
        str(getattr(ds, "InstitutionAddress", "") or ""),
        str(getattr(ds, "StationName", "") or ""),
    ]
    return (
        [v for v in names if v],
        [v for v in ids if v],
        [v for v in institutions if v],
    )


def process_reports(
    src: Path,
    workspace: Workspace,
    *,
    key: bytes | None = None,
    allow_create: bool = True,
) -> ReportResult:
    """Scrub and label every report paired with a de-identified object."""
    src = Path(src)
    workspace.ensure()
    pseudo = Pseudonymiser(key or load_or_create_key(workspace.key_file, allow_create=allow_create))

    raw_by_source = {row["source_path"]: row for row in read_jsonl(workspace.raw_manifest)}
    result = ReportResult()

    for row in read_jsonl(workspace.deid_manifest):
        raw = raw_by_source.get(row["source_path"])
        if raw is None or not raw.get("report_path"):
            continue

        report_path = src / raw["report_path"]
        if not report_path.exists():
            continue

        text = report_path.read_text(encoding="utf-8")
        ds = pydicom.dcmread(src / row["source_path"], stop_before_pixels=True)
        names, ids, institutions = _identifiers_from_source(ds)

        scrubbed = scrub_report(
            text,
            known_names=names,
            known_ids=ids,
            known_institutions=institutions,
            pseudonymiser=pseudo,
            pseudo_id=row["pseudo_patient_id"],
        )

        sections = parse_sections(scrubbed.text)
        labels = extract_labels(clinical_text(sections))

        relative = f"{row['site_id']}/{row['study_uid']}.txt"
        out_path = workspace.reports_dir / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(scrubbed.text, encoding="utf-8", newline="\n")

        result.records.append(
            ReportRecord(
                study_uid=row["study_uid"],
                site_id=row["site_id"],
                relative_path=relative,
                sections={s.value: t for s, t in sections.items()},
                findings=[label.finding.value for label in labels if label.present],
                denied_findings=[
                    label.finding.value for label in labels if not label.present
                ],
                redaction_count=scrubbed.redaction_count,
                source_path=row["source_path"],
            )
        )

    result.records.sort(key=lambda r: (r.site_id, r.relative_path))
    write_jsonl(workspace.reports_manifest, (r.to_dict() for r in result.records))
    return result


__all__ = ["ReportRecord", "ReportResult", "ReportSection", "process_reports"]
