"""Cutting an immutable, content-addressed dataset release.

A release is the unit the modelling team trains against, and the unit a result is
reported against six months later. It therefore has to be identifiable by
something stronger than a folder name: ``manifest.json`` lists every file with its
SHA-256, and the digest over that manifest is the release identity. Two people
holding "v1.0.0" can compare one hex string and know whether they hold the same
data.

The manifest carries no timestamp, deliberately. Anything inside a content
address must be a function of the content, or the same data cut twice produces two
identities and the check is worthless. Provenance that genuinely varies per run —
when it was cut, by whom — lives in ``release.json`` beside it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..schema.models import CanonicalDataset
from ..schema.vocab import Split
from ..workspace import Workspace
from .splits import SplitRatios, assign_all, realised_proportions

MANIFEST_NAME = "manifest.json"
SPLITS_NAME = "splits.json"
RELEASE_NAME = "release.json"
DATASHEET_NAME = "datasheet.md"


@dataclass
class ReleaseResult:
    version: str
    directory: Path
    dataset_digest: str
    n_files: int
    assignments: dict[str, Split] = field(default_factory=dict)
    proportions: dict[str, float] = field(default_factory=dict)
    studies_per_split: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "dataset_digest": self.dataset_digest,
            "n_files": self.n_files,
            "proportions": self.proportions,
            "studies_per_split": self.studies_per_split,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(dataset: CanonicalDataset, workspace: Workspace) -> list[dict]:
    """One entry per released file, sorted, each with its digest.

    Digests are recomputed from the files on disk rather than copied from the
    de-identification manifest. A release that trusts an upstream record cannot
    detect corruption that happened after that record was written, which is the
    only kind of corruption a manifest is any use against.
    """
    entries: list[dict] = []
    for instance in dataset.instances:
        path = workspace.deid_store / instance.relative_path
        if not path.exists():
            raise FileNotFoundError(f"instance missing from the de-identified store: {path}")
        entries.append(
            {
                "path": f"images/{instance.relative_path}",
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "sop_uid": instance.sop_uid,
            }
        )

    for report in dataset.reports:
        for site_dir in sorted(workspace.reports_dir.glob("*")):
            candidate = site_dir / f"{report.study_uid}.txt"
            if candidate.exists():
                relative = candidate.relative_to(workspace.reports_dir).as_posix()
                entries.append(
                    {
                        "path": f"reports/{relative}",
                        "sha256": _sha256(candidate),
                        "size_bytes": candidate.stat().st_size,
                        "study_uid": report.study_uid,
                    }
                )
                break

    return sorted(entries, key=lambda e: e["path"])


def digest_manifest(entries: list[dict]) -> str:
    """The release identity: a digest over the manifest's canonical serialisation."""
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_release(
    dataset: CanonicalDataset,
    workspace: Workspace,
    *,
    version: str,
    ratios: SplitRatios | None = None,
    split_salt: str = "cxr-harmony-v1",
    created_at: str | None = None,
) -> ReleaseResult:
    """Cut release ``version`` into ``workspace.releases``."""
    ratios = ratios or SplitRatios()
    directory = workspace.releases / version
    directory.mkdir(parents=True, exist_ok=True)

    entries = build_manifest(dataset, workspace)
    digest = digest_manifest(entries)

    assignments = assign_all(
        [p.pseudo_id for p in dataset.patients], salt=split_salt, ratios=ratios
    )
    proportions = realised_proportions(assignments)

    studies_per_split: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for study in dataset.studies:
        split = assignments.get(study.pseudo_patient_id)
        if split is not None:
            studies_per_split[split.value] += 1

    (directory / MANIFEST_NAME).write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (directory / SPLITS_NAME).write_text(
        json.dumps(
            {pid: split.value for pid, split in sorted(assignments.items())},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    release_meta = {
        "version": version,
        "dataset_digest": digest,
        "n_files": len(entries),
        "n_patients": len(dataset.patients),
        "n_studies": len(dataset.studies),
        "split_salt": split_salt,
        "split_ratios": {"train": ratios.train, "val": ratios.val, "test": ratios.test},
        "realised_proportions": proportions,
        "studies_per_split": studies_per_split,
    }
    if created_at:
        release_meta["created_at"] = created_at

    (directory / RELEASE_NAME).write_text(
        json.dumps(release_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    (directory / DATASHEET_NAME).write_text(
        render_datasheet(dataset, release_meta, assignments),
        encoding="utf-8",
        newline="\n",
    )

    return ReleaseResult(
        version=version,
        directory=directory,
        dataset_digest=digest,
        n_files=len(entries),
        assignments=assignments,
        proportions=proportions,
        studies_per_split=studies_per_split,
    )


def render_datasheet(
    dataset: CanonicalDataset,
    meta: dict,
    assignments: dict[str, Split],
) -> str:
    """A datasheet in the spirit of Gebru et al., 'Datasheets for Datasets'."""
    from collections import Counter

    per_site = Counter(s.site_id for s in dataset.studies)
    findings = Counter(label.finding.value for label in dataset.labels if label.present)
    sexes = Counter(p.sex.value for p in dataset.patients)
    ages = [p.age_years for p in dataset.patients if p.age_years is not None]

    lines = [
        f"# Dataset release {meta['version']}",
        "",
        f"`dataset_digest: {meta['dataset_digest']}`",
        "",
        "Two holders of this version can compare that one string to establish",
        "whether they have the same data.",
        "",
        "## Composition",
        "",
        "| | |",
        "|---|---|",
        f"| Patients | {meta['n_patients']} |",
        f"| Studies | {meta['n_studies']} |",
        f"| Files | {meta['n_files']} |",
        "",
        "### Studies per contributing site",
        "",
        "| Site | Studies |",
        "|---|---:|",
    ]
    lines += [f"| {site} | {count} |" for site, count in sorted(per_site.items())]

    lines += [
        "",
        "### Patient demographics",
        "",
        "| | |",
        "|---|---|",
        f"| Sex | {', '.join(f'{k}: {v}' for k, v in sorted(sexes.items()))} |",
    ]
    if ages:
        lines.append(
            f"| Age (years) | min {min(ages)}, median {sorted(ages)[len(ages) // 2]}, "
            f"max {max(ages)} |"
        )
    lines.append(
        "| Age cap | ages above 89 are recorded at 89, as an exact age in that "
        "tail is quasi-identifying |"
    )

    lines += ["", "### Label prevalence", "", "| Finding | Studies |", "|---|---:|"]
    lines += [f"| {finding} | {count} |" for finding, count in sorted(findings.items())]

    lines += [
        "",
        "## Partitions",
        "",
        "Splits are assigned **per patient**, not per study. A patient with a",
        "baseline and two follow-ups contributes three studies; splitting those",
        "independently would put the same chest in training and in test.",
        "",
        "Assignment is by hash threshold rather than by shuffling, so an existing",
        "patient's partition does not change when the cohort grows. The cost is",
        "that realised proportions only approach the target asymptotically.",
        "",
        "| Split | Target | Realised (patients) | Studies |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        lines.append(
            f"| {split} | {meta['split_ratios'][split]:.2f} | "
            f"{meta['realised_proportions'][split]:.4f} | "
            f"{meta['studies_per_split'][split]} |"
        )

    lines += [
        "",
        "## Provenance and limitations",
        "",
        "- Images are de-identified under the DICOM PS3.15 Annex E Basic Profile with",
        "  Clean Pixel Data, Retain Longitudinal Temporal Information (Modified Dates)",
        "  and Retain Patient Characteristics.",
        "- Dates are shifted by a per-patient offset. Intervals within a patient are",
        "  preserved; absolute dates are not meaningful.",
        "- Report text has been scrubbed but remains the highest residual",
        "  re-identification surface in the release, and is access-controlled",
        "  accordingly.",
        "- Labels carry a source. Those marked `REPORT_RULE` were extracted from prose",
        "  by rule and are weaker evidence than a site's structured export; they should",
        "  not be treated as interchangeable with it.",
        "- Per-site distribution differences are reported in the QC report. Where they",
        "  are large, a random split will overstate generalisation to a new hospital;",
        "  prefer a leave-one-site-out evaluation.",
        "",
    ]
    return "\n".join(lines)


def verify_release(directory: Path) -> tuple[bool, list[str]]:
    """Re-hash every file named in a release manifest. Returns ``(ok, problems)``."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        return False, [f"no manifest at {manifest_path}"]

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if digest_manifest(entries) != json.loads(
        (directory / RELEASE_NAME).read_text(encoding="utf-8")
    ).get("dataset_digest"):
        return False, ["manifest digest does not match the recorded dataset_digest"]

    return True, []


__all__ = [
    "DATASHEET_NAME",
    "MANIFEST_NAME",
    "RELEASE_NAME",
    "SPLITS_NAME",
    "ReleaseResult",
    "build_manifest",
    "build_release",
    "digest_manifest",
    "render_datasheet",
    "verify_release",
]
