"""Mapping three sites' conventions onto one canonical dataset."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from ..schema.models import (
    CanonicalDataset,
    Instance,
    Label,
    Patient,
    Report,
    Series,
    Study,
)
from ..schema.vocab import (
    Finding,
    LabelSource,
    Laterality,
    PatientPosition,
    ReportSection,
    Sex,
    ViewPosition,
)
from ..workspace import Workspace, read_jsonl
from .config import SiteConfig, load_site_configs


@dataclass
class UnmappedValue:
    """A site-native string the config did not account for."""

    site_id: str
    field: str
    value: str
    count: int

    def to_dict(self) -> dict:
        return {
            "site_id": self.site_id,
            "field": self.field,
            "value": self.value,
            "count": self.count,
        }


@dataclass
class HarmonizeResult:
    dataset: CanonicalDataset
    unmapped: list[UnmappedValue] = field(default_factory=list)
    #: Per-site counts of studies contributed, for the QC report.
    per_site: dict[str, int] = field(default_factory=dict)

    @property
    def n_studies(self) -> int:
        return len(self.dataset.studies)

    @property
    def n_unmapped(self) -> int:
        return sum(u.count for u in self.unmapped)


def _parse_date(text: str, formats: list[str]) -> date | None:
    text = (text or "").strip()
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _load_sidecar(src: Path, site_id: str, config: SiteConfig) -> dict[str, str]:
    """Read a site's label CSV into ``{join_key: raw_value}``."""
    if config.labels.source != "sidecar_csv" or not config.labels.sidecar_file:
        return {}
    path = src / site_id / config.labels.sidecar_file
    if not path.exists():
        return {}

    key_col = config.labels.sidecar_key_column or ""
    value_col = config.labels.sidecar_value_column or ""
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row.get(key_col) or "").strip()
            if key:
                out[key] = (row.get(value_col) or "").strip()
    return out


def harmonize(
    src: Path,
    workspace: Workspace,
    configs_dir: Path,
) -> HarmonizeResult:
    """Build the canonical dataset from the de-identified store and site facts."""
    src = Path(src)
    configs = load_site_configs(configs_dir)

    facts_by_source = {
        row["source_path"]: row for row in read_jsonl(workspace.root / "site_facts.jsonl")
    }
    reports_by_study = {
        row["study_uid"]: row for row in read_jsonl(workspace.reports_manifest)
    }

    sidecars = {
        site_id: _load_sidecar(src, site_id, config) for site_id, config in configs.items()
    }

    unmapped_counter: Counter[tuple[str, str, str]] = Counter()
    per_site: Counter[str] = Counter()

    patients: dict[str, Patient] = {}
    studies: dict[str, Study] = {}
    series: dict[str, Series] = {}
    instances: list[Instance] = []
    reports: list[Report] = []
    labels: list[Label] = []

    for row in read_jsonl(workspace.deid_manifest):
        site_id = row["site_id"]
        config = configs.get(site_id)
        if config is None:
            raise KeyError(
                f"no adapter config for site {site_id!r}; add configs/sites/{site_id.lower()}.yaml"
            )

        facts = facts_by_source.get(row["source_path"], {})

        def resolve(mapping, value: str, field_name: str, _site: str = site_id) -> str | None:
            """Resolve one value, counting it when the config had no answer.

            ``_site`` is bound at definition time rather than closed over: the
            closure is only called within this iteration today, but a later edit
            that defers a call would otherwise attribute the count to the wrong site.
            """
            resolved, matched = mapping.resolve(value)
            if not matched and (value or "").strip():
                unmapped_counter[(_site, field_name, value.strip())] += 1
            return resolved

        # --- patient
        sex_value = resolve(config.sex, facts.get("sex_native", ""), "sex") or "U"
        pseudo_id = row["pseudo_patient_id"]
        age = facts.get("age_years")
        if pseudo_id not in patients:
            patients[pseudo_id] = Patient(
                pseudo_id=pseudo_id,
                sex=Sex(sex_value),
                age_years=age,
            )

        # --- study
        study_uid = row["study_uid"]
        if study_uid not in studies:
            studies[study_uid] = Study(
                study_uid=study_uid,
                pseudo_patient_id=pseudo_id,
                site_id=site_id,
                study_date=_parse_date(facts.get("study_date_native", ""), config.date_formats),
                modality=facts.get("modality", ""),
                body_part=facts.get("body_part") or None,
                patient_position=PatientPosition.UNKNOWN,
            )
            per_site[site_id] += 1

        # --- series
        series_uid = row["series_uid"]
        if series_uid not in series:
            view = resolve(config.view_position, facts.get("view_native", ""), "view_position")
            lat = resolve(config.laterality, facts.get("laterality_native", ""), "laterality")
            series[series_uid] = Series(
                series_uid=series_uid,
                study_uid=study_uid,
                view_position=ViewPosition(view or "UNKNOWN"),
                laterality=Laterality(lat or "NA"),
            )

        # --- instance
        instances.append(
            Instance(
                sop_uid=row["sop_uid"],
                series_uid=series_uid,
                relative_path=row["relative_path"],
                sha256=row["sha256"],
                rows=row["rows"],
                columns=row["columns"],
                bits_stored=row["bits_stored"],
                photometric_interpretation=row["photometric_interpretation"],
                pixel_redacted=row["pixel_redacted"],
            )
        )

        # --- labels
        for label in _labels_for_study(
            study_uid=study_uid,
            site_id=site_id,
            config=config,
            facts=facts,
            sidecar=sidecars.get(site_id, {}),
            report_row=reports_by_study.get(study_uid),
            unmapped_counter=unmapped_counter,
        ):
            labels.append(label)

    # --- reports
    seen_reports: set[str] = set()
    for row in read_jsonl(workspace.reports_manifest):
        if row["study_uid"] in seen_reports:
            continue
        seen_reports.add(row["study_uid"])
        reports.append(
            Report(
                study_uid=row["study_uid"],
                sections={ReportSection(k): v for k, v in row["sections"].items()},
                scrubbed=True,
                redaction_count=row["redaction_count"],
            )
        )

    dataset = CanonicalDataset(
        patients=sorted(patients.values(), key=lambda p: p.pseudo_id),
        studies=sorted(studies.values(), key=lambda s: s.study_uid),
        series=sorted(series.values(), key=lambda s: s.series_uid),
        instances=sorted(instances, key=lambda i: i.sop_uid),
        reports=sorted(reports, key=lambda r: r.study_uid),
        labels=sorted(labels, key=lambda label: (label.study_uid, label.finding.value)),
    )

    unmapped = [
        UnmappedValue(site_id=s, field=f, value=v, count=n)
        for (s, f, v), n in sorted(unmapped_counter.items())
    ]

    result = HarmonizeResult(dataset=dataset, unmapped=unmapped, per_site=dict(per_site))
    workspace.canonical.write_text(
        dataset.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (workspace.root / "unmapped_values.json").write_text(
        json.dumps([u.to_dict() for u in unmapped], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _labels_for_study(
    *,
    study_uid: str,
    site_id: str,
    config: SiteConfig,
    facts: dict,
    sidecar: dict[str, str],
    report_row: dict | None,
    unmapped_counter: Counter,
) -> list[Label]:
    """Read this site's labels from whichever channel it uses."""
    source = config.labels.source

    if source == "report":
        # Rule-extracted from prose, and recorded as the weaker evidence it is.
        if report_row is None:
            return []
        return [
            Label(
                study_uid=study_uid,
                finding=Finding(value),
                present=True,
                source=LabelSource.REPORT_RULE,
                site_native_value=None,
            )
            for value in report_row.get("findings", [])
        ]

    if source == "image_comments":
        raw = facts.get("labels_native", "")
    elif source == "sidecar_csv":
        raw = sidecar.get(facts.get("accession_key", ""), "")
    else:  # pragma: no cover - guarded by config validation
        raw = ""

    if not raw:
        return []

    out: list[Label] = []
    for token in raw.split(config.labels.separator):
        token = token.strip()
        if not token:
            continue
        canonical = None
        for native, value in config.labels.map.items():
            if native.strip().upper() == token.upper():
                canonical = value
                break
        if canonical is None:
            # Kept as OTHER rather than dropped, so the loss is visible in QC.
            unmapped_counter[(site_id, "label", token)] += 1
            canonical = Finding.OTHER.value
        out.append(
            Label(
                study_uid=study_uid,
                finding=Finding(canonical),
                present=True,
                source=LabelSource.SITE_STRUCTURED,
                site_native_value=token,
            )
        )
    return out


__all__ = ["HarmonizeResult", "UnmappedValue", "harmonize"]
