"""Catalogue, access control, and quality control.

Half of these tests deliberately break the cohort first. A check that has never
been observed to fail is not evidence of anything, and a QC suite that only ever
reports "all clear" on healthy data is the easiest kind of reassurance to ship by
accident.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from cxr_harmony.catalog import (
    AccessDenied,
    Role,
    build_catalog,
    open_engine,
    query_report_text,
    query_studies,
    record_splits,
    summary_counts,
)
from cxr_harmony.catalog.schema_sql import StudyRow
from cxr_harmony.deid import deidentify
from cxr_harmony.harmonize import harmonize
from cxr_harmony.ingest import ingest
from cxr_harmony.qc import Severity, render_markdown, run_checks, write_report
from cxr_harmony.reports import process_reports
from cxr_harmony.schema.models import CanonicalDataset, Label, Patient, Series, Study
from cxr_harmony.schema.vocab import Finding, LabelSource, Sex, ViewPosition
from cxr_harmony.workspace import read_jsonl

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "sites"
KEY = b"catalog-stage-test-key-32-bytes!"
UID = "2.25.1234567890123456789012345678"


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    from cxr_harmony.synth import generate_corpus
    from cxr_harmony.workspace import Workspace

    src = tmp_path_factory.mktemp("src")
    truth = generate_corpus(src, seed=717, n_patients=32, n_cross_site=6, image_size=128)
    ws = Workspace(tmp_path_factory.mktemp("ws") / "work").ensure()
    ingest(src, ws)
    deidentify(src, ws, key=KEY)
    process_reports(src, ws, key=KEY)
    result = harmonize(src, ws, CONFIGS)
    stats = build_catalog(result.dataset, ws)
    return ws, truth, result, stats


# --- Catalogue --------------------------------------------------------------


def test_catalogue_counts_match_the_dataset(cohort):
    _, _, result, stats = cohort
    assert stats.n_patients == len(result.dataset.patients)
    assert stats.n_studies == len(result.dataset.studies)
    assert stats.n_instances == len(result.dataset.instances)


def test_catalogue_is_rebuilt_not_appended(cohort):
    """Re-running a stage must not double every row."""
    ws, _, result, stats = cohort
    again = build_catalog(result.dataset, ws)
    assert again.n_studies == stats.n_studies


def test_foreign_keys_are_enforced(cohort):
    """SQLite ignores them unless asked, which is how orphan rows accumulate."""
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import Session

    ws, _, _, _ = cohort
    engine = open_engine(ws.catalog_db)
    with Session(engine, future=True) as session:
        session.add(
            StudyRow(
                study_uid=UID,
                pseudo_patient_id="ffffffffffffffff",  # no such patient
                site_id="SITE_A",
                study_date=date(2024, 1, 1),
                modality="DX",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_splits_are_recorded_per_release(cohort):
    ws, _, result, _ = cohort
    assignments = {p.pseudo_id: "train" for p in result.dataset.patients}
    assert record_splits(ws, assignments, "v1.0.0") == len(assignments)
    assert record_splits(ws, assignments, "v1.0.0") == len(assignments)


# --- Access control ---------------------------------------------------------


def test_curator_may_read_report_text(cohort):
    ws, _, result, _ = cohort
    study_uid = result.dataset.reports[0].study_uid
    assert isinstance(query_report_text(ws, Role.CURATOR, study_uid), dict)


def test_modeller_is_refused_report_text(cohort):
    """Scrubbed prose is the highest residual re-identification surface left."""
    ws, _, result, _ = cohort
    study_uid = result.dataset.reports[0].study_uid
    with pytest.raises(AccessDenied):
        query_report_text(ws, Role.MODELLER, study_uid)


def test_modeller_may_read_studies(cohort):
    ws, _, _, stats = cohort
    assert len(query_studies(ws, Role.MODELLER)) == stats.n_studies


def test_auditor_is_refused_patient_level_rows(cohort):
    ws, _, _, _ = cohort
    with pytest.raises(AccessDenied):
        query_studies(ws, Role.AUDITOR)


def test_auditor_may_still_read_aggregates(cohort):
    """Governance oversight must not require access to the rows themselves."""
    ws, _, _, stats = cohort
    assert summary_counts(ws, Role.AUDITOR)["studies"] == stats.n_studies


def test_site_filter_narrows_the_query(cohort):
    ws, _, _, stats = cohort
    rows = query_studies(ws, Role.CURATOR, site_id="SITE_A")
    assert rows
    assert {r["site_id"] for r in rows} == {"SITE_A"}
    assert len(rows) == stats.per_site["SITE_A"]


# --- QC on a healthy cohort -------------------------------------------------


def test_a_clean_cohort_passes(cohort):
    ws, _, result, _ = cohort
    unmapped = json.loads((ws.root / "unmapped_values.json").read_text(encoding="utf-8"))
    report = run_checks(
        result.dataset, unmapped=unmapped, quarantined=list(read_jsonl(ws.quarantine))
    )
    assert report.passed, [c.message for c in report.failures]
    assert report.warnings == [], [c.message for c in report.warnings]


def test_report_is_written_in_both_formats(cohort):
    ws, _, result, _ = cohort
    md_path, json_path = write_report(run_checks(result.dataset), ws)
    assert md_path.exists() and json_path.exists()
    json.loads(json_path.read_text(encoding="utf-8"))
    assert "# Cohort quality control" in md_path.read_text(encoding="utf-8")


def test_cross_site_linkage_is_reported(cohort):
    _, truth, result, _ = cohort
    report = run_checks(result.dataset)
    check = next(c for c in report.checks if c.name == "cross_site_patients_linked")
    assert str(len(truth["cross_site_patients"])) in check.message


def test_markdown_leads_with_problems_when_there_are_any():
    dataset = CanonicalDataset(
        patients=[Patient(pseudo_id="a" * 16, sex=Sex.UNKNOWN)],
        studies=[
            Study(
                study_uid=UID,
                pseudo_patient_id="b" * 16,  # orphan
                site_id="SITE_A",
                modality="DX",
            )
        ],
    )
    markdown = render_markdown(run_checks(dataset))
    assert "## Failures" in markdown
    assert markdown.index("## Failures") < markdown.index("## Cohort")


# --- QC must be able to fail ------------------------------------------------


def test_orphan_study_is_caught():
    dataset = CanonicalDataset(
        patients=[Patient(pseudo_id="a" * 16)],
        studies=[
            Study(study_uid=UID, pseudo_patient_id="b" * 16, site_id="SITE_A", modality="DX")
        ],
    )
    report = run_checks(dataset)
    assert not report.passed
    assert any(c.name == "study_patient_integrity" for c in report.failures)


def test_orphan_series_is_caught():
    dataset = CanonicalDataset(
        series=[Series(series_uid=UID, study_uid="2.25.999", view_position=ViewPosition.PA)]
    )
    report = run_checks(dataset)
    assert any(c.name == "series_study_integrity" for c in report.failures)


def test_orphan_label_is_caught():
    dataset = CanonicalDataset(
        labels=[
            Label(
                study_uid=UID,
                finding=Finding.CARDIOMEGALY,
                present=True,
                source=LabelSource.SITE_STRUCTURED,
            )
        ]
    )
    report = run_checks(dataset)
    assert any(c.name == "label_study_integrity" for c in report.failures)


def test_study_without_images_is_caught():
    dataset = CanonicalDataset(
        patients=[Patient(pseudo_id="a" * 16)],
        studies=[
            Study(study_uid=UID, pseudo_patient_id="a" * 16, site_id="SITE_A", modality="DX")
        ],
    )
    report = run_checks(dataset)
    assert any(c.name == "studies_have_images" for c in report.failures)


def test_unresolved_projection_is_warned_about():
    dataset = CanonicalDataset(
        patients=[Patient(pseudo_id="a" * 16)],
        studies=[
            Study(study_uid=UID, pseudo_patient_id="a" * 16, site_id="SITE_A", modality="DX")
        ],
        series=[Series(series_uid="2.25.5", study_uid=UID)],  # defaults to UNKNOWN
    )
    report = run_checks(dataset)
    assert any(c.name == "projection_resolved" for c in report.warnings)


def test_unmapped_values_are_surfaced_as_a_warning():
    report = run_checks(
        CanonicalDataset(),
        unmapped=[{"site_id": "SITE_B", "field": "label", "value": "XYZ", "count": 4}],
    )
    check = next(c for c in report.checks if c.name == "all_values_mapped")
    assert not check.passed
    assert "4" in check.message


def test_projection_imbalance_across_sites_is_detected():
    """The confound this project exists to make visible."""
    patients = [Patient(pseudo_id=f"{i:016x}") for i in range(2)]
    studies, series = [], []
    for i in range(20):
        # Site A sends only erect PA films; site B only portable AP.
        site = "SITE_A" if i < 10 else "SITE_B"
        view = ViewPosition.PA if site == "SITE_A" else ViewPosition.AP
        uid = f"2.25.{1000 + i}"
        studies.append(
            Study(
                study_uid=uid,
                pseudo_patient_id=patients[i % 2].pseudo_id,
                site_id=site,
                modality="DX",
            )
        )
        series.append(Series(series_uid=f"2.25.{2000 + i}", study_uid=uid, view_position=view))

    report = run_checks(CanonicalDataset(patients=patients, studies=studies, series=series))
    check = next(c for c in report.checks if c.name == "projection_balance_across_sites")
    assert not check.passed
    assert check.severity is Severity.WARN
    assert check.detail["ap_fraction_by_site"] == {"SITE_A": 0.0, "SITE_B": 1.0}


def test_duplicate_pixel_content_is_detected():
    from cxr_harmony.schema.models import Instance

    shared = "c" * 64
    dataset = CanonicalDataset(
        instances=[
            Instance(
                sop_uid=f"2.25.{i}",
                series_uid="2.25.1",
                relative_path=f"x{i}.dcm",
                sha256=shared,
                rows=8,
                columns=8,
                bits_stored=12,
                photometric_interpretation="MONOCHROME2",
            )
            for i in (1, 2)
        ]
    )
    report = run_checks(dataset)
    assert any(c.name == "no_duplicate_pixel_content" for c in report.warnings)


def test_quarantine_attrition_is_reported_when_present():
    report = run_checks(
        CanonicalDataset(), quarantined=[{"reason": "WRONG_MODALITY"}, {"reason": "NOT_DICOM"}]
    )
    check = next(c for c in report.checks if c.name == "ingest_attrition_accounted")
    assert check.detail["by_reason"] == {"NOT_DICOM": 1, "WRONG_MODALITY": 1}
