"""FHIR export.

The identifier tests matter most. FHIR's data model has a natural place for a
patient's name in almost every resource, so an export is the easiest way to
reintroduce everything the de-identification stage removed.
"""

from __future__ import annotations

import json
import re
from datetime import date

import pytest

from cxr_harmony.interop.fhir import (
    FINDING_CODES,
    LOCAL_SYSTEM,
    build_bundle,
    diagnostic_report_resource,
    imaging_study_resource,
    observation_resources,
    write_bundle,
)
from cxr_harmony.schema.models import (
    CanonicalDataset,
    Instance,
    Label,
    Patient,
    Report,
    Series,
    Study,
)
from cxr_harmony.schema.vocab import (
    Finding,
    LabelSource,
    ReportSection,
    Sex,
    ViewPosition,
)

UID = "2.25.1234567890"
PID = "a1b2c3d4e5f60718"


@pytest.fixture
def dataset() -> CanonicalDataset:
    return CanonicalDataset(
        patients=[Patient(pseudo_id=PID, sex=Sex.FEMALE, age_years=54)],
        studies=[
            Study(
                study_uid=UID,
                pseudo_patient_id=PID,
                site_id="SITE_A",
                study_date=date(2024, 3, 14),
                modality="DX",
                body_part="CHEST",
            )
        ],
        series=[
            Series(series_uid="2.25.99", study_uid=UID, view_position=ViewPosition.PA)
        ],
        instances=[
            Instance(
                sop_uid="2.25.111",
                series_uid="2.25.99",
                relative_path="SITE_A/x.dcm",
                sha256="a" * 64,
                rows=512,
                columns=512,
                bits_stored=12,
                photometric_interpretation="MONOCHROME2",
            )
        ],
        reports=[
            Report(
                study_uid=UID,
                sections={
                    ReportSection.FINDINGS: "The cardiac silhouette is enlarged.",
                    ReportSection.IMPRESSION: "Cardiomegaly.",
                },
                redaction_count=4,
            )
        ],
        labels=[
            Label(
                study_uid=UID,
                finding=Finding.CARDIOMEGALY,
                present=True,
                source=LabelSource.SITE_STRUCTURED,
                site_native_value="CM",
            )
        ],
    )


# --- Identifiers must not come back -----------------------------------------


def test_no_patient_resource_is_emitted(dataset):
    """A Patient resource is a container designed to hold a name; offering one
    invites the next person to populate it."""
    bundle, _ = build_bundle(dataset)
    types = {e["resource"]["resourceType"] for e in bundle["entry"]}
    assert "Patient" not in types


def test_subject_carries_the_pseudonym_and_no_display_name(dataset):
    """FHIR permits a human-readable label on a reference and every viewer shows
    it, which makes it exactly where a name ends up."""
    resource = imaging_study_resource(dataset, UID)
    subject = resource["subject"]
    assert subject["identifier"]["value"] == PID
    assert "display" not in subject


def test_no_identifier_shaped_field_appears_anywhere(dataset):
    bundle, _ = build_bundle(dataset)
    text = json.dumps(bundle).lower()
    for banned in ("patientname", "birthdate", "accession", "\"mrn\"", "address"):
        assert banned not in text, banned


def test_only_the_impression_is_exported_not_the_findings_prose(dataset):
    """Findings prose is the richest residual re-identification surface left."""
    report = diagnostic_report_resource(dataset, UID, [])
    assert report["conclusion"] == "Cardiomegaly."
    assert "cardiac silhouette" not in json.dumps(report)


# --- Structure --------------------------------------------------------------


def test_imaging_study_required_fields(dataset):
    resource = imaging_study_resource(dataset, UID)
    assert resource["resourceType"] == "ImagingStudy"
    assert resource["status"] == "available"
    assert resource["subject"]
    assert resource["identifier"][0]["value"] == f"urn:oid:{UID}"
    assert resource["numberOfSeries"] == 1
    assert resource["numberOfInstances"] == 1
    assert resource["series"][0]["instance"][0]["uid"] == "2.25.111"


def test_diagnostic_report_links_study_and_observations(dataset):
    observations = observation_resources(dataset, UID)
    report = diagnostic_report_resource(dataset, UID, [o["id"] for o in observations])
    assert report["imagingStudy"][0]["reference"].startswith("ImagingStudy/")
    assert report["result"][0]["reference"].startswith("Observation/")
    assert len(report["result"]) == len(observations)


def test_observation_carries_a_code_and_a_value(dataset):
    observation = observation_resources(dataset, UID)[0]
    coding = observation["code"]["coding"][0]
    assert coding["code"] == FINDING_CODES[Finding.CARDIOMEGALY][1]
    assert observation["valueBoolean"] is True
    assert observation["status"] == "final"


def test_label_provenance_travels_with_the_observation(dataset):
    """A rule-extracted label is weaker evidence than a site's export, and a
    consumer that cannot tell them apart treats them as equal."""
    observation = observation_resources(dataset, UID)[0]
    method = observation["method"]["coding"][0]
    assert method["code"] == "SITE_STRUCTURED"


def test_out_of_vocabulary_findings_use_a_local_code_not_a_guessed_snomed():
    """A wrong SNOMED code looks authoritative to everything downstream."""
    system, code, _ = FINDING_CODES[Finding.OTHER]
    assert system == LOCAL_SYSTEM
    assert code == "OTHER"


def test_bundle_is_a_valid_collection(dataset):
    bundle, stats = build_bundle(dataset)
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert all("fullUrl" in e and "resource" in e for e in bundle["entry"])
    assert stats.n_studies == 1
    assert stats.n_reports == 1
    assert stats.n_observations == 1


def test_a_study_without_a_report_still_exports(dataset):
    stripped = CanonicalDataset(
        patients=dataset.patients,
        studies=dataset.studies,
        series=dataset.series,
        instances=dataset.instances,
        labels=dataset.labels,
    )
    _, stats = build_bundle(stripped)
    assert stats.n_reports == 0
    assert stats.n_studies == 1


def test_fhir_ids_are_legal(dataset):
    bundle, _ = build_bundle(dataset)
    for entry in bundle["entry"]:
        fhir_id = entry["resource"]["id"]
        assert re.fullmatch(r"[A-Za-z0-9\-.]{1,64}", fhir_id), fhir_id


def test_bundle_is_written_and_reloadable(dataset, tmp_path):
    path, stats = write_bundle(dataset, tmp_path)
    assert path.exists()
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["resourceType"] == "Bundle"
    assert len(reloaded["entry"]) == stats.to_dict()["n_entries"]


def test_export_is_deterministic(dataset, tmp_path):
    first, _ = write_bundle(dataset, tmp_path / "a")
    second, _ = write_bundle(dataset, tmp_path / "b")
    assert first.read_bytes() == second.read_bytes()
