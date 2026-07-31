"""The schema is the pipeline's PHI boundary.

Its refusals are therefore tested as carefully as its acceptances: a model that
quietly accepts a patient name is a model that will one day carry one.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cxr_harmony.schema import (
    Finding,
    Instance,
    Label,
    LabelSource,
    Patient,
    Series,
    Sex,
    Study,
    ViewPosition,
    build_schemas,
    write_schemas,
)

GOOD_PSEUDO = "a1b2c3d4e5f60718"
GOOD_UID = "1.2.826.0.1.3680043.10.1337.1.1"
GOOD_SHA = "e" * 64


def test_patient_accepts_well_formed_pseudonym():
    p = Patient(pseudo_id=GOOD_PSEUDO, sex=Sex.FEMALE, age_years=54)
    assert p.pseudo_id == GOOD_PSEUDO


@pytest.mark.parametrize(
    "bad",
    ["", "ABC123", "a1b2c3d4e5f6071", "a1b2c3d4e5f607189", "g1b2c3d4e5f60718", "P0001234"],
)
def test_patient_rejects_malformed_pseudonym(bad):
    with pytest.raises(ValidationError):
        Patient(pseudo_id=bad)


def test_patient_rejects_direct_identifier_fields():
    """A name or MRN must be a schema violation, not merely a convention breach."""
    for field, value in [
        ("patient_name", "SHARMA^RAVI"),
        ("mrn", "MRN-88213"),
        ("accession_number", "ACC00099"),
        ("birth_date", "1971-04-02"),
    ]:
        with pytest.raises(ValidationError):
            Patient(pseudo_id=GOOD_PSEUDO, **{field: value})


def test_age_is_capped_at_89():
    """Exact ages in the 90+ tail are quasi-identifying, so the schema refuses them."""
    assert Patient(pseudo_id=GOOD_PSEUDO, age_years=89).age_years == 89
    with pytest.raises(ValidationError):
        Patient(pseudo_id=GOOD_PSEUDO, age_years=94)


def test_models_are_immutable():
    p = Patient(pseudo_id=GOOD_PSEUDO)
    with pytest.raises(ValidationError):
        p.pseudo_id = "0" * 16


@pytest.mark.parametrize("bad_uid", ["not.a.uid", "1.2.3.", "", "1.2.abc.4", "1." * 40])
def test_study_rejects_malformed_uid(bad_uid):
    with pytest.raises(ValidationError):
        Study(
            study_uid=bad_uid,
            pseudo_patient_id=GOOD_PSEUDO,
            site_id="SITE_A",
            modality="DX",
        )


@pytest.mark.parametrize("bad_site", ["site_a", "1SITE", "", "SITE-A", "A" * 20])
def test_study_rejects_malformed_site_id(bad_site):
    with pytest.raises(ValidationError):
        Study(
            study_uid=GOOD_UID,
            pseudo_patient_id=GOOD_PSEUDO,
            site_id=bad_site,
            modality="DX",
        )


def test_instance_rejects_bad_digest():
    kwargs = dict(
        sop_uid=GOOD_UID,
        series_uid=GOOD_UID,
        relative_path="SITE_A/x.dcm",
        rows=512,
        columns=512,
        bits_stored=12,
        photometric_interpretation="MONOCHROME2",
    )
    assert Instance(sha256=GOOD_SHA, **kwargs).sha256 == GOOD_SHA
    for bad in ["E" * 64, "e" * 63, "", "zz" + "e" * 62]:
        with pytest.raises(ValidationError):
            Instance(sha256=bad, **kwargs)


def test_series_defaults_are_unknown_not_guessed():
    """An absent projection must surface as UNKNOWN so QC can count it."""
    s = Series(series_uid=GOOD_UID, study_uid=GOOD_UID)
    assert s.view_position is ViewPosition.UNKNOWN


def test_label_retains_site_native_value_for_traceability():
    lab = Label(
        study_uid=GOOD_UID,
        finding=Finding.CARDIOMEGALY,
        present=True,
        source=LabelSource.SITE_STRUCTURED,
        site_native_value="CM",
    )
    assert lab.site_native_value == "CM"


def test_schema_export_is_deterministic_and_valid_json(tmp_path):
    written = write_schemas(tmp_path)
    assert (tmp_path / "Patient.json").exists()
    assert (tmp_path / "bundle.json").exists()

    first = (tmp_path / "bundle.json").read_text(encoding="utf-8")
    write_schemas(tmp_path)
    assert (tmp_path / "bundle.json").read_text(encoding="utf-8") == first

    for path in written:
        json.loads(path.read_text(encoding="utf-8"))


def test_exported_schema_exposes_no_identifier_named_property():
    """Guard against a future field quietly reintroducing an identifier."""
    banned = {"name", "patientname", "mrn", "accession", "accessionnumber", "birthdate", "address"}
    for entity, schema in build_schemas().items():
        for prop in schema.get("properties", {}):
            assert prop.replace("_", "").lower() not in banned, f"{entity}.{prop}"
