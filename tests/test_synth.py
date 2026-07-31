"""The generated corpus is the fixture every later stage is judged against.

If it silently stopped containing PHI, or stopped diverging between sites, the
de-identification and harmonisation tests would keep passing while proving
nothing. These tests exist to keep that from happening quietly.
"""

from __future__ import annotations

import json

import numpy as np
import pydicom
import pytest

from cxr_harmony.synth import SITES_BY_ID, burn_in_text, generate_corpus, synthesise_radiograph
from cxr_harmony.synth.identifiers import FEMALE_GIVEN_NAMES, MALE_GIVEN_NAMES, make_person_name
from cxr_harmony.synth.pixels import MAX_VALUE


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    truth = generate_corpus(out, seed=4242, n_patients=16, n_cross_site=4, image_size=128)
    return out, truth


def test_all_three_sites_deliver(corpus):
    out, _ = corpus
    for site_id in ("SITE_A", "SITE_B", "SITE_C"):
        assert list((out / site_id / "images").glob("*.dcm")), f"{site_id} delivered no images"
        assert list((out / site_id / "reports").glob("*.txt")), f"{site_id} delivered no reports"


def test_every_image_has_a_paired_report(corpus):
    out, _ = corpus
    for site_id in ("SITE_A", "SITE_B", "SITE_C"):
        images = {p.stem for p in (out / site_id / "images").glob("*.dcm")}
        reports = {p.stem for p in (out / site_id / "reports").glob("*.txt")}
        assert images == reports


def test_headers_carry_live_direct_identifiers(corpus):
    """De-identification must have something to remove, or it proves nothing."""
    out, _ = corpus
    path = next((out / "SITE_A" / "images").glob("*.dcm"))
    ds = pydicom.dcmread(path)
    assert str(ds.PatientName)
    assert ds.PatientID
    assert ds.PatientBirthDate
    assert ds.AccessionNumber
    assert ds.InstitutionName
    assert ds.PatientAddress
    assert ds.PatientTelephoneNumbers


def test_sites_disagree_on_projection_encoding(corpus):
    out, _ = corpus
    a = pydicom.dcmread(next((out / "SITE_A" / "images").glob("*.dcm")))
    b = pydicom.dcmread(next((out / "SITE_B" / "images").glob("*.dcm")))

    # Site A fills ViewPosition; site B leaves it empty and hides the projection
    # inside a free-text series description.
    assert a.ViewPosition
    assert not b.ViewPosition
    assert "CHEST" in b.SeriesDescription


def test_site_b_hides_the_study_date_in_a_private_block(corpus):
    out, _ = corpus
    ds = pydicom.dcmread(next((out / "SITE_B" / "images").glob("*.dcm")))
    assert ds.StudyDate == ""
    block = ds.private_block(0x0033, "MIMS RIS EXPORT")
    # DD-MM-YYYY, which no standard date parser will accept as a DICOM DA.
    assert block[0x01].value.count("-") == 2


def test_private_block_also_leaks_the_patient_name(corpus):
    """Justifies blanket private-tag removal rather than a curated allowlist."""
    out, _ = corpus
    ds = pydicom.dcmread(next((out / "SITE_B" / "images").glob("*.dcm")))
    block = ds.private_block(0x0033, "MIMS RIS EXPORT")
    assert block[0x02].value


def test_sites_disagree_on_sex_encoding(corpus):
    out, _ = corpus
    values = {}
    for site_id in ("SITE_A", "SITE_B", "SITE_C"):
        ds = pydicom.dcmread(next((out / site_id / "images").glob("*.dcm")))
        values[site_id] = str(ds.PatientSex)
    assert values["SITE_A"] in {"M", "F"}
    assert values["SITE_B"] in {"MALE", "FEMALE"}
    assert values["SITE_C"] in {"1", "2"}


def test_labels_arrive_through_three_different_channels(corpus):
    out, _ = corpus
    a = pydicom.dcmread(next((out / "SITE_A" / "images").glob("*.dcm")))
    assert a.ImageComments  # in the header

    assert (out / "SITE_B" / "labels.csv").exists()  # in a sidecar

    c = pydicom.dcmread(next((out / "SITE_C" / "images").glob("*.dcm")))
    assert "ImageComments" not in c  # nowhere but the report text
    assert not (out / "SITE_C" / "labels.csv").exists()


def test_cross_site_patients_share_a_national_id_but_not_an_mrn(corpus):
    """The premise of cross-site linkage: local MRNs cannot possibly match."""
    _, truth = corpus
    assert truth["cross_site_patients"], "no cross-site overlap was generated"
    for patient in truth["cross_site_patients"]:
        assert patient["abha"]
        assert len(set(patient["sites"])) > 1
        mrns = list(patient["mrns"].values())
        assert len(set(mrns)) == len(mrns)


def test_some_images_carry_burned_in_text(corpus):
    out, truth = corpus
    burned = [s for s in truth["studies"] if s["burned_in"]]
    assert burned, "no burned-in annotation was generated"

    ds = pydicom.dcmread(out / burned[0]["site_id"] / "images" / f"{burned[0]['accession']}.dcm")
    assert ds.BurnedInAnnotation == "YES"


def test_generation_is_reproducible_for_a_fixed_seed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_corpus(a, seed=99, n_patients=6, n_cross_site=2, image_size=64)
    generate_corpus(b, seed=99, n_patients=6, n_cross_site=2, image_size=64)

    for path in sorted(a.rglob("*")):
        if path.is_file():
            other = b / path.relative_to(a)
            assert other.read_bytes() == path.read_bytes(), path.name


def test_different_seeds_produce_different_corpora(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_corpus(a, seed=1, n_patients=6, n_cross_site=2, image_size=64)
    generate_corpus(b, seed=2, n_patients=6, n_cross_site=2, image_size=64)
    assert (a / "_ground_truth.json").read_bytes() != (b / "_ground_truth.json").read_bytes()


def test_ground_truth_lists_the_phi_that_should_disappear(corpus):
    _, truth = corpus
    assert truth["phi_values"]
    assert any("Sunrise" in v or "Meridian" in v or "Northstar" in v for v in truth["phi_values"])


def test_given_names_agree_with_sex():
    import random

    rng = random.Random(0)
    for _ in range(50):
        _, display = make_person_name(rng, "F")
        assert display.split()[0] in FEMALE_GIVEN_NAMES
        _, display = make_person_name(rng, "M")
        assert display.split()[0] in MALE_GIVEN_NAMES


def test_burn_in_preserves_bit_depth_and_brightens_pixels():
    rng = np.random.default_rng(0)
    base = synthesise_radiograph(rng, size=128)
    burned = burn_in_text(base, ["SHARMA RAVI", "MRN SMC-001234"])

    assert burned.dtype == np.uint16
    assert burned.max() <= MAX_VALUE
    assert burned.sum() > base.sum()
    # The overlay is confined to the corner it was requested in.
    assert not np.array_equal(burned[:60, :], base[:60, :])
    assert np.array_equal(burned[100:, :], base[100:, :])


def test_synthetic_image_has_structure_not_noise():
    """A flat or purely random image would make text detection trivially easy."""
    rng = np.random.default_rng(7)
    img = synthesise_radiograph(rng, size=256)
    # A bright central mediastinal column against darker lung fields.
    centre = img[:, 118:138].mean()
    left_lung = img[:, 40:90].mean()
    assert centre > left_lung * 1.2


def test_ground_truth_is_valid_json(corpus):
    out, _ = corpus
    json.loads((out / "_ground_truth.json").read_text(encoding="utf-8"))


def test_site_profiles_are_mutually_distinct():
    a, b, c = SITES_BY_ID["SITE_A"], SITES_BY_ID["SITE_B"], SITES_BY_ID["SITE_C"]
    assert len({a.view_channel, b.view_channel}) == 2
    assert len({a.label_channel, b.label_channel, c.label_channel}) == 3
    assert len({a.date_channel, b.date_channel}) == 2
