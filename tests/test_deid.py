"""De-identification is the stage where a defect is a disclosure, so it is tested hardest.

The substring checks below are only possible because the corpus is synthetic and
its ground truth is known. That is the argument for validating a de-identifier
against generated data before ever pointing it at patients: against real data you
can verify structure, but you cannot verify that a specific name is gone, because
you are not allowed to keep the list of names to search for.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pydicom
import pytest

from cxr_harmony.deid import Pseudonymiser, deidentify, verify_store
from cxr_harmony.deid.extract import AGE_CAP, compute_age, parse_native_date
from cxr_harmony.deid.pixels import clean_pixel_data, detect_text_regions
from cxr_harmony.deid.pseudonym import normalise_identifier
from cxr_harmony.deid.verify import pixels_are_clean
from cxr_harmony.ingest import ingest
from cxr_harmony.synth.pixels import burn_in_text, synthesise_radiograph
from cxr_harmony.workspace import read_jsonl

KEY = b"deterministic-test-key-32-bytes!"
OTHER_KEY = b"a-completely-different-key-32b!!"


@pytest.fixture(scope="module")
def deidentified(tmp_path_factory):
    """Run ingest + de-identification once for the module."""
    from cxr_harmony.synth import generate_corpus
    from cxr_harmony.workspace import Workspace

    src = tmp_path_factory.mktemp("src")
    truth = generate_corpus(src, seed=515, n_patients=24, n_cross_site=6, image_size=256)
    ws = Workspace(tmp_path_factory.mktemp("ws") / "work").ensure()
    ingest(src, ws)
    result = deidentify(src, ws, key=KEY)
    return src, ws, truth, result


# --- The headline guarantee -------------------------------------------------


def test_verification_passes_on_the_whole_store(deidentified):
    _, ws, truth, _ = deidentified
    report = verify_store(ws.deid_store, phi_values=truth["phi_values"])
    assert report.n_checked > 0
    assert report.passed, report.by_kind()


def test_no_generated_identifier_survives_in_any_header(deidentified):
    """The strong check: search for the exact values that went in."""
    _, ws, truth, _ = deidentified
    report = verify_store(ws.deid_store, phi_values=truth["phi_values"])
    assert [v for v in report.violations if v.kind == "phi_substring"] == []


def test_every_ingested_object_was_written(deidentified):
    _, ws, _, result = deidentified
    assert result.n_objects == len(list(read_jsonl(ws.raw_manifest)))


def test_private_blocks_do_not_survive(deidentified):
    """Site B's private block carried a duplicate of the patient's name."""
    _, ws, _, _ = deidentified
    for path in ws.deid_store.rglob("*.dcm"):
        ds = pydicom.dcmread(path)
        assert [e for e in ds if e.tag.group % 2 == 1] == []


# --- Cross-site linkage -----------------------------------------------------


def test_cross_site_patients_collapse_to_one_pseudonym(deidentified):
    _, _, truth, result = deidentified
    linked = {pid for pid, sites in result.patient_sites().items() if len(sites) > 1}
    assert len(linked) == len(truth["cross_site_patients"])


def test_patients_without_a_national_id_are_not_linked(deidentified):
    """The conservative outcome: unlinkable records stay separate rather than guess."""
    _, _, _, result = deidentified
    for record in result.records:
        if not record.linked_across_sites:
            sites = result.patient_sites()[record.pseudo_patient_id]
            assert len(sites) == 1


def test_national_id_formatting_does_not_affect_linkage():
    pseudo = Pseudonymiser(KEY)
    a, _ = pseudo.patient_pseudonym(
        national_id="99-1234-5678-9012", site_id="SITE_A", local_mrn="SMC-1"
    )
    b, _ = pseudo.patient_pseudonym(
        national_id="99123456789012", site_id="SITE_B", local_mrn="ND-9"
    )
    assert a == b


def test_local_mrns_alone_never_link_across_sites():
    """Two hospitals can issue the same MRN string to different people."""
    pseudo = Pseudonymiser(KEY)
    a, linked_a = pseudo.patient_pseudonym(national_id=None, site_id="SITE_A", local_mrn="00123")
    b, linked_b = pseudo.patient_pseudonym(national_id=None, site_id="SITE_B", local_mrn="00123")
    assert a != b
    assert not linked_a and not linked_b


def test_normalise_identifier_strips_only_formatting():
    assert normalise_identifier("99-1234-5678-9012") == "99123456789012"
    assert normalise_identifier(" smc/001 ") == "SMC001"
    assert normalise_identifier("") == ""


# --- Keying -----------------------------------------------------------------


def test_pseudonyms_depend_on_the_key(deidentified):
    """An unkeyed hash of a 14-digit identifier is enumerable, hence reversible."""
    src, ws, _, result = deidentified
    from cxr_harmony.workspace import Workspace

    other = Workspace(ws.root.parent / "other").ensure()
    ingest(src, other)
    rekeyed = deidentify(src, other, key=OTHER_KEY)

    original = {r.pseudo_patient_id for r in result.records}
    changed = {r.pseudo_patient_id for r in rekeyed.records}
    assert original.isdisjoint(changed)


def test_same_key_reproduces_the_same_pseudonyms(deidentified):
    src, ws, _, result = deidentified
    from cxr_harmony.workspace import Workspace

    repeat = Workspace(ws.root.parent / "repeat").ensure()
    ingest(src, repeat)
    again = deidentify(src, repeat, key=KEY)
    assert [r.pseudo_patient_id for r in again.records] == [
        r.pseudo_patient_id for r in result.records
    ]


def test_key_must_be_long_enough():
    with pytest.raises(ValueError):
        Pseudonymiser(b"tooshort")


def test_domain_separation_between_derived_values():
    """A pseudonym must never coincide with a UID derived from the same input."""
    pseudo = Pseudonymiser(KEY)
    value = "1.2.826.0.1.99"
    pid, _ = pseudo.patient_pseudonym(national_id=value, site_id="S", local_mrn="")
    assert pid not in pseudo.remap_uid(value)


# --- Temporal ---------------------------------------------------------------


def test_dates_are_shifted_not_erased(deidentified):
    _, ws, _, _ = deidentified
    dates = []
    for path in ws.deid_store.rglob("*.dcm"):
        ds = pydicom.dcmread(path)
        if ds.StudyDate:
            dates.append(str(ds.StudyDate))
    assert dates, "every date was destroyed; the Modified Dates option was not applied"
    assert all(len(d) == 8 and d.isdigit() for d in dates)


def test_intervals_within_a_patient_survive_the_shift(deidentified):
    """The point of a per-patient offset: a follow-up interval must be preserved."""
    src, ws, truth, result = deidentified

    original_by_accession = {s["accession"]: s["study_date"] for s in truth["studies"]}
    by_patient: dict[str, list[tuple[str, str]]] = {}
    for record in result.records:
        accession = record.source_path.rsplit("/", 1)[-1].removesuffix(".dcm")
        ds = pydicom.dcmread(ws.deid_store / record.relative_path)
        if not ds.StudyDate:
            continue
        by_patient.setdefault(record.pseudo_patient_id, []).append(
            (original_by_accession[accession], str(ds.StudyDate))
        )

    compared = 0
    for entries in by_patient.values():
        if len(entries) < 2:
            continue
        entries.sort()
        (orig_a, shift_a), (orig_b, shift_b) = entries[0], entries[1]
        original_gap = (
            datetime.strptime(orig_b, "%Y-%m-%d") - datetime.strptime(orig_a, "%Y-%m-%d")
        ).days
        shifted_gap = (
            datetime.strptime(shift_b, "%Y%m%d") - datetime.strptime(shift_a, "%Y%m%d")
        ).days
        assert original_gap == shifted_gap
        compared += 1
    assert compared > 0, "no patient had two dated studies to compare"


def test_shifted_dates_differ_from_the_originals(deidentified):
    src, ws, truth, result = deidentified
    original_by_accession = {s["accession"]: s["study_date"] for s in truth["studies"]}

    differed = 0
    for record in result.records:
        accession = record.source_path.rsplit("/", 1)[-1].removesuffix(".dcm")
        ds = pydicom.dcmread(ws.deid_store / record.relative_path)
        if not ds.StudyDate:
            continue
        original = original_by_accession[accession].replace("-", "")
        if str(ds.StudyDate) != original:
            differed += 1
    assert differed > 0


def test_date_offset_is_per_patient_not_global():
    pseudo = Pseudonymiser(KEY)
    offsets = {pseudo.date_offset_days(f"{i:016x}") for i in range(40)}
    assert len(offsets) > 20, "offsets are not sufficiently patient-specific"


def test_unparseable_date_becomes_empty_rather_than_wrong():
    pseudo = Pseudonymiser(KEY)
    assert pseudo.shift_da("not-a-date", "0" * 16) == ""
    assert pseudo.shift_da("", "0" * 16) == ""


# --- UIDs -------------------------------------------------------------------


def test_uids_are_remapped_and_original_uids_do_not_appear(deidentified):
    _, ws, _, result = deidentified
    for original, remapped in result.uid_map.items():
        assert original != remapped
        assert remapped.startswith("2.25.")
        assert len(remapped) <= 64


def test_uid_remapping_is_a_function_not_a_lottery():
    pseudo = Pseudonymiser(KEY)
    uid = "1.2.826.0.1.3680043.10.1337.1.5"
    assert pseudo.remap_uid(uid) == pseudo.remap_uid(uid)
    assert pseudo.remap_uid(uid) != pseudo.remap_uid(uid + "1")


def test_file_meta_uid_tracks_the_remapped_dataset_uid(deidentified):
    """An object whose file meta disagrees with its dataset is rejected by some readers."""
    _, ws, _, _ = deidentified
    for path in list(ws.deid_store.rglob("*.dcm"))[:10]:
        ds = pydicom.dcmread(path)
        assert ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID


# --- Pixels -----------------------------------------------------------------


def test_burned_in_text_is_detected_and_zeroed():
    base = synthesise_radiograph(np.random.default_rng(11), size=512)
    burned = burn_in_text(base, ["MEERA NAIR", "MRN ND0001234", "28-07-2025"])
    cleaned, regions = clean_pixel_data(burned)
    assert regions
    assert pixels_are_clean(burned, cleaned)


def test_clean_images_are_left_alone():
    base = synthesise_radiograph(np.random.default_rng(12), size=512)
    cleaned, regions = clean_pixel_data(base)
    assert regions == []
    assert np.array_equal(cleaned, base)


def test_detector_is_not_fooled_by_saturated_anatomy():
    """Spine and diaphragm reach the same value the text is drawn at."""
    base = synthesise_radiograph(np.random.default_rng(13), size=512)
    assert base.max() >= 4000
    assert detect_text_regions(base) == []


def test_redaction_is_destructive_not_a_blur():
    base = synthesise_radiograph(np.random.default_rng(14), size=512)
    burned = burn_in_text(base, ["ANANYA SHARMA", "MRN SMC-001234"])
    cleaned, regions = clean_pixel_data(burned)
    for region in regions:
        patch = cleaned[region.y : region.y + region.height, region.x : region.x + region.width]
        assert patch.max() == 0


def test_redacted_objects_declare_no_burned_in_annotation(deidentified):
    _, ws, _, _ = deidentified
    for path in ws.deid_store.rglob("*.dcm"):
        assert str(pydicom.dcmread(path).BurnedInAnnotation) == "NO"


def test_pixel_redaction_is_recorded_for_audit(deidentified):
    _, _, truth, result = deidentified
    redacted = [r for r in result.records if r.pixel_redacted]
    assert redacted
    assert all(r.redacted_regions for r in redacted)
    expected = sum(1 for s in truth["studies"] if s["burned_in"])
    assert len(redacted) >= expected * 0.9


# --- Extraction before redaction --------------------------------------------


def test_clinical_values_are_captured_before_the_profile_destroys_them(deidentified):
    """SeriesDescription and ImageComments are removed, but their content is needed."""
    _, ws, _, result = deidentified
    facts = {f.site_id: f for f in result.facts}

    # Site B keeps the projection only in SeriesDescription, which is now gone.
    assert facts["SITE_B"].view_native
    for path in (ws.deid_store / "SITE_B").rglob("*.dcm"):
        assert "SeriesDescription" not in pydicom.dcmread(path)

    # Site A keeps labels only in ImageComments, likewise gone.
    assert facts["SITE_A"].labels_native
    for path in (ws.deid_store / "SITE_A").rglob("*.dcm"):
        assert "ImageComments" not in pydicom.dcmread(path)


def test_site_b_private_date_is_recovered_before_the_block_is_stripped(deidentified):
    _, _, _, result = deidentified
    site_b = [f for f in result.facts if f.site_id == "SITE_B"]
    assert site_b
    assert all(f.study_date_native for f in site_b)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("20240412", (2024, 4, 12)),
        ("12-04-2024", (2024, 4, 12)),
        ("2024-04-12", (2024, 4, 12)),
        ("12 Apr 2024", (2024, 4, 12)),
    ],
)
def test_every_site_date_format_parses(text, expected):
    parsed = parse_native_date(text)
    assert (parsed.year, parsed.month, parsed.day) == expected


def test_unknown_date_format_returns_none_rather_than_a_guess():
    assert parse_native_date("last Tuesday") is None
    assert parse_native_date("") is None


def test_age_is_capped_in_the_identifying_tail():
    from datetime import date

    assert compute_age("19300101", date(2025, 1, 1)) == AGE_CAP
    assert compute_age("19800101", date(2025, 1, 1)) == 45
    assert compute_age("", date(2025, 1, 1)) is None


def test_age_survives_but_birth_date_does_not(deidentified):
    _, ws, _, _ = deidentified
    seen_age = False
    for path in ws.deid_store.rglob("*.dcm"):
        ds = pydicom.dcmread(path)
        assert "PatientBirthDate" not in ds
        if getattr(ds, "PatientAge", ""):
            seen_age = True
            assert int(str(ds.PatientAge)[:3]) <= AGE_CAP
    assert seen_age


# --- Reproducibility --------------------------------------------------------


def test_the_stage_is_byte_reproducible(deidentified):
    src, ws, _, _ = deidentified
    from cxr_harmony.workspace import Workspace

    repeat = Workspace(ws.root.parent / "byte_repeat").ensure()
    ingest(src, repeat)
    deidentify(src, repeat, key=KEY)

    for path in sorted(ws.deid_store.rglob("*.dcm")):
        mirror = repeat.deid_store / path.relative_to(ws.deid_store)
        assert mirror.read_bytes() == path.read_bytes(), path.name
    assert repeat.deid_manifest.read_bytes() == ws.deid_manifest.read_bytes()


# --- Key management ---------------------------------------------------------


def test_key_is_created_once_and_then_reused(tmp_path):
    from cxr_harmony.deid import load_or_create_key

    path = tmp_path / "pseudonym.key"
    with pytest.warns(RuntimeWarning, match="local filesystem"):
        first = load_or_create_key(path, allow_create=True)
    assert path.exists()
    assert len(first) >= 32
    # Reading an existing key is not a creation event and must not warn.
    assert load_or_create_key(path) == first, "a second call must not rotate the key"


def test_key_creation_is_opt_in(tmp_path):
    """Silent creation is how a deployment ends up with its re-identification
    secret in a backed-up working directory, discovered only at audit."""
    from cxr_harmony.deid import load_or_create_key

    with pytest.raises(FileNotFoundError, match="CXR_HARMONY_KEY"):
        load_or_create_key(tmp_path / "absent.key")


def test_creating_a_filesystem_key_warns(tmp_path):
    from cxr_harmony.deid import load_or_create_key

    with pytest.warns(RuntimeWarning, match="not for production"):
        load_or_create_key(tmp_path / "dev.key", allow_create=True)


def test_pseudonymiser_from_environment(monkeypatch):
    """The deployment path: a mounted secret never touches the filesystem."""
    from cxr_harmony.deid import Pseudonymiser

    monkeypatch.setenv("CXR_HARMONY_KEY", (b"k" * 32).hex())
    pseudo = Pseudonymiser.from_env()
    expected = Pseudonymiser(b"k" * 32)
    assert pseudo.patient_pseudonym(
        national_id="99-1", site_id="S", local_mrn="1"
    ) == expected.patient_pseudonym(national_id="99-1", site_id="S", local_mrn="1")


def test_from_env_refuses_an_unset_or_malformed_variable(monkeypatch):
    from cxr_harmony.deid import Pseudonymiser

    monkeypatch.delenv("CXR_HARMONY_KEY", raising=False)
    with pytest.raises(KeyError):
        Pseudonymiser.from_env()

    monkeypatch.setenv("CXR_HARMONY_KEY", "not-hex!")
    with pytest.raises(ValueError, match="hex-encoded"):
        Pseudonymiser.from_env()


def test_a_truncated_key_file_is_refused_rather_than_used(tmp_path):
    """Silently proceeding with a weak key would produce reversible pseudonyms."""
    from cxr_harmony.deid import load_or_create_key

    path = tmp_path / "pseudonym.key"
    path.write_bytes(b"short")
    with pytest.raises(ValueError):
        load_or_create_key(path)


def test_generated_keys_differ_between_deployments(tmp_path):
    from cxr_harmony.deid import load_or_create_key

    with pytest.warns(RuntimeWarning):
        a = load_or_create_key(tmp_path / "a.key", allow_create=True)
    with pytest.warns(RuntimeWarning):
        b = load_or_create_key(tmp_path / "b.key", allow_create=True)
    assert a != b


# --- The verifier must be able to fail --------------------------------------


def test_verifier_catches_a_surviving_identifier(deidentified, tmp_path):
    """A verifier that cannot fail is not evidence of anything."""
    from cxr_harmony.deid.verify import verify_object

    _, ws, _, _ = deidentified
    source = next(ws.deid_store.rglob("*.dcm"))
    ds = pydicom.dcmread(source)
    ds.InstitutionName = "Sunrise Medical College and Hospital"
    tampered = tmp_path / "tampered.dcm"
    ds.save_as(tampered, enforce_file_format=True)

    violations = verify_object(tampered, "tampered.dcm", ["Sunrise Medical College and Hospital"])
    kinds = {v.kind for v in violations}
    assert "attribute_survived" in kinds
    assert "phi_substring" in kinds


def test_verifier_catches_a_missing_deidentification_marker(deidentified, tmp_path):
    from cxr_harmony.deid.verify import verify_object

    _, ws, _, _ = deidentified
    ds = pydicom.dcmread(next(ws.deid_store.rglob("*.dcm")))
    del ds.PatientIdentityRemoved
    tampered = tmp_path / "unmarked.dcm"
    ds.save_as(tampered, enforce_file_format=True)

    violations = verify_object(tampered, "unmarked.dcm", [])
    assert any(v.kind == "missing_marker" for v in violations)


def test_verifier_catches_a_reintroduced_private_tag(deidentified, tmp_path):
    from cxr_harmony.deid.verify import verify_object

    _, ws, _, _ = deidentified
    ds = pydicom.dcmread(next(ws.deid_store.rglob("*.dcm")))
    block = ds.private_block(0x0033, "SOME VENDOR", create=True)
    block.add_new(0x01, "LO", "Ravi Sharma")
    tampered = tmp_path / "private.dcm"
    ds.save_as(tampered, enforce_file_format=True)

    violations = verify_object(tampered, "private.dcm", [])
    assert any(v.kind == "private_tag_survived" for v in violations)


def _greyscale_dataset(interpretation: str, pixels: np.ndarray) -> pydicom.Dataset:
    """A minimal dataset whose pixel data is actually decodable.

    pydicom needs file_meta with a transfer syntax before pixel_array will work,
    so a bare Dataset() is not enough to exercise the conversion.
    """
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    ds = pydicom.Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.PhotometricInterpretation = interpretation
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.Rows, ds.Columns = pixels.shape
    ds.PixelData = np.ascontiguousarray(pixels).tobytes()
    return ds


# --- Photometric interpretation (found via real hospital DICOM) --------------


def test_monochrome1_is_converted_to_monochrome2():
    """A real 400-file CR sample was 372 MONOCHROME1 to 28 MONOCHROME2.

    Left unnormalised, roughly 7% of that cohort reaches the model as a
    photographic negative of the rest. Nothing errors and nothing looks obviously
    wrong, which is what makes it dangerous.
    """
    from cxr_harmony.deid import normalise_photometric

    ds = _greyscale_dataset("MONOCHROME1", np.array([[0, 100, 2000, 4095]] * 4, dtype=np.uint16))

    assert normalise_photometric(ds) is True
    assert ds.PhotometricInterpretation == "MONOCHROME2"
    # Inverted about the stored-value ceiling, not about the observed maximum.
    assert list(ds.pixel_array[0]) == [4095, 3995, 2095, 0]


def test_monochrome2_is_left_alone():
    from cxr_harmony.deid import normalise_photometric

    ds = _greyscale_dataset("MONOCHROME2", np.array([[0, 4095], [1, 2]], dtype=np.uint16))

    assert normalise_photometric(ds) is False
    assert ds.PhotometricInterpretation == "MONOCHROME2"


def test_inversion_uses_the_declared_range_not_the_observed_maximum():
    """Otherwise the transform depends on image content, so two studies from the
    same device are mapped differently."""
    from cxr_harmony.deid import invert_pixels

    dim = np.array([0, 10, 20], dtype=np.uint16)
    bright = np.array([0, 10, 4000], dtype=np.uint16)
    assert list(invert_pixels(dim, 12)) == [4095, 4085, 4075]
    assert list(invert_pixels(bright, 12)) == [4095, 4085, 95]


def test_window_settings_are_dropped_on_conversion():
    """A centre calibrated for the old polarity would display the result badly."""
    from cxr_harmony.deid import normalise_photometric

    ds = _greyscale_dataset("MONOCHROME1", np.zeros((2, 2), dtype=np.uint16))
    ds.WindowCenter = 2048
    ds.WindowWidth = 4096

    normalise_photometric(ds)
    assert "WindowCenter" not in ds
    assert "WindowWidth" not in ds


def test_every_released_object_is_monochrome2(deidentified):
    _, ws, _, result = deidentified
    for path in ws.deid_store.rglob("*.dcm"):
        assert pydicom.dcmread(path).PhotometricInterpretation == "MONOCHROME2"
    assert all(r.photometric_interpretation == "MONOCHROME2" for r in result.records)


# --- Nested sequences -------------------------------------------------------


def test_identifiers_nested_in_a_sequence_are_removed(deidentified):
    """A de-identifier that walks only the top level reports success and leaves
    the accession number one level down."""
    _, ws, _, _ = deidentified
    for path in ws.deid_store.rglob("*.dcm"):
        ds = pydicom.dcmread(path)
        for item in getattr(ds, "RequestAttributesSequence", []):
            assert not getattr(item, "AccessionNumber", "")
            assert "RequestingPhysician" not in item
            assert "RequestedProcedureID" not in item


def test_institution_nested_in_a_sequence_is_removed(deidentified):
    _, ws, _, _ = deidentified
    for path in ws.deid_store.rglob("*.dcm"):
        ds = pydicom.dcmread(path)
        for item in getattr(ds, "ContributingEquipmentSequence", []):
            assert "InstitutionName" not in item
            assert "InstitutionAddress" not in item
            assert "StationName" not in item
            assert "DeviceSerialNumber" not in item


def test_uids_are_remapped_inside_sequences(deidentified):
    """A surviving original UID inside a sequence re-links the object to its source."""
    _, ws, _, result = deidentified
    originals = set(result.uid_map)
    for path in ws.deid_store.rglob("*.dcm"):
        ds = pydicom.dcmread(path)
        for item in getattr(ds, "RequestAttributesSequence", []):
            uid = str(getattr(item, "StudyInstanceUID", ""))
            if uid:
                assert uid not in originals
                assert uid.startswith("2.25.")


def test_a_sequence_inside_a_sequence_is_reached(deidentified):
    """Two levels down is where a shallow recursion stops."""
    _, ws, _, result = deidentified
    originals = set(result.uid_map)
    checked = 0
    for path in ws.deid_store.rglob("*.dcm"):
        ds = pydicom.dcmread(path)
        for item in getattr(ds, "RequestAttributesSequence", []):
            for inner in getattr(item, "ReferencedImageSequence", []):
                uid = str(getattr(inner, "ReferencedSOPInstanceUID", ""))
                if uid:
                    assert uid not in originals, "original UID survived two levels down"
                    assert uid.startswith("2.25.")
                    checked += 1
    assert checked > 0, "no nested sequence was present, so nothing was tested"


def test_nested_uid_remapping_is_consistent_with_the_top_level(deidentified):
    """The reference must still point at the object it pointed at before."""
    _, ws, _, _ = deidentified
    for path in ws.deid_store.rglob("*.dcm"):
        ds = pydicom.dcmread(path)
        for item in getattr(ds, "RequestAttributesSequence", []):
            for inner in getattr(item, "ReferencedImageSequence", []):
                referenced = str(getattr(inner, "ReferencedSOPInstanceUID", ""))
                if referenced:
                    assert referenced == str(ds.SOPInstanceUID)


def test_the_generator_actually_emits_nested_sequences(delivery):
    """Guards the tests above: if the fixture stopped nesting, they would all
    vacuously pass."""
    src, _ = delivery
    path = next((src / "SITE_A" / "images").glob("*.dcm"))
    ds = pydicom.dcmread(path)
    assert "RequestAttributesSequence" in ds
    assert "ContributingEquipmentSequence" in ds
    item = ds.RequestAttributesSequence[0]
    assert item.AccessionNumber
    assert "ReferencedImageSequence" in item


def test_verifier_catches_an_identifier_hidden_in_a_sequence(deidentified, tmp_path):
    from pydicom.sequence import Sequence

    from cxr_harmony.deid.verify import verify_object

    _, ws, _, _ = deidentified
    ds = pydicom.dcmread(next(ws.deid_store.rglob("*.dcm")))
    item = pydicom.Dataset()
    item.InstitutionName = "Sunrise Medical College and Hospital"
    ds.ContributingEquipmentSequence = Sequence([item])
    tampered = tmp_path / "nested.dcm"
    ds.save_as(tampered, enforce_file_format=True)

    violations = verify_object(
        tampered, "nested.dcm", ["Sunrise Medical College and Hospital"]
    )
    assert any(v.kind == "phi_substring" for v in violations)
