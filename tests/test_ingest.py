"""Ingest is judged on what it refuses and how legibly it accounts for it."""

from __future__ import annotations

import shutil

import pydicom
import pytest

from cxr_harmony.ingest import discover_sites, ingest, sha256_file
from cxr_harmony.schema.vocab import QuarantineReason
from cxr_harmony.workspace import read_jsonl


def test_discovers_every_site(delivery):
    src, _ = delivery
    assert discover_sites(src) == ["SITE_A", "SITE_B", "SITE_C"]


def test_ignores_files_at_the_delivery_root(delivery):
    """_ground_truth.json sits beside the site folders and must not be ingested."""
    src, _ = delivery
    assert (src / "_ground_truth.json").exists()
    assert "_ground_truth.json" not in discover_sites(src)


def test_accepts_the_whole_valid_delivery(delivery, workspace):
    src, truth = delivery
    result = ingest(src, workspace)
    assert result.n_accepted == truth["n_studies"]
    assert result.n_quarantined == 0


def test_manifest_is_written_and_reloadable(delivery, workspace):
    src, _ = delivery
    result = ingest(src, workspace)
    rows = list(read_jsonl(workspace.raw_manifest))
    assert len(rows) == result.n_accepted
    assert {"sha256", "site_id", "sop_uid", "study_uid", "source_path"} <= set(rows[0])


def test_ingest_writes_no_pixel_data(delivery, workspace):
    """The working store must not gain an identifiable copy of anything."""
    src, _ = delivery
    ingest(src, workspace)
    assert not workspace.deid_store.exists()
    assert not list(workspace.root.rglob("*.dcm"))


def test_reports_are_paired_to_their_images(delivery, workspace):
    src, _ = delivery
    result = ingest(src, workspace)
    assert all(r.report_path is not None for r in result.accepted)


def test_rerunning_produces_identical_manifests(delivery, workspace):
    src, _ = delivery
    ingest(src, workspace)
    first = workspace.raw_manifest.read_bytes()
    ingest(src, workspace)
    assert workspace.raw_manifest.read_bytes() == first


def test_duplicate_content_is_quarantined_not_indexed_twice(delivery, workspace, tmp_path):
    """Sites resend overlapping batches after a failed transfer more often than they admit."""
    src, _ = delivery
    staged = tmp_path / "staged"
    shutil.copytree(src, staged)

    original = next((staged / "SITE_A" / "images").glob("*.dcm"))
    shutil.copy(original, original.with_name("RESENT_" + original.name))

    result = ingest(staged, workspace)
    reasons = result.reasons()
    assert reasons.get(QuarantineReason.DUPLICATE_CONTENT.value) == 1
    digests = [r.sha256 for r in result.accepted]
    assert len(digests) == len(set(digests))


def test_non_dicom_file_is_quarantined(delivery, workspace, tmp_path):
    src, _ = delivery
    staged = tmp_path / "staged"
    shutil.copytree(src, staged)
    (staged / "SITE_A" / "images" / "readme.dcm").write_text("not a dicom", encoding="utf-8")

    result = ingest(staged, workspace)
    assert result.reasons().get(QuarantineReason.NOT_DICOM.value) == 1


@pytest.mark.parametrize(
    ("attribute", "value", "reason"),
    [
        ("Modality", "MR", QuarantineReason.WRONG_MODALITY),
        ("BodyPartExamined", "ABDOMEN", QuarantineReason.WRONG_BODY_PART),
    ],
)
def test_out_of_scope_objects_are_quarantined(
    delivery, workspace, tmp_path, attribute, value, reason
):
    src, _ = delivery
    staged = tmp_path / "staged"
    shutil.copytree(src, staged)

    path = next((staged / "SITE_A" / "images").glob("*.dcm"))
    ds = pydicom.dcmread(path)
    setattr(ds, attribute, value)
    ds.save_as(path, enforce_file_format=True)

    result = ingest(staged, workspace)
    assert result.reasons().get(reason.value) == 1


def test_missing_required_tag_is_quarantined_with_the_tag_named(delivery, workspace, tmp_path):
    src, _ = delivery
    staged = tmp_path / "staged"
    shutil.copytree(src, staged)

    path = next((staged / "SITE_A" / "images").glob("*.dcm"))
    ds = pydicom.dcmread(path)
    ds.PatientID = ""
    ds.save_as(path, enforce_file_format=True)

    result = ingest(staged, workspace)
    record = next(
        r
        for r in result.quarantined
        if r["reason"] == QuarantineReason.MISSING_REQUIRED_TAG.value
    )
    assert record["detail"] == "PatientID"


def test_empty_body_part_is_tolerated(delivery, workspace, tmp_path):
    """Absent is not the same as wrong; only a populated mismatch disqualifies."""
    src, _ = delivery
    staged = tmp_path / "staged"
    shutil.copytree(src, staged)

    path = next((staged / "SITE_A" / "images").glob("*.dcm"))
    ds = pydicom.dcmread(path)
    ds.BodyPartExamined = ""
    ds.save_as(path, enforce_file_format=True)

    result = ingest(staged, workspace)
    assert result.reasons().get(QuarantineReason.WRONG_BODY_PART.value) is None


def test_quarantine_file_accounts_for_every_rejection(delivery, workspace, tmp_path):
    src, _ = delivery
    staged = tmp_path / "staged"
    shutil.copytree(src, staged)
    (staged / "SITE_B" / "images" / "junk.dcm").write_text("x", encoding="utf-8")

    result = ingest(staged, workspace)
    rows = list(read_jsonl(workspace.quarantine))
    assert len(rows) == result.n_quarantined
    assert all({"source_path", "site_id", "reason"} <= set(r) for r in rows)


def test_digest_matches_the_file_on_disk(delivery, workspace):
    src, _ = delivery
    result = ingest(src, workspace)
    record = result.accepted[0]
    assert sha256_file(src / record.source_path) == record.sha256
