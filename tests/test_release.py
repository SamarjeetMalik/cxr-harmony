"""Releases: content addressing, and the leakage guarantee.

The leakage test is the one that matters. Splitting per study rather than per
patient is the commonest serious defect in medical imaging datasets, and it is
invisible to every metric a random-split evaluation produces — the score just
comes out better than the model deserves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cxr_harmony.deid import deidentify
from cxr_harmony.harmonize import harmonize
from cxr_harmony.ingest import ingest
from cxr_harmony.release import (
    SplitRatios,
    assign_all,
    assign_split,
    build_release,
    digest_manifest,
    realised_proportions,
    verify_release,
)
from cxr_harmony.release.builder import MANIFEST_NAME, RELEASE_NAME, SPLITS_NAME
from cxr_harmony.reports import process_reports
from cxr_harmony.schema.vocab import Split

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "sites"
KEY = b"release-stage-test-key-32-bytes!"


@pytest.fixture(scope="module")
def released(tmp_path_factory):
    from cxr_harmony.synth import generate_corpus
    from cxr_harmony.workspace import Workspace

    src = tmp_path_factory.mktemp("src")
    truth = generate_corpus(src, seed=909, n_patients=48, n_cross_site=10, image_size=128)
    ws = Workspace(tmp_path_factory.mktemp("ws") / "work").ensure()
    ingest(src, ws)
    deidentify(src, ws, key=KEY)
    process_reports(src, ws, key=KEY)
    harmonised = harmonize(src, ws, CONFIGS)
    result = build_release(harmonised.dataset, ws, version="v1.0.0")
    return src, ws, truth, harmonised, result


# --- The leakage guarantee --------------------------------------------------


def test_no_patient_appears_in_two_splits(released):
    _, _, _, harmonised, result = released
    studies_by_patient: dict[str, set[str]] = {}
    for study in harmonised.dataset.studies:
        split = result.assignments[study.pseudo_patient_id]
        studies_by_patient.setdefault(study.pseudo_patient_id, set()).add(split.value)

    offenders = {pid: splits for pid, splits in studies_by_patient.items() if len(splits) > 1}
    assert not offenders, offenders


def test_patients_with_several_studies_keep_them_together(released):
    """The specific failure a per-study split would produce."""
    _, _, _, harmonised, result = released
    by_patient: dict[str, list[str]] = {}
    for study in harmonised.dataset.studies:
        by_patient.setdefault(study.pseudo_patient_id, []).append(study.study_uid)

    multi = {pid: uids for pid, uids in by_patient.items() if len(uids) > 1}
    assert multi, "no patient had more than one study, so the test proves nothing"
    for pid in multi:
        assert isinstance(result.assignments[pid], Split)


def test_cross_site_patients_do_not_straddle_the_split(released):
    """Linkage in de-identification is what makes this possible at all."""
    _, _, truth, harmonised, result = released
    sites_by_patient: dict[str, set[str]] = {}
    for study in harmonised.dataset.studies:
        sites_by_patient.setdefault(study.pseudo_patient_id, set()).add(study.site_id)

    cross_site = [pid for pid, sites in sites_by_patient.items() if len(sites) > 1]
    assert len(cross_site) == len(truth["cross_site_patients"])
    for pid in cross_site:
        assert pid in result.assignments


# --- Split assignment properties --------------------------------------------


def test_assignment_depends_only_on_the_patient():
    """Which is what makes it stable as the cohort grows."""
    ratios = SplitRatios()
    small = assign_all([f"{i:016x}" for i in range(50)], ratios=ratios)
    grown = assign_all([f"{i:016x}" for i in range(5000)], ratios=ratios)
    for pid, split in small.items():
        assert grown[pid] == split, pid


def test_assignment_is_order_independent():
    ids = [f"{i:016x}" for i in range(200)]
    assert assign_all(ids) == assign_all(list(reversed(ids)))


def test_changing_the_salt_reassigns():
    ids = [f"{i:016x}" for i in range(400)]
    a = assign_all(ids, salt="one")
    b = assign_all(ids, salt="two")
    assert sum(1 for pid in ids if a[pid] != b[pid]) > 50


def test_proportions_approach_the_target_at_scale():
    ids = [f"{i:016x}" for i in range(20000)]
    proportions = realised_proportions(assign_all(ids))
    assert abs(proportions["train"] - 0.70) < 0.02
    assert abs(proportions["val"] - 0.15) < 0.02
    assert abs(proportions["test"] - 0.15) < 0.02


def test_every_patient_receives_exactly_one_split(released):
    _, _, _, harmonised, result = released
    assert set(result.assignments) == {p.pseudo_id for p in harmonised.dataset.patients}


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        SplitRatios(train=0.8, val=0.3, test=0.1)


def test_custom_ratios_are_honoured():
    ids = [f"{i:016x}" for i in range(20000)]
    proportions = realised_proportions(
        assign_all(ids, ratios=SplitRatios(train=0.5, val=0.25, test=0.25))
    )
    assert abs(proportions["train"] - 0.50) < 0.02


def test_a_single_patient_still_gets_a_split():
    assert isinstance(assign_split("a" * 16, salt="s", ratios=SplitRatios()), Split)


# --- Content addressing -----------------------------------------------------


def test_release_artefacts_are_written(released):
    _, _, _, _, result = released
    for name in (MANIFEST_NAME, SPLITS_NAME, RELEASE_NAME, "datasheet.md"):
        assert (result.directory / name).exists(), name


def test_manifest_covers_every_image_and_report(released):
    _, _, _, harmonised, result = released
    entries = json.loads((result.directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    images = [e for e in entries if e["path"].startswith("images/")]
    reports = [e for e in entries if e["path"].startswith("reports/")]
    assert len(images) == len(harmonised.dataset.instances)
    assert len(reports) == len(harmonised.dataset.reports)


def test_digests_match_the_files_on_disk(released):
    """Recomputed at release time, not copied from an upstream record."""
    import hashlib

    _, ws, _, _, result = released
    entries = json.loads((result.directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    for entry in entries[:15]:
        kind, _, rest = entry["path"].partition("/")
        root = ws.deid_store if kind == "images" else ws.reports_dir
        actual = hashlib.sha256((root / rest).read_bytes()).hexdigest()
        assert actual == entry["sha256"], entry["path"]


def test_release_is_byte_reproducible(released):
    """Same data cut twice must yield the same identity, or the digest is useless."""
    _, ws, _, harmonised, result = released
    again = build_release(harmonised.dataset, ws, version="v1.0.0-again")
    assert again.dataset_digest == result.dataset_digest
    assert (again.directory / MANIFEST_NAME).read_bytes() == (
        result.directory / MANIFEST_NAME
    ).read_bytes()
    assert (again.directory / SPLITS_NAME).read_bytes() == (
        result.directory / SPLITS_NAME
    ).read_bytes()


def test_manifest_carries_no_timestamp(released):
    """Anything inside a content address must be a function of the content."""
    _, _, _, _, result = released
    text = (result.directory / MANIFEST_NAME).read_text(encoding="utf-8")
    assert "created_at" not in text
    assert "timestamp" not in text


def test_provenance_that_varies_lives_outside_the_manifest(released):
    _, ws, _, harmonised, _ = released
    stamped = build_release(
        harmonised.dataset, ws, version="v1.0.1", created_at="2026-07-31T10:00:00Z"
    )
    meta = json.loads((stamped.directory / RELEASE_NAME).read_text(encoding="utf-8"))
    assert meta["created_at"] == "2026-07-31T10:00:00Z"
    # ...and does not perturb the identity.
    unstamped = build_release(harmonised.dataset, ws, version="v1.0.2")
    assert stamped.dataset_digest == unstamped.dataset_digest


def test_digest_changes_when_content_changes():
    a = [{"path": "images/x.dcm", "sha256": "a" * 64}]
    b = [{"path": "images/x.dcm", "sha256": "b" * 64}]
    assert digest_manifest(a) != digest_manifest(b)


def test_verify_release_accepts_an_intact_release(released):
    _, _, _, _, result = released
    ok, problems = verify_release(result.directory)
    assert ok, problems


def test_verify_release_rejects_a_tampered_manifest(released, tmp_path):
    import shutil

    _, _, _, _, result = released
    copied = tmp_path / "tampered"
    shutil.copytree(result.directory, copied)

    entries = json.loads((copied / MANIFEST_NAME).read_text(encoding="utf-8"))
    entries[0]["sha256"] = "0" * 64
    (copied / MANIFEST_NAME).write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")

    ok, problems = verify_release(copied)
    assert not ok
    assert problems


def test_missing_image_fails_the_release_loudly(released, tmp_path):
    from cxr_harmony.release import build_manifest
    from cxr_harmony.workspace import Workspace

    _, _, _, harmonised, _ = released
    empty = Workspace(tmp_path / "empty").ensure()
    with pytest.raises(FileNotFoundError):
        build_manifest(harmonised.dataset, empty)


# --- Datasheet --------------------------------------------------------------


def test_datasheet_records_realised_not_target_proportions(released):
    _, _, _, _, result = released
    text = (result.directory / "datasheet.md").read_text(encoding="utf-8")
    assert "Realised" in text
    assert result.dataset_digest in text


def test_datasheet_states_the_limitations_that_affect_use(released):
    _, _, _, _, result = released
    text = (result.directory / "datasheet.md").read_text(encoding="utf-8").lower()
    assert "per patient" in text
    assert "leave-one-site-out" in text
    assert "report_rule" in text
