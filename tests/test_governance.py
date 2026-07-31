"""Audit chaining, statutory mapping, and the end-to-end verification pass."""

from __future__ import annotations

import json

import pytest

from cxr_harmony.governance import AuditLog, render_dpdp_markdown, run_verification
from cxr_harmony.governance.audit import GENESIS
from cxr_harmony.governance.dpdp import MAPPINGS
from cxr_harmony.governance.verify import write_policy_docs

# --- Audit log --------------------------------------------------------------


def test_entries_chain_to_their_predecessor(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    first = log.record("ingest", "scanned", counts={"accepted": 40})
    second = log.record("deid", "applied", counts={"objects": 40})

    assert first.previous_hash == GENESIS
    assert second.previous_hash == first.entry_hash
    assert second.sequence == 2


def test_an_intact_chain_verifies(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        log.record("stage", f"action-{i}", counts={"n": i})
    ok, problems = log.verify_chain()
    assert ok, problems


def test_editing_an_entry_breaks_the_chain(tmp_path):
    """The realistic tampering case: one line quietly changed months later."""
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(4):
        log.record("stage", f"action-{i}", counts={"n": i})

    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[1])
    entry["counts"] = {"n": 9999}
    lines[1] = json.dumps(entry, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, problems = log.verify_chain()
    assert not ok
    assert any("do not match its hash" in p for p in problems)


def test_deleting_an_entry_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(4):
        log.record("stage", f"action-{i}")

    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, problems = log.verify_chain()
    assert not ok
    assert problems


def test_reordering_entries_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(4):
        log.record("stage", f"action-{i}")

    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert not log.verify_chain()[0]


def test_an_empty_log_verifies(tmp_path):
    assert AuditLog(tmp_path / "missing.jsonl").verify_chain() == (True, [])


def test_entries_carry_no_patient_identifiers(tmp_path):
    """A log that records who was processed becomes another copy of the cohort."""
    log = AuditLog(tmp_path / "audit.jsonl")
    entry = log.record("deid", "applied", counts={"objects": 40}, detail={"profile": "annex-e"})
    serialised = json.dumps(entry.to_dict()).lower()
    for forbidden in ("patientname", "mrn", "pseudo_patient_id", "abha"):
        assert forbidden not in serialised


def test_the_log_is_append_only_in_practice(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.record("a", "one")
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    log.record("b", "two")
    assert path.read_text(encoding="utf-8").splitlines()[0] == first_line


# --- DPDP mapping -----------------------------------------------------------


def test_every_mapping_names_a_reference_and_a_control():
    assert MAPPINGS
    for mapping in MAPPINGS:
        assert mapping.reference.startswith("s.")
        assert mapping.obligation and mapping.control and mapping.where


def test_the_document_states_it_is_not_legal_advice():
    text = render_dpdp_markdown()
    assert "not legal advice" in text.lower()


def test_the_document_is_explicit_that_the_data_is_pseudonymised():
    """Claiming anonymity for keyed pseudonyms is the common and costly error."""
    text = render_dpdp_markdown().lower()
    assert "pseudonymised, not anonymised" in text
    assert "remains personal data" in text


def test_gaps_are_recorded_rather_than_omitted():
    text = render_dpdp_markdown()
    assert "## Not addressed by this codebase" in text
    assert "Cross-border transfer" in text


def test_policy_documents_are_written(tmp_path):
    written = write_policy_docs(tmp_path)
    assert written
    assert all(p.exists() and p.read_text(encoding="utf-8") for p in written)


# --- End-to-end verification ------------------------------------------------


@pytest.fixture(scope="module")
def verified(tmp_path_factory):
    from pathlib import Path

    from cxr_harmony.deid import deidentify
    from cxr_harmony.harmonize import harmonize
    from cxr_harmony.ingest import ingest
    from cxr_harmony.release import build_release
    from cxr_harmony.reports import process_reports
    from cxr_harmony.synth import generate_corpus
    from cxr_harmony.workspace import Workspace

    configs = Path(__file__).resolve().parents[1] / "configs" / "sites"
    key = b"governance-test-key-32-bytes!!!!"

    src = tmp_path_factory.mktemp("src")
    truth = generate_corpus(src, seed=1212, n_patients=24, n_cross_site=5, image_size=128)
    ws = Workspace(tmp_path_factory.mktemp("ws") / "work").ensure()

    log = AuditLog(ws.audit_log)
    ingested = ingest(src, ws)
    log.record("ingest", "scanned", counts={"accepted": ingested.n_accepted})
    deid = deidentify(src, ws, key=key)
    log.record("deid", "applied", counts={"objects": deid.n_objects})
    process_reports(src, ws, key=key)
    log.record("reports", "scrubbed", counts={"reports": 0})
    harmonised = harmonize(src, ws, configs)
    log.record("harmonize", "mapped", counts={"studies": harmonised.n_studies})
    build_release(harmonised.dataset, ws, version="v1.0.0")
    log.record("release", "cut", counts={"version": 1})

    return ws, truth


def test_verification_passes_on_a_complete_run(verified):
    ws, truth = verified
    report = run_verification(ws, phi_values=truth["phi_values"])
    assert report.passed, report.to_dict()
    assert report.deid_checked > 0
    assert report.audit_ok
    assert report.releases_ok == {"v1.0.0": True}


def test_verification_writes_its_evidence(verified):
    ws, truth = verified
    run_verification(ws, phi_values=truth["phi_values"])
    path = ws.qc_dir / "governance.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["passed"] is True


def test_verification_fails_when_an_identifier_is_reintroduced(verified, tmp_path):
    """'The pipeline completed' and 'the output is safe' are separate claims."""
    import pydicom

    ws, truth = verified
    path = next(ws.deid_store.rglob("*.dcm"))
    original = path.read_bytes()
    try:
        ds = pydicom.dcmread(path)
        ds.InstitutionName = "Sunrise Medical College and Hospital"
        ds.save_as(path, enforce_file_format=True)

        report = run_verification(ws, phi_values=truth["phi_values"])
        assert not report.passed
        assert report.deid_violations
    finally:
        path.write_bytes(original)


def test_verification_fails_when_the_audit_chain_is_broken(verified):
    ws, truth = verified
    original = ws.audit_log.read_text(encoding="utf-8")
    try:
        lines = original.splitlines()
        entry = json.loads(lines[0])
        entry["counts"] = {"accepted": 1}
        lines[0] = json.dumps(entry, sort_keys=True)
        ws.audit_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

        report = run_verification(ws, phi_values=truth["phi_values"])
        assert not report.passed
        assert not report.audit_ok
    finally:
        ws.audit_log.write_text(original, encoding="utf-8")
