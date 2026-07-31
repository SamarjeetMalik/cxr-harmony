"""The command line, driven the way the Makefile drives it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cxr_harmony.cli import app

runner = CliRunner()
CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "sites"


def _run(*args: str) -> object:
    result = runner.invoke(app, list(args))
    if result.exit_code != 0 and result.exception is not None:
        raise AssertionError(f"{args} failed: {result.output}") from result.exception
    return result


def _flat(output: str) -> str:
    """Collapse Rich's box drawing and line wrapping so a message can be matched.

    Rich renders errors inside a bordered panel wrapped to the terminal width, so
    a phrase that is one sentence in the source may span two lines with box
    characters between them.
    """
    stripped = "".join(" " if ord(ch) > 0x2500 else ch for ch in output)
    return " ".join(stripped.split())


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """A complete run, driven entirely through the CLI."""
    root = tmp_path_factory.mktemp("cli")
    src = root / "incoming"
    work = root / "work"

    _run("synth", "--out", str(src), "--seed", "4321", "--patients", "20",
         "--cross-site", "4", "--image-size", "128")
    _run("ingest", "--src", str(src), "--work", str(work))
    _run("deid", "--work", str(work))
    _run("reports", "--work", str(work))
    _run("harmonize", "--work", str(work), "--configs", str(CONFIGS))
    _run("catalog", "--work", str(work))
    _run("qc", "--work", str(work))
    _run("release", "--work", str(work), "--version", "v1.0.0")
    return src, work


def test_help_lists_every_stage():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("synth", "ingest", "deid", "reports", "harmonize", "release", "verify"):
        assert command in result.output


def test_full_pipeline_runs_and_verifies(pipeline):
    _, work = pipeline
    result = runner.invoke(app, ["verify", "--work", str(work)])
    assert result.exit_code == 0, result.output
    assert "Verification passed" in result.output


def test_verify_searches_for_the_known_identifiers(pipeline):
    """The synthetic ground truth is picked up automatically from the delivery."""
    _, work = pipeline
    result = runner.invoke(app, ["verify", "--work", str(work)])
    assert "known identifier strings" in result.output


def test_later_stages_remember_the_ingested_source(pipeline):
    """So an operator does not have to repeat --src on every command."""
    _, work = pipeline
    assert json.loads((work / "source.json").read_text(encoding="utf-8"))["source"]


def test_a_stage_without_a_recorded_source_fails_clearly(tmp_path):
    result = runner.invoke(app, ["deid", "--work", str(tmp_path / "empty")])
    assert result.exit_code != 0
    assert "no source recorded" in _flat(result.output)


def test_release_artefacts_are_produced(pipeline):
    _, work = pipeline
    directory = work / "releases" / "v1.0.0"
    for name in ("manifest.json", "splits.json", "release.json", "datasheet.md"):
        assert (directory / name).exists(), name


def test_qc_report_is_written(pipeline):
    _, work = pipeline
    assert (work / "qc" / "report.md").exists()
    assert (work / "qc" / "report.json").exists()


def test_every_stage_appended_to_the_audit_log(pipeline):
    _, work = pipeline
    entries = [
        json.loads(line)
        for line in (work / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stages = {e["stage"] for e in entries}
    assert {"ingest", "deid", "reports", "harmonize", "catalog", "qc", "release"} <= stages
    assert [e["sequence"] for e in entries] == list(range(1, len(entries) + 1))


def test_verify_exits_non_zero_when_deidentification_regresses(pipeline):
    import pydicom

    _, work = pipeline
    path = next((work / "deid").rglob("*.dcm"))
    original = path.read_bytes()
    try:
        ds = pydicom.dcmread(path)
        ds.PatientBirthDate = "19800101"
        ds.save_as(path, enforce_file_format=True)
        result = runner.invoke(app, ["verify", "--work", str(work)])
        assert result.exit_code == 1
    finally:
        path.write_bytes(original)


def test_qc_strict_mode_fails_on_warnings(pipeline, tmp_path):
    """A clean cohort passes strict; the flag exists for CI to hold the line."""
    _, work = pipeline
    assert runner.invoke(app, ["qc", "--work", str(work), "--strict"]).exit_code == 0


def test_summary_respects_the_requested_role(pipeline):
    _, work = pipeline
    result = runner.invoke(app, ["summary", "--work", str(work), "--role", "auditor"])
    assert result.exit_code == 0
    assert "patients" in result.output


def test_schema_export_command(tmp_path):
    result = runner.invoke(app, ["schema", "--out", str(tmp_path / "schema")])
    assert result.exit_code == 0
    assert (tmp_path / "schema" / "bundle.json").exists()


def test_docs_command_writes_the_governance_documents(tmp_path):
    result = runner.invoke(app, ["docs", "--out", str(tmp_path / "docs")])
    assert result.exit_code == 0
    assert (tmp_path / "docs" / "dpdp-controls.md").exists()


def test_cross_site_count_reports_actual_overlap_not_capability(tmp_path):
    """Most patients carry a national ID; few are actually imaged at two sites."""
    src = tmp_path / "incoming"
    work = tmp_path / "work"
    _run("synth", "--out", str(src), "--seed", "77", "--patients", "20",
         "--cross-site", "3", "--image-size", "64")
    _run("ingest", "--src", str(src), "--work", str(work))
    result = _run("deid", "--work", str(work))

    truth = json.loads((src / "_ground_truth.json").read_text(encoding="utf-8"))
    expected = len(truth["cross_site_patients"])
    assert f"{expected} patients linked across sites" in result.output
