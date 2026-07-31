"""Command line interface.

Each pipeline stage is a separate command rather than one monolithic run. That is
how the work actually goes: a delivery arrives, ingest is run and the quarantine
file is read, a site config is corrected, harmonisation is re-run. Forcing a
re-ingest and a re-de-identification to fix a label mapping would make the loop
long enough that people stop closing it.

Every stage records itself in the audit log, so the sequence is reconstructible
afterwards even though it was driven by hand.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .catalog import Role, build_catalog, record_splits, summary_counts
from .deid import deidentify
from .governance import AuditLog, run_verification, write_policy_docs
from .harmonize import harmonize as harmonize_stage
from .ingest import ingest as ingest_stage
from .qc import run_checks, write_report
from .release import SplitRatios, build_release
from .reports import process_reports
from .schema import CanonicalDataset, write_schemas
from .synth import generate_corpus
from .workspace import Workspace, read_jsonl

app = typer.Typer(
    add_completion=False,
    help="Multi-site chest radiograph ingestion, de-identification, harmonisation and release.",
)
console = Console()

SOURCE_RECORD = "source.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _workspace(work: Path) -> Workspace:
    return Workspace(work).ensure()


def _remember_source(ws: Workspace, src: Path) -> None:
    (ws.root / SOURCE_RECORD).write_text(
        json.dumps({"source": str(Path(src).resolve())}, indent=2) + "\n", encoding="utf-8"
    )


def _recall_source(ws: Workspace, src: Path | None) -> Path:
    """Later stages default to the delivery ingest was pointed at."""
    if src is not None:
        return Path(src)
    record = ws.root / SOURCE_RECORD
    if not record.exists():
        raise typer.BadParameter(
            "no --src given and no source recorded; run `cxr-harmony ingest --src ...` first"
        )
    return Path(json.loads(record.read_text(encoding="utf-8"))["source"])


def _audit(ws: Workspace, stage: str, action: str, **counts: int) -> None:
    AuditLog(ws.audit_log).record(stage, action, counts=counts, timestamp=_now())


@app.command()
def synth(
    out: Path = typer.Option(..., help="Directory to write the synthetic delivery into."),
    seed: int = typer.Option(20260731, help="Seed; a given seed reproduces the corpus exactly."),
    patients: int = typer.Option(48, help="Number of patients to generate."),
    cross_site: int = typer.Option(8, help="How many are imaged at two hospitals."),
    image_size: int = typer.Option(512, help="Pixel dimension of the generated radiographs."),
) -> None:
    """Generate a synthetic three-site delivery."""
    truth = generate_corpus(
        out, seed=seed, n_patients=patients, n_cross_site=cross_site, image_size=image_size
    )
    console.print(
        f"[green]Generated[/] {truth['n_studies']} studies for {truth['n_patients']} patients "
        f"across 3 sites in {out}"
    )
    console.print(
        f"  {len(truth['cross_site_patients'])} patients were imaged at more than one site"
    )


@app.command()
def ingest(
    src: Path = typer.Option(..., help="Incoming delivery root."),
    work: Path = typer.Option(Path("work"), help="Working directory."),
) -> None:
    """Index and validate an incoming delivery."""
    ws = _workspace(work)
    result = ingest_stage(src, ws)
    _remember_source(ws, src)
    _audit(ws, "ingest", "scanned", accepted=result.n_accepted, quarantined=result.n_quarantined)

    console.print(f"[green]Accepted[/] {result.n_accepted} objects")
    if result.n_quarantined:
        console.print(f"[yellow]Quarantined[/] {result.n_quarantined}:")
        for reason, count in result.reasons().items():
            console.print(f"    {reason}: {count}")


@app.command()
def deid(
    work: Path = typer.Option(Path("work"), help="Working directory."),
    src: Path = typer.Option(None, help="Delivery root; defaults to the one ingested."),
) -> None:
    """Apply the confidentiality profile and write the de-identified store."""
    ws = _workspace(work)
    result = deidentify(_recall_source(ws, src), ws)
    _audit(ws, "deid", "applied", objects=result.n_objects, pixel_redacted=result.n_redacted)

    console.print(f"[green]De-identified[/] {result.n_objects} objects")
    console.print(f"  {result.n_redacted} had burned-in annotation redacted")
    console.print(f"  {result.n_cross_site_patients} patients linked across sites")


@app.command()
def reports(
    work: Path = typer.Option(Path("work"), help="Working directory."),
    src: Path = typer.Option(None, help="Delivery root; defaults to the one ingested."),
) -> None:
    """Section, scrub and label the radiology reports."""
    ws = _workspace(work)
    result = process_reports(_recall_source(ws, src), ws)
    _audit(
        ws, "reports", "scrubbed", reports=result.n_reports, redactions=result.total_redactions
    )
    console.print(f"[green]Processed[/] {result.n_reports} reports")
    console.print(f"  {result.total_redactions} redactions applied")


@app.command()
def harmonize(
    work: Path = typer.Option(Path("work"), help="Working directory."),
    configs: Path = typer.Option(Path("configs/sites"), help="Site adapter configs."),
    src: Path = typer.Option(None, help="Delivery root; defaults to the one ingested."),
) -> None:
    """Map site conventions onto the canonical schema."""
    ws = _workspace(work)
    result = harmonize_stage(_recall_source(ws, src), ws, configs)
    _audit(ws, "harmonize", "mapped", studies=result.n_studies, unmapped=result.n_unmapped)

    console.print(f"[green]Harmonised[/] {result.n_studies} studies")
    for site, count in sorted(result.per_site.items()):
        console.print(f"    {site}: {count}")
    if result.unmapped:
        console.print(f"[yellow]{result.n_unmapped} values had no mapping:[/]")
        for entry in result.unmapped[:10]:
            console.print(f"    {entry.site_id} {entry.field}={entry.value!r} x{entry.count}")


@app.command()
def catalog(
    work: Path = typer.Option(Path("work"), help="Working directory."),
) -> None:
    """Load the canonical dataset into the queryable catalogue."""
    ws = _workspace(work)
    dataset = CanonicalDataset.model_validate_json(ws.canonical.read_text(encoding="utf-8"))
    stats = build_catalog(dataset, ws)
    _audit(ws, "catalog", "built", studies=stats.n_studies, instances=stats.n_instances)

    table = Table(title="Catalogue")
    table.add_column("Entity")
    table.add_column("Rows", justify="right")
    for name, value in stats.to_dict().items():
        if name != "per_site":
            table.add_row(name.removeprefix("n_"), str(value))
    console.print(table)


@app.command()
def qc(
    work: Path = typer.Option(Path("work"), help="Working directory."),
    strict: bool = typer.Option(False, help="Exit non-zero on warnings as well as failures."),
) -> None:
    """Run quality control over the harmonised cohort."""
    ws = _workspace(work)
    dataset = CanonicalDataset.model_validate_json(ws.canonical.read_text(encoding="utf-8"))
    unmapped_path = ws.root / "unmapped_values.json"
    unmapped = (
        json.loads(unmapped_path.read_text(encoding="utf-8"))
        if unmapped_path.exists()
        else []
    )

    report = run_checks(
        dataset, unmapped=unmapped, quarantined=list(read_jsonl(ws.quarantine))
    )
    md_path, _ = write_report(report, ws)
    _audit(ws, "qc", "checked", failures=len(report.failures), warnings=len(report.warnings))

    if report.failures:
        console.print(f"[red]{len(report.failures)} failures[/]")
        for check in report.failures:
            console.print(f"    {check.name}: {check.message}")
    if report.warnings:
        console.print(f"[yellow]{len(report.warnings)} warnings[/]")
        for check in report.warnings:
            console.print(f"    {check.name}: {check.message}")
    if report.passed and not report.warnings:
        console.print("[green]All checks passed[/]")
    console.print(f"  report written to {md_path}")

    if report.failures or (strict and report.warnings):
        raise typer.Exit(code=1)


@app.command()
def release(
    work: Path = typer.Option(Path("work"), help="Working directory."),
    version: str = typer.Option(..., help="Release version, e.g. v1.0.0."),
    train: float = typer.Option(0.70, help="Target training proportion."),
    val: float = typer.Option(0.15, help="Target validation proportion."),
    test: float = typer.Option(0.15, help="Target test proportion."),
    salt: str = typer.Option("cxr-harmony-v1", help="Split salt; changing it re-splits."),
    stamp: bool = typer.Option(True, help="Record the cut time in release.json."),
) -> None:
    """Cut an immutable, content-addressed release."""
    ws = _workspace(work)
    dataset = CanonicalDataset.model_validate_json(ws.canonical.read_text(encoding="utf-8"))
    result = build_release(
        dataset,
        ws,
        version=version,
        ratios=SplitRatios(train=train, val=val, test=test),
        split_salt=salt,
        created_at=_now() if stamp else None,
    )
    record_splits(ws, {k: v.value for k, v in result.assignments.items()}, version)
    _audit(ws, "release", "cut", files=result.n_files)

    console.print(f"[green]Release {version}[/]  digest {result.dataset_digest[:16]}...")
    table = Table()
    table.add_column("Split")
    table.add_column("Patients", justify="right")
    table.add_column("Studies", justify="right")
    for split in ("train", "val", "test"):
        n_patients = sum(1 for s in result.assignments.values() if s.value == split)
        table.add_row(split, str(n_patients), str(result.studies_per_split[split]))
    console.print(table)
    console.print(f"  written to {result.directory}")


@app.command()
def verify(
    work: Path = typer.Option(Path("work"), help="Working directory."),
    phi_file: Path = typer.Option(
        None,
        help="Ground-truth file listing identifiers that must not survive (synthetic runs only).",
    ),
) -> None:
    """Verify de-identification, the audit chain, and every release."""
    ws = _workspace(work)

    phi_values: list[str] = []
    candidate = phi_file
    if candidate is None:
        recorded = ws.root / SOURCE_RECORD
        if recorded.exists():
            source = Path(json.loads(recorded.read_text(encoding="utf-8"))["source"])
            if (source / "_ground_truth.json").exists():
                candidate = source / "_ground_truth.json"
    if candidate is not None and candidate.exists():
        phi_values = json.loads(candidate.read_text(encoding="utf-8")).get("phi_values", [])

    report = run_verification(ws, phi_values=phi_values)

    console.print(f"De-identification: {report.deid_checked} objects checked")
    if report.deid_violations:
        console.print(f"[red]  {len(report.deid_violations)} violations[/]")
        for violation in report.deid_violations[:10]:
            console.print(f"      {violation['kind']}: {violation['detail'][:70]}")
    else:
        console.print("[green]  no violations[/]")
        if phi_values:
            console.print(f"  searched for {len(phi_values)} known identifier strings")

    console.print(f"Audit chain: {'[green]intact[/]' if report.audit_ok else '[red]broken[/]'}")
    for name, ok in sorted(report.releases_ok.items()):
        console.print(f"Release {name}: {'[green]verified[/]' if ok else '[red]failed[/]'}")

    if not report.passed:
        raise typer.Exit(code=1)
    console.print("[green]Verification passed[/]")


@app.command()
def schema(
    out: Path = typer.Option(Path("docs/schema"), help="Directory for the JSON Schema bundle."),
) -> None:
    """Emit the canonical schema as JSON Schema for partner sites."""
    written = write_schemas(out)
    console.print(f"[green]Wrote[/] {len(written)} schema files to {out}")


@app.command()
def docs(
    out: Path = typer.Option(Path("docs"), help="Directory for the governance documents."),
) -> None:
    """Write the governance documents."""
    for path in write_policy_docs(out):
        console.print(f"[green]Wrote[/] {path}")


@app.command()
def summary(
    work: Path = typer.Option(Path("work"), help="Working directory."),
    role: str = typer.Option("auditor", help="curator, modeller or auditor."),
) -> None:
    """Print cohort aggregates, as the given role would see them."""
    ws = _workspace(work)
    counts = summary_counts(ws, Role(role))
    table = Table(title=f"Cohort ({role})")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key in ("patients", "studies", "instances"):
        table.add_row(key, str(counts[key]))
    for site, count in sorted(counts["per_site"].items()):
        table.add_row(f"studies @ {site}", str(count))
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
