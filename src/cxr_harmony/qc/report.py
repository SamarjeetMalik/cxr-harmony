"""Rendering the QC report.

Two outputs, for two readers. The JSON is what a CI job or a dashboard consumes.
The Markdown is what gets pasted into an email to the partner site when something
needs asking about, so it leads with what is wrong rather than with how many rows
were loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..workspace import Workspace
from .checks import QCReport, Severity

_ICON = {True: "pass", False: "FAIL"}


def render_markdown(report: QCReport) -> str:
    """A report that leads with the problems."""
    lines: list[str] = ["# Cohort quality control", ""]

    if report.passed and not report.warnings:
        lines += ["All checks passed.", ""]
    else:
        lines += [
            f"**{len(report.failures)} failures, {len(report.warnings)} warnings.**",
            "",
        ]

    if report.failures:
        lines += ["## Failures", ""]
        for check in report.failures:
            lines.append(f"- **{check.name}** — {check.message}")
        lines.append("")

    if report.warnings:
        lines += ["## Warnings", ""]
        for check in report.warnings:
            lines.append(f"- **{check.name}** — {check.message}")
        lines.append("")

    stats = report.stats
    lines += [
        "## Cohort",
        "",
        "| | |",
        "|---|---|",
        f"| Patients | {stats.get('n_patients', 0)} |",
        f"| Studies | {stats.get('n_studies', 0)} |",
        f"| Series | {stats.get('n_series', 0)} |",
        f"| Images | {stats.get('n_instances', 0)} |",
        f"| Reports | {stats.get('n_reports', 0)} |",
        f"| Labels | {stats.get('n_labels', 0)} |",
        f"| Patients imaged at >1 site | {stats.get('n_cross_site_patients', 0)} |",
        "",
    ]

    per_site = stats.get("studies_per_site", {})
    if per_site:
        lines += ["## Studies per site", "", "| Site | Studies |", "|---|---:|"]
        lines += [f"| {site} | {count} |" for site, count in per_site.items()]
        lines.append("")

    view_by_site = stats.get("view_by_site", {})
    if view_by_site:
        views = sorted({v for counts in view_by_site.values() for v in counts})
        lines += [
            "## Projection by site",
            "",
            "Divergence here is the confound most likely to let a model identify the",
            "contributing hospital rather than the finding.",
            "",
            "| Site | " + " | ".join(views) + " |",
            "|---" * (len(views) + 1) + "|",
        ]
        for site, counts in view_by_site.items():
            lines.append(
                f"| {site} | " + " | ".join(str(counts.get(v, 0)) for v in views) + " |"
            )
        lines.append("")

    prevalence = stats.get("label_prevalence_pct", {})
    if prevalence:
        findings = sorted({f for site in prevalence.values() for f in site})
        lines += [
            "## Label prevalence by site (%)",
            "",
            "| Site | " + " | ".join(findings) + " |",
            "|---" * (len(findings) + 1) + "|",
        ]
        for site, counts in prevalence.items():
            lines.append(
                f"| {site} | " + " | ".join(f"{counts.get(f, 0):g}" for f in findings) + " |"
            )
        lines.append("")

    lines += ["## All checks", "", "| Check | Severity | Result | Detail |", "|---|---|---|---|"]
    for check in report.checks:
        result = "pass" if check.passed else _ICON[False]
        if check.severity is Severity.INFO:
            result = "info"
        lines.append(
            f"| {check.name} | {check.severity.value} | {result} | {check.message} |"
        )
    lines.append("")

    return "\n".join(lines)


def write_report(report: QCReport, workspace: Workspace) -> tuple[Path, Path]:
    """Write ``qc/report.md`` and ``qc/report.json``. Returns both paths."""
    workspace.qc_dir.mkdir(parents=True, exist_ok=True)
    md_path = workspace.qc_dir / "report.md"
    json_path = workspace.qc_dir / "report.json"

    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return md_path, json_path


__all__ = ["render_markdown", "write_report"]
