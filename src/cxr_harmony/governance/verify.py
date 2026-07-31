"""The end-to-end governance verification pass.

Runs the independent de-identification check over the written store, re-hashes
the current release, and confirms the audit chain is intact. This is what a
``cxr-harmony verify`` invocation runs, and what CI runs, so that "the pipeline
completed" and "the pipeline produced something safe to release" are two separate
claims with two separate pieces of evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..deid.verify import verify_store
from ..release.builder import verify_release
from ..workspace import Workspace
from .audit import AuditLog


@dataclass
class GovernanceReport:
    deid_checked: int = 0
    deid_violations: list[dict] = field(default_factory=list)
    audit_ok: bool = True
    audit_problems: list[str] = field(default_factory=list)
    releases_ok: dict[str, bool] = field(default_factory=dict)
    release_problems: dict[str, list[str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            not self.deid_violations
            and self.audit_ok
            and all(self.releases_ok.values())
        )

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "deid": {
                "checked": self.deid_checked,
                "violations": self.deid_violations[:100],
                "n_violations": len(self.deid_violations),
            },
            "audit": {"ok": self.audit_ok, "problems": self.audit_problems},
            "releases": {
                "ok": self.releases_ok,
                "problems": self.release_problems,
            },
        }


def run_verification(
    workspace: Workspace,
    *,
    phi_values: list[str] | None = None,
) -> GovernanceReport:
    """Verify de-identification, the audit chain, and every cut release."""
    report = GovernanceReport()

    deid = verify_store(workspace.deid_store, phi_values=phi_values)
    report.deid_checked = deid.n_checked
    report.deid_violations = [v.to_dict() for v in deid.violations]

    report.audit_ok, report.audit_problems = AuditLog(workspace.audit_log).verify_chain()

    if workspace.releases.exists():
        for directory in sorted(p for p in workspace.releases.iterdir() if p.is_dir()):
            ok, problems = verify_release(directory)
            report.releases_ok[directory.name] = ok
            if problems:
                report.release_problems[directory.name] = problems

    workspace.qc_dir.mkdir(parents=True, exist_ok=True)
    (workspace.qc_dir / "governance.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def write_policy_docs(directory: Path) -> list[Path]:
    """Write the governance documents that ship with the repository."""
    from .dpdp import render_markdown

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dpdp_path = directory / "dpdp-controls.md"
    dpdp_path.write_text(render_markdown(), encoding="utf-8", newline="\n")
    return [dpdp_path]


__all__ = ["GovernanceReport", "run_verification", "write_policy_docs"]
