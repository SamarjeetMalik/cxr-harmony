"""Append-only, hash-chained audit log.

Every stage records what it did to how many objects. The requirement this serves
is mundane and inevitable: a partner site, an ethics committee, or a grant auditor
will eventually ask what happened to a particular delivery, and "we ran the
pipeline" is not an answer.

Each entry carries the digest of the one before it, so the log is a chain. That
does not make it tamper-*proof* — anyone who can write the file can rewrite the
whole chain — but it makes it tamper-*evident* against the realistic failure,
which is a single entry being quietly edited or dropped long after the fact. An
inconsistent chain is detectable by :func:`verify_chain` without needing a copy of
the original.

Entries never contain patient identifiers. They contain counts, stage names,
digests and configuration, which is what an audit actually needs. A log that
records who was processed rather than what was done becomes another copy of the
cohort, with the same handling obligations and none of the protections.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

GENESIS = "0" * 64


@dataclass
class AuditEntry:
    """One recorded action."""

    sequence: int
    stage: str
    action: str
    counts: dict[str, int] = field(default_factory=dict)
    detail: dict = field(default_factory=dict)
    timestamp: str | None = None
    previous_hash: str = GENESIS
    entry_hash: str = ""

    def payload(self) -> dict:
        """The fields covered by the hash."""
        return {
            "sequence": self.sequence,
            "stage": self.stage,
            "action": self.action,
            "counts": self.counts,
            "detail": self.detail,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
        }

    def compute_hash(self) -> str:
        canonical = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {**self.payload(), "entry_hash": self.entry_hash}


class AuditLog:
    """A hash-chained JSONL log."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def entries(self) -> list[AuditEntry]:
        if not self.path.exists():
            return []
        out: list[AuditEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            out.append(
                AuditEntry(
                    sequence=data["sequence"],
                    stage=data["stage"],
                    action=data["action"],
                    counts=data.get("counts", {}),
                    detail=data.get("detail", {}),
                    timestamp=data.get("timestamp"),
                    previous_hash=data.get("previous_hash", GENESIS),
                    entry_hash=data.get("entry_hash", ""),
                )
            )
        return out

    def record(
        self,
        stage: str,
        action: str,
        *,
        counts: dict[str, int] | None = None,
        detail: dict | None = None,
        timestamp: str | None = None,
    ) -> AuditEntry:
        """Append one entry, chained to the current tail."""
        existing = self.entries()
        previous = existing[-1].entry_hash if existing else GENESIS

        entry = AuditEntry(
            sequence=len(existing) + 1,
            stage=stage,
            action=action,
            counts=counts or {},
            detail=detail or {},
            timestamp=timestamp,
            previous_hash=previous,
        )
        entry.entry_hash = entry.compute_hash()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
        return entry

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Check every link. Returns ``(ok, problems)``."""
        problems: list[str] = []
        previous = GENESIS

        for index, entry in enumerate(self.entries(), start=1):
            if entry.sequence != index:
                problems.append(f"entry {index}: sequence is {entry.sequence}")
            if entry.previous_hash != previous:
                problems.append(
                    f"entry {entry.sequence}: previous_hash does not match the prior entry"
                )
            recomputed = entry.compute_hash()
            if entry.entry_hash != recomputed:
                problems.append(f"entry {entry.sequence}: contents do not match its hash")
            previous = entry.entry_hash

        return not problems, problems


__all__ = ["GENESIS", "AuditEntry", "AuditLog"]
