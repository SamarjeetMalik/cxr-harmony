"""Working-directory layout shared by every pipeline stage.

One deliberate property of this layout: the pipeline never writes an identifiable
copy of anything. Ingest only *indexes* the incoming delivery — it records paths,
digests and header facts, and copies no pixels. The first and only image bytes
this tool writes are the de-identified ones under ``deid/``.

That matters beyond tidiness. Every additional copy of identifiable data is
another location to secure, audit, and eventually destroy under a data-sharing
agreement. A pipeline that stages a raw copy "just for convenience" has silently
doubled the site's exposure.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Workspace:
    """Resolved paths for one pipeline run."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    # --- Stage outputs -------------------------------------------------
    @property
    def raw_manifest(self) -> Path:
        """Index of accepted incoming objects. Contains paths, not pixels."""
        return self.root / "raw_manifest.jsonl"

    @property
    def quarantine(self) -> Path:
        return self.root / "quarantine.jsonl"

    @property
    def deid_store(self) -> Path:
        """The only place this tool writes image data."""
        return self.root / "deid"

    @property
    def deid_manifest(self) -> Path:
        return self.root / "deid_manifest.jsonl"

    @property
    def uid_map(self) -> Path:
        return self.root / "uid_map.json"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def reports_manifest(self) -> Path:
        return self.root / "reports_manifest.jsonl"

    @property
    def canonical(self) -> Path:
        return self.root / "canonical.json"

    @property
    def catalog_db(self) -> Path:
        return self.root / "catalog.db"

    @property
    def qc_dir(self) -> Path:
        return self.root / "qc"

    @property
    def releases(self) -> Path:
        return self.root / "releases"

    @property
    def audit_log(self) -> Path:
        return self.root / "audit.jsonl"

    @property
    def key_file(self) -> Path:
        """Pseudonymisation key. Excluded from version control by .gitignore."""
        return self.root / "pseudonym.key"

    def ensure(self) -> Workspace:
        self.root.mkdir(parents=True, exist_ok=True)
        return self


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write ``records`` as JSON Lines, sorted keys for reproducibility. Returns count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSON Lines file, or nothing if it does not exist."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
