"""Per-site adapter configuration.

A site's conventions live in a YAML file, not in Python. That is the whole design
decision of this module. Receiving a new partner's first delivery should be an
afternoon of writing a config and reading a QC report, not a code change, a
review, and a release — because there will be a fourth site, and a fifth, and the
person onboarding them will not be the person who wrote this.

The corollary is that a config can be wrong, so every mapping failure is counted
and surfaced rather than silently defaulted. A site that starts sending a new
label string appears in the QC report as an unmapped value with a count, which is
the signal to go and ask them what it means.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatternRule(BaseModel):
    """A regular expression tried when no exact mapping matched."""

    model_config = ConfigDict(extra="forbid")

    match: str
    value: str

    @field_validator("match")
    @classmethod
    def _compilable(cls, v: str) -> str:
        """Reject a malformed expression at load time, naming the offending pattern.

        ``re.error`` does not derive from ``ValueError``, so pydantic would let it
        escape as a bare regex error with no indication of which config file or
        which rule produced it.
        """
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"invalid regular expression {v!r}: {exc}") from exc
        return v


class ValueMapping(BaseModel):
    """Exact lookups first, then ordered patterns, then a default."""

    model_config = ConfigDict(extra="forbid")

    map: dict[str, str] = Field(default_factory=dict)
    patterns: list[PatternRule] = Field(default_factory=list)
    default: str | None = None

    def resolve(self, value: str) -> tuple[str | None, bool]:
        """Return ``(canonical_value, matched)``.

        ``matched`` is False when the default had to be used, which is what makes
        an unmapped value countable instead of invisible.
        """
        text = (value or "").strip()
        if not text:
            return self.default, False

        normalised = text.upper()
        for native, canonical in self.map.items():
            if native.strip().upper() == normalised:
                return canonical, True

        for rule in self.patterns:
            if re.search(rule.match, text, re.I):
                return rule.value, True

        return self.default, False


class LabelConfig(BaseModel):
    """Where a site's labels come from and how they are spelled."""

    model_config = ConfigDict(extra="forbid")

    #: ``image_comments``, ``sidecar_csv`` or ``report``.
    source: str
    separator: str = ";"
    map: dict[str, str] = Field(default_factory=dict)
    #: Sidecar only: file name and the columns to join and read.
    sidecar_file: str | None = None
    sidecar_key_column: str | None = None
    sidecar_value_column: str | None = None

    @field_validator("source")
    @classmethod
    def _known_source(cls, v: str) -> str:
        allowed = {"image_comments", "sidecar_csv", "report"}
        if v not in allowed:
            raise ValueError(f"label source must be one of {sorted(allowed)}, got {v!r}")
        return v


class SiteConfig(BaseModel):
    """Everything needed to read one site's delivery into the canonical schema."""

    model_config = ConfigDict(extra="forbid")

    site_id: str
    description: str = ""
    view_position: ValueMapping = Field(default_factory=ValueMapping)
    laterality: ValueMapping = Field(default_factory=ValueMapping)
    sex: ValueMapping = Field(default_factory=ValueMapping)
    labels: LabelConfig
    date_formats: list[str] = Field(default_factory=lambda: ["%Y%m%d"])


def load_site_configs(directory: Path) -> dict[str, SiteConfig]:
    """Load every ``*.yaml`` in ``directory``, keyed by site id."""
    directory = Path(directory)
    configs: dict[str, SiteConfig] = {}
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = SiteConfig.model_validate(data)
        if config.site_id in configs:
            raise ValueError(f"duplicate site_id {config.site_id!r} in {path}")
        configs[config.site_id] = config
    return configs


__all__ = ["LabelConfig", "PatternRule", "SiteConfig", "ValueMapping", "load_site_configs"]
