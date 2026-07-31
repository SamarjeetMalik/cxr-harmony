"""Role-based access to the catalogue.

The point of separating these roles is that the modelling team does not need the
report text. After scrubbing, free-text prose remains the highest residual
re-identification surface in the whole cohort — a scrubber can miss an unusual
spelling or an unanticipated construction in a way that a tag-level profile
cannot — so the population with routine access to it should be as small as the
work allows. A modeller training a classifier needs images, labels and
demographics; giving them prose as well adds risk and buys nothing.

This is an application-level control and is honest about being one: it constrains
what the provided query helpers return, not what someone with the database file
can read. The file itself sits inside the secure zone under the sharing
agreement, and that boundary is enforced by infrastructure, not by this module.
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    """Who is asking."""

    #: Full access, including report text. Data engineering and governance.
    CURATOR = "curator"
    #: Images, labels, demographics. No report prose.
    MODELLER = "modeller"
    #: Aggregates and provenance only. No patient-level rows.
    AUDITOR = "auditor"


#: Which entities each role may read at row level.
ROW_ACCESS: dict[Role, frozenset[str]] = {
    Role.CURATOR: frozenset(
        {"patients", "studies", "series", "instances", "reports", "labels", "splits"}
    ),
    Role.MODELLER: frozenset(
        {"patients", "studies", "series", "instances", "labels", "splits"}
    ),
    Role.AUDITOR: frozenset(),
}


class AccessDenied(PermissionError):
    """Raised when a role reaches for something it is not entitled to."""


def require(role: Role, entity: str) -> None:
    """Raise unless ``role`` may read ``entity`` at row level."""
    if entity not in ROW_ACCESS.get(role, frozenset()):
        raise AccessDenied(
            f"role {role.value!r} may not read {entity!r} at row level"
            + (
                "; report text is restricted to the curator role"
                if entity == "reports"
                else ""
            )
        )


__all__ = ["ROW_ACCESS", "AccessDenied", "Role", "require"]
