"""Canonical entity models.

These are the contract between the pipeline stages. Anything that reaches the
catalogue is one of these; anything a partner site sends that cannot be coerced
into one of these is quarantined with a reason rather than silently coerced.

Two invariants are enforced here rather than left to convention:

* No model carries a direct identifier. ``Patient.pseudo_id`` is the output of a
  keyed HMAC, and there is no field in which a name, MRN or accession number
  could legitimately be stored. A PHI leak therefore has to be a schema
  violation, not merely an oversight.
* ``Patient.pseudo_id`` is global, not per-site. The same person imaged at two
  partner hospitals collapses to one identity, which is what makes the
  leakage-free split in :mod:`cxr_harmony.release` meaningful.
"""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .vocab import (
    Finding,
    LabelSource,
    Laterality,
    PatientPosition,
    QuarantineReason,
    ReportSection,
    Sex,
    Split,
    ViewPosition,
)

_PSEUDO_ID_RE = re.compile(r"^[a-f0-9]{16}$")
_UID_RE = re.compile(r"^[0-9]+(\.[0-9]+)*$")
_SITE_ID_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,15}$")


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class Patient(_Base):
    """A person, identified only by a keyed pseudonym."""

    pseudo_id: str = Field(description="Lowercase hex HMAC-SHA256 prefix, stable across sites")
    sex: Sex = Sex.UNKNOWN
    age_years: int | None = Field(
        default=None,
        ge=0,
        le=89,
        description=(
            "Age at study, capped at 89. Ages above 89 are re-expressed as 89 because "
            "an exact age in the 90+ tail is quasi-identifying in a national cohort."
        ),
    )

    @field_validator("pseudo_id")
    @classmethod
    def _check_pseudo_id(cls, v: str) -> str:
        if not _PSEUDO_ID_RE.match(v):
            raise ValueError(f"pseudo_id must be 16 lowercase hex characters, got {v!r}")
        return v


class Study(_Base):
    """One imaging encounter."""

    study_uid: str
    pseudo_patient_id: str
    site_id: str
    study_date: date | None = Field(
        default=None,
        description="Date after per-patient offset. Intervals within a patient are preserved.",
    )
    modality: str
    body_part: str | None = None
    patient_position: PatientPosition = PatientPosition.UNKNOWN

    @field_validator("study_uid")
    @classmethod
    def _check_uid(cls, v: str) -> str:
        if not _UID_RE.match(v) or len(v) > 64:
            raise ValueError(f"not a valid DICOM UID: {v!r}")
        return v

    @field_validator("site_id")
    @classmethod
    def _check_site(cls, v: str) -> str:
        if not _SITE_ID_RE.match(v):
            raise ValueError(f"site_id must be upper-case alphanumeric, got {v!r}")
        return v


class Series(_Base):
    """A series within a study, carrying the projection geometry."""

    series_uid: str
    study_uid: str
    view_position: ViewPosition = ViewPosition.UNKNOWN
    laterality: Laterality = Laterality.NOT_APPLICABLE

    @field_validator("series_uid", "study_uid")
    @classmethod
    def _check_uid(cls, v: str) -> str:
        if not _UID_RE.match(v) or len(v) > 64:
            raise ValueError(f"not a valid DICOM UID: {v!r}")
        return v


class Instance(_Base):
    """A single stored image."""

    sop_uid: str
    series_uid: str
    relative_path: str = Field(description="Path relative to the de-identified store root")
    sha256: str
    rows: int = Field(gt=0)
    columns: int = Field(gt=0)
    bits_stored: int = Field(gt=0, le=16)
    photometric_interpretation: str
    pixel_redacted: bool = Field(
        default=False,
        description="True when burned-in text was detected and blacked out.",
    )

    @field_validator("sha256")
    @classmethod
    def _check_digest(cls, v: str) -> str:
        if not re.fullmatch(r"[a-f0-9]{64}", v):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return v


class Report(_Base):
    """A radiology report after section parsing and text scrubbing."""

    study_uid: str
    sections: dict[ReportSection, str] = Field(default_factory=dict)
    scrubbed: bool = True
    redaction_count: int = Field(default=0, ge=0)


class Label(_Base):
    """One finding asserted about one study."""

    study_uid: str
    finding: Finding
    present: bool
    source: LabelSource
    site_native_value: str | None = Field(
        default=None,
        description="The site's original string, retained for QC traceability of the mapping.",
    )


class SplitAssignment(_Base):
    """Partition membership, assigned per patient rather than per study."""

    pseudo_patient_id: str
    split: Split


class QuarantineRecord(_Base):
    """An object rejected at ingest, kept so that losses are auditable."""

    source_path: str
    site_id: str
    reason: QuarantineReason
    detail: str = ""


class CanonicalDataset(_Base):
    """The whole harmonised corpus, as handed to the release stage."""

    patients: list[Patient] = Field(default_factory=list)
    studies: list[Study] = Field(default_factory=list)
    series: list[Series] = Field(default_factory=list)
    instances: list[Instance] = Field(default_factory=list)
    reports: list[Report] = Field(default_factory=list)
    labels: list[Label] = Field(default_factory=list)


#: Every model that participates in the emitted JSON Schema bundle.
EXPORTED_MODELS: tuple[type[BaseModel], ...] = (
    Patient,
    Study,
    Series,
    Instance,
    Report,
    Label,
    SplitAssignment,
    QuarantineRecord,
    CanonicalDataset,
)
