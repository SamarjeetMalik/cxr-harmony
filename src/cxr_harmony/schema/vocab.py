"""Canonical controlled vocabularies.

Partner sites encode the same clinical concept in mutually incompatible ways: one
writes ``PA``, another ``Postero-Anterior``, a third buries the projection in a
free-text series description. Every such value is mapped onto exactly one member
of the enumerations below before it reaches the catalogue, so that a downstream
query means the same thing regardless of which hospital contributed the study.

Site-specific spellings live in ``configs/sites/*.yaml``, never here.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """A string enum whose members compare and serialise as their value."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class ViewPosition(StrEnum):
    """Radiographic projection."""

    PA = "PA"
    AP = "AP"
    LATERAL = "LATERAL"
    LATERAL_DECUBITUS = "LATERAL_DECUBITUS"
    OBLIQUE = "OBLIQUE"
    UNKNOWN = "UNKNOWN"


class Laterality(StrEnum):
    """Imaged side, where the site records one."""

    LEFT = "L"
    RIGHT = "R"
    BILATERAL = "B"
    NOT_APPLICABLE = "NA"
    UNKNOWN = "UNKNOWN"


class Sex(StrEnum):
    """Administrative sex as recorded by the contributing site."""

    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"


class PatientPosition(StrEnum):
    """Patient orientation during acquisition."""

    ERECT = "ERECT"
    SUPINE = "SUPINE"
    UNKNOWN = "UNKNOWN"


class Finding(StrEnum):
    """Canonical finding vocabulary for chest radiography.

    Deliberately coarse: it is the intersection that every partner site can
    populate reliably. Site-native labels that fall outside it map to ``OTHER``
    rather than being silently dropped, so the loss is visible in QC.
    """

    NO_FINDING = "NO_FINDING"
    CARDIOMEGALY = "CARDIOMEGALY"
    PLEURAL_EFFUSION = "PLEURAL_EFFUSION"
    CONSOLIDATION = "CONSOLIDATION"
    PNEUMOTHORAX = "PNEUMOTHORAX"
    PULMONARY_EDEMA = "PULMONARY_EDEMA"
    ATELECTASIS = "ATELECTASIS"
    NODULE = "NODULE"
    FRACTURE = "FRACTURE"
    TUBERCULOSIS = "TUBERCULOSIS"
    OTHER = "OTHER"


class LabelSource(StrEnum):
    """Provenance of a label, which governs how much it may be trusted."""

    SITE_STRUCTURED = "SITE_STRUCTURED"
    REPORT_RULE = "REPORT_RULE"
    RADIOLOGIST = "RADIOLOGIST"


class ReportSection(StrEnum):
    """Sections recognised in a free-text radiology report."""

    INDICATION = "INDICATION"
    TECHNIQUE = "TECHNIQUE"
    COMPARISON = "COMPARISON"
    FINDINGS = "FINDINGS"
    IMPRESSION = "IMPRESSION"


class Split(StrEnum):
    """Dataset partition assigned at release time."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class QuarantineReason(StrEnum):
    """Why an incoming object was rejected at ingest."""

    UNREADABLE = "UNREADABLE"
    NOT_DICOM = "NOT_DICOM"
    WRONG_MODALITY = "WRONG_MODALITY"
    WRONG_BODY_PART = "WRONG_BODY_PART"
    MISSING_REQUIRED_TAG = "MISSING_REQUIRED_TAG"
    NO_PIXEL_DATA = "NO_PIXEL_DATA"
    DUPLICATE_CONTENT = "DUPLICATE_CONTENT"
    #: BodyPartExamined was absent and a classifier judged the image non-chest.
    #: Distinct from WRONG_BODY_PART, which means the tag itself said so: this one
    #: records that the sender told us nothing and we had to look at the pixels.
    BODY_PART_UNVERIFIED = "BODY_PART_UNVERIFIED"


#: Modalities accepted by a chest-radiograph pipeline. ``DX`` is digital
#: radiography, ``CR`` computed radiography; both appear in practice.
ACCEPTED_MODALITIES = frozenset({"DX", "CR"})

#: BodyPartExamined values that denote a chest study.
ACCEPTED_BODY_PARTS = frozenset({"CHEST", "THORAX", "LUNG"})
