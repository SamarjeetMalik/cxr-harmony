"""Clinical fact extraction, performed *before* redaction.

There is an ordering constraint in this pipeline that is easy to get wrong and
expensive to discover late. The confidentiality profile removes SeriesDescription
and ImageComments, because both are free-text fields that routinely carry names
and dates. But site B encodes the radiographic projection in SeriesDescription,
and site A ships its labels in ImageComments — so applying the profile first
destroys the only copy of the clinical content the cohort exists to hold.

The resolution is to read what is needed while it is still there. This module
pulls the site-native clinical values into a side record; harmonisation later maps
those strings onto the canonical vocabulary. The extracted record contains no
direct identifiers: the national ID and local MRN are consumed to derive the
pseudonym and are not carried forward, and the birth date is consumed to compute
an age and then discarded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime

import pydicom

#: Ages above this are re-expressed at the cap. An exact age in the 90+ tail is
#: quasi-identifying in a national cohort, and it is the one age band where a
#: single value can single a person out within a hospital.
AGE_CAP = 89


@dataclass
class SiteFacts:
    """Site-native clinical values, read before the profile destroys them."""

    source_path: str
    site_id: str
    view_native: str
    laterality_native: str
    labels_native: str
    study_date_native: str
    sex_native: str
    body_part: str
    modality: str
    age_years: int | None
    #: Retained so harmonisation can join a sidecar CSV or a report file. It is an
    #: internal working-directory key and never appears in a release.
    accession_key: str

    def to_dict(self) -> dict:
        return asdict(self)


def _first_nonempty(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def read_private_date(ds: pydicom.Dataset) -> str:
    """Recover a study date from a known private block, if one is present.

    Site B leaves the standard date tags empty and files ``DD-MM-YYYY`` into a
    private block. The block is about to be removed wholesale, so the value has
    to be lifted here or it is lost.
    """
    try:
        block = ds.private_block(0x0033, "MIMS RIS EXPORT")
    except KeyError:
        return ""
    try:
        return str(block[0x01].value or "").strip()
    except KeyError:
        return ""


def parse_native_date(text: str) -> date | None:
    """Parse the date formats seen across the contributing sites."""
    text = (text or "").strip()
    if not text:
        return None
    for fmt in ("%Y%m%d", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def compute_age(birth_date: str, study_date: date | None) -> int | None:
    """Age in whole years at the study, capped. Returns ``None`` if undeterminable."""
    parsed_birth = parse_native_date(birth_date)
    if parsed_birth is None or study_date is None:
        return None
    years = (study_date - parsed_birth).days // 365
    if years < 0:
        return None
    return min(years, AGE_CAP)


def extract_facts(ds: pydicom.Dataset, *, source_path: str, site_id: str) -> SiteFacts:
    """Read the site's clinical values while the identifying header is still intact."""
    native_date_text = _first_nonempty(
        getattr(ds, "StudyDate", ""),
        getattr(ds, "ContentDate", ""),
        getattr(ds, "AcquisitionDate", ""),
        read_private_date(ds),
    )
    study_date = parse_native_date(native_date_text)

    return SiteFacts(
        source_path=source_path,
        site_id=site_id,
        # Sites put the projection in whichever of these they favour.
        view_native=_first_nonempty(
            getattr(ds, "ViewPosition", ""), getattr(ds, "SeriesDescription", "")
        ),
        laterality_native=str(getattr(ds, "Laterality", "") or "").strip(),
        labels_native=str(getattr(ds, "ImageComments", "") or "").strip(),
        study_date_native=native_date_text,
        sex_native=str(getattr(ds, "PatientSex", "") or "").strip(),
        body_part=str(getattr(ds, "BodyPartExamined", "") or "").strip().upper(),
        modality=str(getattr(ds, "Modality", "") or "").strip().upper(),
        age_years=compute_age(str(getattr(ds, "PatientBirthDate", "") or ""), study_date),
        accession_key=str(getattr(ds, "AccessionNumber", "") or "").strip(),
    )


def read_linkage_identifiers(ds: pydicom.Dataset) -> tuple[str, str]:
    """Return ``(national_id, local_mrn)``, consumed to derive the pseudonym.

    Neither value is carried into any output. They exist in memory only long
    enough to be fed through the keyed HMAC.
    """
    national = str(getattr(ds, "OtherPatientIDs", "") or "").strip()
    local = str(getattr(ds, "PatientID", "") or "").strip()
    return national, local


__all__ = [
    "AGE_CAP",
    "SiteFacts",
    "compute_age",
    "extract_facts",
    "parse_native_date",
    "read_linkage_identifiers",
    "read_private_date",
]
