"""Partner-site profiles describing how each hospital's export diverges.

Every site below records the same clinical facts, and no two record them the same
way. That is not an invented difficulty: projection encoded in a free-text series
description, dates in a private tag in ``DD-MM-YYYY``, sex as an HL7-style
numeric code, and labels arriving through three different channels are all
conventions that real contributed archives exhibit.

The harmonisation configs in ``configs/sites/*.yaml`` are written against these
profiles. Keeping the two in separate places is deliberate: the config is what a
data engineer would author on receiving a new site's first delivery, and it must
be possible to get it wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schema.vocab import Finding


@dataclass(frozen=True)
class SiteProfile:
    """How one contributing site encodes its export."""

    site_id: str
    institution_name: str
    institution_address: str
    city: str

    #: ``str.format`` template taking ``n`` (and possibly ``year``) for the local MRN.
    mrn_template: str

    #: How the radiographic projection is expressed, and where.
    view_encoding: dict[str, str]
    #: ``"view_position_tag"``, ``"series_description"`` — where a reader must look.
    view_channel: str

    #: ``"standard"`` puts StudyDate in (0008,0020); ``"private_ddmmyyyy"`` leaves
    #: that tag empty and writes ``DD-MM-YYYY`` into a private block instead.
    date_channel: str

    #: ``"image_comments"``, ``"sidecar_csv"`` or ``"report_only"``.
    label_channel: str
    #: Canonical finding to the site's own spelling.
    label_encoding: dict[Finding, str]

    #: Canonical sex code to the site's own spelling.
    sex_encoding: dict[str, str]

    body_part: str
    #: Fraction of images on which the acquisition console burns identifiers in.
    burn_in_rate: float
    #: Fraction of patients for whom a national health identifier is recorded.
    abha_rate: float
    #: Whether the site populates Laterality, and how.
    laterality_encoding: dict[str, str] = field(default_factory=dict)


_ENGLISH_LABELS = {
    Finding.NO_FINDING: "No Finding",
    Finding.CARDIOMEGALY: "Cardiomegaly",
    Finding.PLEURAL_EFFUSION: "Pleural Effusion",
    Finding.CONSOLIDATION: "Consolidation",
    Finding.PNEUMOTHORAX: "Pneumothorax",
    Finding.PULMONARY_EDEMA: "Pulmonary Edema",
    Finding.ATELECTASIS: "Atelectasis",
    Finding.NODULE: "Nodule",
    Finding.FRACTURE: "Rib Fracture",
    Finding.TUBERCULOSIS: "Pulmonary Tuberculosis",
}

_ABBREVIATED_LABELS = {
    Finding.NO_FINDING: "NAD",
    Finding.CARDIOMEGALY: "CM",
    Finding.PLEURAL_EFFUSION: "PE",
    Finding.CONSOLIDATION: "CONS",
    Finding.PNEUMOTHORAX: "PTX",
    Finding.PULMONARY_EDEMA: "EDEMA",
    Finding.ATELECTASIS: "ATEL",
    Finding.NODULE: "NOD",
    Finding.FRACTURE: "FRAC",
    Finding.TUBERCULOSIS: "KOCHS",
}


SITE_A = SiteProfile(
    site_id="SITE_A",
    institution_name="Sunrise Medical College and Hospital",
    institution_address="Kothrud, Pune, Maharashtra 411038",
    city="Pune",
    mrn_template="SMC-{n:06d}",
    view_encoding={"PA": "PA", "AP": "AP", "LATERAL": "LL"},
    view_channel="view_position_tag",
    date_channel="standard",
    label_channel="image_comments",
    label_encoding=_ENGLISH_LABELS,
    sex_encoding={"M": "M", "F": "F"},
    body_part="CHEST",
    burn_in_rate=0.0,
    abha_rate=0.6,
)

SITE_B = SiteProfile(
    site_id="SITE_B",
    institution_name="Meridian Institute of Medical Sciences",
    institution_address="Palarivattom, Kochi, Kerala 682025",
    city="Kochi",
    mrn_template="MIMS/{year}/{n:05d}",
    view_encoding={
        "PA": "CHEST PA ERECT",
        "AP": "CHEST AP SUPINE PORTABLE",
        "LATERAL": "CHEST LAT",
    },
    view_channel="series_description",
    date_channel="private_ddmmyyyy",
    label_channel="sidecar_csv",
    label_encoding=_ABBREVIATED_LABELS,
    sex_encoding={"M": "MALE", "F": "FEMALE"},
    body_part="CHEST",
    burn_in_rate=0.85,
    abha_rate=0.7,
)

SITE_C = SiteProfile(
    site_id="SITE_C",
    institution_name="Northstar Diagnostics Centre",
    institution_address="GS Road, Guwahati, Assam 781005",
    city="Guwahati",
    mrn_template="ND{n:07d}",
    # A legacy arrow convention for beam direction, still seen in older RIS exports.
    view_encoding={"PA": "P->A", "AP": "A->P", "LATERAL": "LAT"},
    view_channel="view_position_tag",
    date_channel="standard",
    label_channel="report_only",
    label_encoding=_ENGLISH_LABELS,
    # HL7 v2 administrative sex codes rather than the DICOM CS values.
    sex_encoding={"M": "1", "F": "2"},
    body_part="THORAX",
    burn_in_rate=0.35,
    abha_rate=0.5,
    laterality_encoding={"L": "LEFT", "R": "RIGHT"},
)

SITES: tuple[SiteProfile, ...] = (SITE_A, SITE_B, SITE_C)

SITES_BY_ID = {s.site_id: s for s in SITES}
