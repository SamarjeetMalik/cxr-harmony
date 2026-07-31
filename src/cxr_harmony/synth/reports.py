"""Free-text radiology report synthesis.

Reports are where the hardest PHI lives. Header tags are a bounded, enumerable
set; prose is not, and a report that reads "compared with the film of 12-04-2024
performed at Sunrise Medical College" carries three identifiers that no tag-level
profile will ever touch.

The finding sentences below are deliberately phrased the way reports actually
are — hedged, indirect, and rarely containing the canonical label as a literal
substring ("the cardiac silhouette is enlarged", not "cardiomegaly"). That is
what makes rule-based label extraction in :mod:`cxr_harmony.reports` a genuine
test rather than a keyword search.
"""

from __future__ import annotations

import random

from ..schema.vocab import Finding

FINDING_SENTENCES: dict[Finding, tuple[str, ...]] = {
    Finding.CARDIOMEGALY: (
        "The cardiac silhouette is enlarged with a cardiothoracic ratio exceeding 0.5.",
        "There is enlargement of the cardiac shadow.",
        "Cardiomegaly is noted.",
    ),
    Finding.PLEURAL_EFFUSION: (
        "Blunting of the {side} costophrenic angle is noted.",
        "There is a moderate {side}-sided pleural fluid collection.",
        "Free fluid is seen in the {side} pleural space.",
    ),
    # Phrased as the diagnosis rather than as the descriptor. Evaluation against
    # the Open-i corpus showed that "air-space opacity" and "infiltrate" are
    # indexed by radiologist annotators under a separate Opacity heading, not as
    # consolidation, so generating them here would have taught the extractor a
    # conflation that real annotation does not make.
    Finding.CONSOLIDATION: (
        "Patchy consolidation is seen in the {side} lower zone.",
        "There is dense consolidation involving the {side} mid zone.",
        "Homogeneous opacity with air bronchograms in the {side} lung field.",
    ),
    Finding.PNEUMOTHORAX: (
        "A thin visceral pleural line is visible along the {side} apex with absent "
        "lung markings peripherally.",
        "There is a small apical pneumothorax on the {side}.",
    ),
    Finding.PULMONARY_EDEMA: (
        "Bilateral perihilar haziness with upper lobe diversion is present.",
        "Interstitial septal thickening consistent with fluid overload is seen.",
    ),
    Finding.ATELECTASIS: (
        "Linear bands of collapse are seen at the {side} base.",
        "There is volume loss in the {side} lower lobe with elevation of the hemidiaphragm.",
    ),
    Finding.NODULE: (
        "A well-circumscribed {size} mm nodular density is noted in the {side} upper zone.",
        "There is a rounded opacity measuring approximately {size} mm in the {side} lung.",
    ),
    Finding.FRACTURE: (
        "A cortical break is seen involving the {side} {rib}th rib laterally.",
        "There is an undisplaced fracture of the {side} {rib}th rib.",
    ),
    Finding.TUBERCULOSIS: (
        "Fibrocavitary changes with volume loss are seen in the {side} upper zone, "
        "suggestive of post-primary infection.",
        "Nodular opacities with cavitation in the {side} apex; correlate with sputum studies.",
    ),
    Finding.NO_FINDING: (
        "Both lung fields are clear. The cardiac silhouette is within normal limits. "
        "Costophrenic angles are sharp and the bony thorax appears intact.",
        "No focal parenchymal opacity, effusion or pneumothorax is identified. "
        "The mediastinal contours are unremarkable.",
    ),
}

IMPRESSION_PHRASES: dict[Finding, tuple[str, ...]] = {
    Finding.CARDIOMEGALY: ("Cardiomegaly.",),
    Finding.PLEURAL_EFFUSION: ("{Side} pleural effusion.",),
    Finding.CONSOLIDATION: ("{Side} zone consolidation, likely infective.",),
    Finding.PNEUMOTHORAX: ("Small {side}-sided pneumothorax.",),
    Finding.PULMONARY_EDEMA: ("Features of pulmonary venous congestion.",),
    Finding.ATELECTASIS: ("{Side} basal atelectasis.",),
    Finding.NODULE: ("Solitary pulmonary nodule, {side} upper zone. Further evaluation advised.",),
    Finding.FRACTURE: ("Undisplaced {side} rib fracture.",),
    Finding.TUBERCULOSIS: ("Findings suspicious for pulmonary tuberculosis.",),
    Finding.NO_FINDING: ("No significant abnormality detected.",),
}

INDICATIONS = (
    "Cough with fever for {days} days.",
    "Breathlessness on exertion, evaluate.",
    "Pre-operative screening.",
    "Chest pain, rule out acute pathology.",
    "Follow-up of known {condition}.",
    "Persistent cough for {weeks} weeks, evaluate for tuberculosis.",
)

CONDITIONS = ("hypertension", "diabetes mellitus", "COPD", "ischaemic heart disease")


def _render_sentence(template: str, rng: random.Random) -> str:
    side = rng.choice(("left", "right"))
    return template.format(
        side=side,
        Side=side.capitalize(),
        size=rng.choice((8, 9, 11, 12, 14, 16, 18)),
        rib=rng.choice((5, 6, 7, 8, 9)),
    )


def build_report(
    *,
    rng: random.Random,
    findings: list[Finding],
    patient_display_name: str,
    mrn: str,
    abha: str | None,
    age: int,
    sex: str,
    study_date_text: str,
    accession: str,
    referring_physician: str,
    reporting_radiologist: str,
    institution_name: str,
    institution_address: str,
    phone: str,
    examination: str,
    prior_study_date: str | None,
) -> str:
    """Compose a full report, header block and all.

    The header block is PHI-dense on purpose: name, local MRN, national health
    identifier, exact age, study date, accession, and two clinician names, all in
    prose that no DICOM profile governs.
    """
    indication = rng.choice(INDICATIONS).format(
        days=rng.randint(2, 10),
        weeks=rng.randint(3, 8),
        condition=rng.choice(CONDITIONS),
    )

    if prior_study_date:
        comparison = (
            f"Compared with the previous radiograph dated {prior_study_date} "
            f"performed at {institution_name}."
        )
    else:
        comparison = "No previous imaging available for comparison."

    positives = [f for f in findings if f is not Finding.NO_FINDING]
    if positives:
        finding_lines = [
            _render_sentence(rng.choice(FINDING_SENTENCES[f]), rng) for f in positives
        ]
        finding_lines.append("The visualised bony thorax and soft tissues are otherwise normal.")
        impressions = [
            _render_sentence(rng.choice(IMPRESSION_PHRASES[f]), rng) for f in positives
        ]
    else:
        finding_lines = [_render_sentence(rng.choice(FINDING_SENTENCES[Finding.NO_FINDING]), rng)]
        impressions = [IMPRESSION_PHRASES[Finding.NO_FINDING][0]]

    impression_block = "\n".join(f"{i}. {t}" for i, t in enumerate(impressions, start=1))
    abha_line = f"ABHA Number  : {abha}\n" if abha else ""

    return f"""{institution_name.upper()}
{institution_address}
Tel: {phone}

DEPARTMENT OF RADIODIAGNOSIS

Patient Name : {patient_display_name}
MRN          : {mrn}
{abha_line}Age / Sex    : {age} Y / {sex}
Study Date   : {study_date_text}
Accession No : {accession}
Referred by  : {referring_physician}

EXAMINATION: {examination}

INDICATION:
{indication}

TECHNIQUE:
Digital radiograph of the chest acquired in the standard projection.

COMPARISON:
{comparison}

FINDINGS:
{chr(10).join(finding_lines)}

IMPRESSION:
{impression_block}

Reported by {reporting_radiologist}
Electronically signed on {study_date_text}
This report was generated at {institution_name}, {institution_address}.
"""
