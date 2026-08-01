"""Adapter for the Open-i / Indiana University Chest X-ray collection.

3,955 real radiology reports released by the U.S. National Library of Medicine
under CC BY-NC-ND, drawn from two Indiana hospital systems. Each report carries
sectioned free text *and* manually assigned MeSH terms, which is what makes it
usable as ground truth rather than merely as a sample of prose.

Two properties of this corpus matter for how it is used here.

**It is already de-identified**, by the NLM, who replaced removed spans with the
literal token ``XXXX``. So it cannot be used to demonstrate that a de-identifier
removes PHI — there is none left to remove. What it can do, and what nothing
synthetic can, is test whether the report *parser*, the *negation scope* and the
*label extractor* survive contact with how radiologists actually write.

**The ``XXXX`` placeholder is itself a hazard.** It appears mid-sentence
("no XXXX of a pleural effusion", "Normal chest x-XXXX"), so a naive parser can
have its negation scope silently broken by it. That is a realistic problem: every
de-identified corpus a partner site ships will carry some placeholder convention.

Source: https://openi.nlm.nih.gov/faq  —  Demner-Fushman et al., JAMIA 2016.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

from ..schema.vocab import Finding

#: NLM's redaction placeholder. Runs of it collapse to a single marker.
REDACTION_TOKEN = "XXXX"

#: MeSH major topics mapped onto the canonical vocabulary.
#:
#: Open-i terms are slash-qualified ("Cardiomegaly/mild", "Pulmonary
#: Atelectasis/base"), so matching is on the head term. Terms with no canonical
#: counterpart map to OTHER rather than being dropped, so coverage stays visible.
#:
#: Head-only matching was questioned, because the corpus also contains terms whose
#: head is *anatomy* and whose qualifier is the finding — "Lung/hypoinflation",
#: "Thoracic Vertebrae/degenerative". Those look like the head-only rule would
#: lose the finding. Measured over all 3,927 parsable reports, it does not: no
#: qualifier anywhere in the corpus names a canonical finding that its head
#: missed, so head-only and head-plus-qualifier scoring produce identical
#: reference sets and identical out-of-vocabulary verdicts. That equivalence is
#: pinned by ``test_qualifier_parsing_does_not_change_scoring`` rather than left
#: as a comment, because it is a property of this corpus and could stop holding
#: on another one.
#:
#: The qualifiers still matter for *describing* the out-of-vocabulary tail — see
#: :func:`mesh_concepts` — just not for scoring against it.
MESH_TO_FINDING: dict[str, Finding] = {
    "normal": Finding.NO_FINDING,
    "cardiomegaly": Finding.CARDIOMEGALY,
    "pleural effusion": Finding.PLEURAL_EFFUSION,
    "pleural effusions": Finding.PLEURAL_EFFUSION,
    "consolidation": Finding.CONSOLIDATION,
    "pneumonia": Finding.CONSOLIDATION,
    "pneumothorax": Finding.PNEUMOTHORAX,
    "pulmonary edema": Finding.PULMONARY_EDEMA,
    "pulmonary congestion": Finding.PULMONARY_EDEMA,
    "pulmonary atelectasis": Finding.ATELECTASIS,
    "atelectasis": Finding.ATELECTASIS,
    # The corpus uses the plural 348 times. Its absence here was an oversight
    # rather than a decision — the plural of "pleural effusion" is mapped two
    # lines above. The correction is small: 346 of those 348 studies already
    # carry ATELECTASIS through another term, so only 2 gain it.
    "atelectases": Finding.ATELECTASIS,
    "nodule": Finding.NODULE,
    "pulmonary nodule": Finding.NODULE,
    "solitary pulmonary nodule": Finding.NODULE,
    "fractures, bone": Finding.FRACTURE,
    "fracture": Finding.FRACTURE,
    "rib fractures": Finding.FRACTURE,
    "tuberculosis": Finding.TUBERCULOSIS,
    "tuberculosis, pulmonary": Finding.TUBERCULOSIS,
}

#: Terms that assert the study is unremarkable.
NORMAL_TERMS = {"normal", "no indexing"}

#: Qualifiers that grade or place a finding without naming one.
#:
#: Drawn from the 87 distinct qualifiers the corpus actually uses, not guessed.
#: Counting these as out-of-vocabulary "terms" would inflate the tail with words
#: like "mild" that no vocabulary would ever want to express.
MESH_MODIFIERS = frozenset(
    {
        # Degree.
        "mild", "moderate", "severe", "small", "large", "borderline", "prominent",
        "multiple", "scattered", "chronic", "acute", "focal", "diffuse", "patchy",
        "streaky", "round", "enlarged", "elevated", "flattened", "blunted",
        # Laterality and position. These name where, never what.
        "right", "left", "bilateral", "anterior", "posterior", "lateral",
        "upper", "lower", "middle", "base", "apex",
    }
)


def mesh_concepts(term: str) -> list[str]:
    """Split a slash-qualified MeSH term into the concepts it actually names.

    ``"Thoracic Vertebrae/degenerative/mild"`` yields ``["thoracic vertebrae",
    "degenerative"]`` — the grading qualifier is dropped, the finding kept.

    **This is for description, not scoring.** Scoring uses the head term alone,
    and switching it to use this function changes nothing: measured over the
    whole corpus, no qualifier names a canonical finding its head missed. What
    this does change is how the out-of-vocabulary tail is *characterised*.
    Head-only counting reports ``Lung/hypoinflation`` as the term "lung", which
    makes the tail look like a list of anatomy when it is really a list of
    findings the vocabulary cannot express. Naming "hypoinflation" is the honest
    description of what is missing.
    """
    parts = [p.strip().lower() for p in term.split("/") if p.strip()]
    if not parts:
        return []
    concepts = [parts[0]]
    concepts.extend(q for q in parts[1:] if q not in MESH_MODIFIERS)
    seen: set[str] = set()
    return [c for c in concepts if not (c in seen or seen.add(c))]

_SECTION_LABELS = ("COMPARISON", "INDICATION", "FINDINGS", "IMPRESSION")


@dataclass
class OpeniReport:
    """One parsed Open-i record."""

    uid: str
    sections: dict[str, str] = field(default_factory=dict)
    mesh_terms: list[str] = field(default_factory=list)
    findings: set[Finding] = field(default_factory=set)
    #: True when the annotation says the study is normal.
    is_normal: bool = False

    @property
    def clinical_text(self) -> str:
        """FINDINGS and IMPRESSION only, never INDICATION."""
        return "\n".join(
            self.sections.get(label, "") for label in ("FINDINGS", "IMPRESSION")
        ).strip()

    @property
    def has_text(self) -> bool:
        return bool(self.clinical_text)

    def as_report_text(self) -> str:
        """Re-render in the layout the pipeline's report parser expects."""
        parts = []
        for label in _SECTION_LABELS:
            body = self.sections.get(label, "").strip()
            if body:
                parts.append(f"{label}:\n{body}\n")
        return "\n".join(parts)


def normalise_redactions(text: str) -> str:
    """Collapse runs of the NLM placeholder to a single marker.

    ``XXXX XXXX opacities`` and ``no XXXX of a pleural effusion`` both occur. Left
    as-is the repeated tokens add noise to sentence splitting without carrying
    information, and a run of them can push a finding outside its negation scope.
    """
    text = re.sub(rf"(?:\b{REDACTION_TOKEN}\b[\s,]*)+", f"{REDACTION_TOKEN} ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def parse_report(path: Path) -> OpeniReport | None:
    """Parse one Open-i XML record. Returns ``None`` if it has no usable content."""
    try:
        tree = ElementTree.parse(path)
    except ElementTree.ParseError:
        return None
    root = tree.getroot()

    uid_el = root.find("uId")
    uid = uid_el.get("id") if uid_el is not None else path.stem

    sections: dict[str, str] = {}
    for element in root.iter("AbstractText"):
        label = (element.get("Label") or "").strip().upper()
        body = (element.text or "").strip()
        if label in _SECTION_LABELS and body:
            sections[label] = normalise_redactions(body)

    mesh_terms: list[str] = []
    mesh = root.find("MeSH")
    if mesh is not None:
        for element in list(mesh.iter("major")) + list(mesh.iter("automatic")):
            term = (element.text or "").strip()
            if term:
                mesh_terms.append(term)

    findings: set[Finding] = set()
    is_normal = False
    for term in mesh_terms:
        head = term.split("/")[0].strip().lower()
        if head in NORMAL_TERMS:
            is_normal = True
            continue
        mapped = MESH_TO_FINDING.get(head)
        if mapped is not None and mapped is not Finding.NO_FINDING:
            findings.add(mapped)

    if is_normal and not findings:
        findings.add(Finding.NO_FINDING)

    report = OpeniReport(
        uid=uid,
        sections=sections,
        mesh_terms=mesh_terms,
        findings=findings,
        is_normal=is_normal,
    )
    return report if report.has_text else None


def load_corpus(directory: Path, *, limit: int | None = None) -> list[OpeniReport]:
    """Load every parseable report under ``directory``, in stable id order."""
    paths = sorted(
        Path(directory).rglob("*.xml"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    if limit is not None:
        paths = paths[:limit]

    reports = [parse_report(path) for path in paths]
    return [r for r in reports if r is not None]


__all__ = [
    "MESH_MODIFIERS",
    "MESH_TO_FINDING",
    "REDACTION_TOKEN",
    "OpeniReport",
    "load_corpus",
    "mesh_concepts",
    "normalise_redactions",
    "parse_report",
]
