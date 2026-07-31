"""Sectioning of free-text radiology reports.

Sectioning is not cosmetic. Label extraction must run over FINDINGS and
IMPRESSION and nowhere else, because the INDICATION states what the clinician was
worried about, not what was seen. A report headed "persistent cough, evaluate for
tuberculosis" that concludes with clear lungs would otherwise be labelled
tuberculosis-positive — and since the indication is exactly the clinical suspicion
the model is meant to resolve, that error correlates with the outcome and will
inflate validation performance rather than degrade it.
"""

from __future__ import annotations

import re

from ..schema.vocab import ReportSection

#: Section headings as the contributing sites write them.
_HEADINGS: list[tuple[ReportSection, str]] = [
    (ReportSection.INDICATION, r"INDICATION|CLINICAL\s+HISTORY|HISTORY"),
    (ReportSection.TECHNIQUE, r"TECHNIQUE|PROCEDURE|EXAMINATION\s+TECHNIQUE"),
    (ReportSection.COMPARISON, r"COMPARISON|PRIOR\s+STUDIES?"),
    (ReportSection.FINDINGS, r"FINDINGS|OBSERVATIONS?"),
    (ReportSection.IMPRESSION, r"IMPRESSION|CONCLUSION|OPINION|ADVICE"),
]

_SECTION_PATTERNS: list[tuple[ReportSection, re.Pattern[str]]] = [
    (section, re.compile(rf"^\s*(?:{alternatives})\s*:?\s*$", re.I))
    for section, alternatives in _HEADINGS
]

#: Lines that terminate the body of the report.
_TRAILER = re.compile(
    r"^\s*(Reported\s+by|Electronically\s+signed|This\s+report\s+was\s+generated|"
    r"Signed\s+by|Verified\s+by)\b",
    re.I,
)


def parse_sections(text: str) -> dict[ReportSection, str]:
    """Split a report into its recognised sections.

    Text before the first recognised heading is discarded: that is the demographic
    header block, which is entirely PHI and holds no clinical content.
    """
    sections: dict[ReportSection, list[str]] = {}
    current: ReportSection | None = None

    for line in (text or "").splitlines():
        heading = _match_heading(line)
        if heading is not None:
            current = heading
            sections.setdefault(current, [])
            continue
        if _TRAILER.match(line):
            current = None
            continue
        if current is not None:
            sections[current].append(line)

    return {
        section: "\n".join(lines).strip()
        for section, lines in sections.items()
        if "\n".join(lines).strip()
    }


def _match_heading(line: str) -> ReportSection | None:
    for section, pattern in _SECTION_PATTERNS:
        if pattern.match(line):
            return section
    return None


def clinical_text(sections: dict[ReportSection, str]) -> str:
    """The portion a label may legitimately be drawn from.

    Deliberately excludes INDICATION and COMPARISON: the first states the
    suspicion, the second describes a different examination.
    """
    parts = [
        sections.get(ReportSection.FINDINGS, ""),
        sections.get(ReportSection.IMPRESSION, ""),
    ]
    return "\n".join(p for p in parts if p).strip()


__all__ = ["clinical_text", "parse_sections"]
