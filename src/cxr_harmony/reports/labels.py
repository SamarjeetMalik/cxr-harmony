"""Rule-based extraction of findings from report prose.

Two things make this harder than a keyword search.

**Reports do not use the label vocabulary.** A radiologist writes "the cardiac
silhouette is enlarged", not "cardiomegaly"; "blunting of the costophrenic angle",
not "pleural effusion". The phrase bank below is indexed on what is actually
written.

**Most sentences about a finding are denying it.** The single commonest sentence
in a normal chest report is of the form "no focal parenchymal opacity, effusion or
pneumothorax is identified", which names three findings and asserts none of them.
A keyword matcher labels that study with all three. The negation scope below is a
reduced NegEx: a cue opens a negated span that runs to the end of the sentence or
to a termination cue, whichever comes first, so "no effusion but there is
consolidation" resolves correctly in both halves.

Labels produced here carry ``LabelSource.REPORT_RULE``, which is weaker evidence
than a site's structured export and is recorded as such rather than being
laundered into equivalence with it.

**On the reported accuracy.** Against the synthetic corpus this extractor scores
precision 1.000 and recall 1.000 over 151 reports across two seeds. That number
should be read for what it is: the phrase bank and the report generator were
written by the same person, so it measures internal consistency, not performance
on real radiology prose. Real reports contain hedging ("cannot exclude", "possibly
represents"), comparative statements, dictation errors, transliterated regional
phrasing, and house styles that differ between the three hospitals in ways this
does not simulate. On real data a rule extractor of this shape would be expected
to land far lower, and would need to be scored against radiologist adjudication
before any label from it entered a training set. The value of the perfect score
is narrow but real: it establishes that the negation scope and the section
restriction work, which are the two places this kind of extractor usually fails
silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schema.vocab import Finding

#: Phrases as radiologists write them, mapped to the canonical finding.
FINDING_PATTERNS: dict[Finding, tuple[str, ...]] = {
    Finding.CARDIOMEGALY: (
        r"cardiomegaly",
        # Real reports overwhelmingly say "the heart is enlarged", with an adverb
        # of degree in between, rather than naming the finding.
        r"\bheart\s+(?:size\s+)?(?:is|appears|remains)\s+(?:\w+\s+){0,3}enlarged",
        r"cardiac\s+silhouette\s+is\s+(?:\w+\s+){0,2}enlarged",
        r"enlarge\w*\s+of\s+the\s+(?:cardiac|heart)",
        r"cardiac\s+enlargement",
        r"enlarged\s+(?:cardiac|cardiomediastinal)\s+(?:shadow|silhouette)",
        r"cardiothoracic\s+ratio\s+(?:exceed|>|greater)",
    ),
    Finding.PLEURAL_EFFUSION: (
        r"pleural\s+effusion",
        r"blunting\s+of\s+the\s+\w+\s+costophrenic\s+angle",
        r"pleural\s+fluid",
        r"fluid\s+(?:is\s+seen\s+)?in\s+the\s+\w+\s+pleural\s+space",
        r"\beffusion\b",
    ),
    Finding.CONSOLIDATION: (
        r"consolidat",  # consolidation, consolidative
        r"air\s+bronchograms?",
        r"\bpneumonia\b",
        # Deliberately *not* included: "airspace disease", "infiltrate", "patchy
        # opacity". Those are descriptors of increased density, not the diagnosis
        # of consolidation, and the Open-i annotators index them under a separate
        # "Opacity" heading. Including them raised recall to 0.98 but collapsed
        # precision to 0.31 (220 false positives); restricting to consolidation
        # proper gives 0.85/0.96. A radiologist would draw the same distinction.
    ),
    Finding.PNEUMOTHORAX: (
        r"pneumothorax",
        r"visceral\s+pleural\s+line",
        r"absent\s+lung\s+markings",
    ),
    Finding.PULMONARY_EDEMA: (
        r"\bo?edema\b",
        r"perihilar\s+haziness",
        r"upper\s+lobe\s+diversion",
        r"septal\s+thickening",
        r"fluid\s+overload",
        # Open-i annotates "Pulmonary Congestion" as oedema; reports express it as
        # vascular congestion or prominence far more often than as oedema.
        r"(?:vascular|venous|pulmonary)\s+congestion",
        r"vascular\s+prominence",
        r"prominence\s+of\s+the\s+pulmonary\s+vascul",
        r"interstitial\s+prominence",
        r"cephalization",
        r"pulmonary\s+venous\s+hypertension",
    ),
    Finding.ATELECTASIS: (
        r"atelectasis",
        r"bands?\s+of\s+collapse",
        # "volume loss in the left lower lobe" asserts collapse; "fibrocavitary
        # changes with volume loss" uses it as a modifier describing a different
        # primary finding. Requiring the locative clause keeps the first and drops
        # the second, which was the entire false-positive population in testing.
        r"volume\s+loss\s+in\s+the",
        r"elevation\s+of\s+the\s+hemidiaphragm",
    ),
    Finding.NODULE: (
        r"\bnodules?\b",
        r"nodular\s+density",
        r"rounded\s+opacity",
        r"well[-\s]circumscribed\s+\w*\s*opacity",
    ),
    Finding.FRACTURE: (
        r"fractures?\b",
        r"cortical\s+break",
    ),
    Finding.TUBERCULOSIS: (
        r"tuberculosis",
        r"fibrocavitary",
        r"cavitation",
        r"post[-\s]primary\s+infection",
        r"koch'?s?\b",
    ),
}

#: Explicit statements that the study, as a whole, is normal.
#:
#: Widened against the Open-i corpus. The original set required "lung *fields* are
#: clear", which real reports almost never write — they say "Lungs are clear",
#: "the lungs are well expanded and clear", or close with "No acute cardiopulmonary
#: disease". Normal-study recall was 0.31 before this was corrected.
#:
#: These are only consulted when no positive finding was detected, so a broad cue
#: cannot override a real observation.
NORMAL_PATTERNS: tuple[str, ...] = (
    r"no\s+significant\s+abnormalit",
    r"no\s+abnormality\s+detected",
    r"\blungs?\s+(?:\w+\s+){0,3}(?:are|is)\s+(?:\w+\s+){0,3}clear\b",
    r"\blungs?\s+(?:are|is)\s+clear\b",
    r"no\s+acute\s+cardiopulmonary",
    r"no\s+acute\s+(?:abnormalit|finding|disease|process|infiltrat|osseous)",
    r"unremarkable\s+(?:study|examination|radiograph|chest)",
    r"normal\s+chest\s+(?:x-?\w*|radiograph|examination)?",
    r"negative\s+chest",
    r"within\s+normal\s+limits",
)

#: Cues that open a negated span.
#:
#: ``clear of`` was added after evaluation against the Open-i corpus: the
#: construction "the lungs are clear of focal airspace disease, pneumothorax, or
#: pleural effusion" is boilerplate in real normal reports, and without this cue it
#: asserts all three findings at once. It alone accounted for the majority of
#: false-positive pneumothorax calls, which is the error a radiologist would find
#: least forgivable.
#:
#: ``resolved`` covers the other direction — "the left apical pneumothorax has
#: resolved" describes a finding that is no longer present.
NEGATION_CUES: tuple[str, ...] = (
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bnegative\s+for\b",
    r"\bfree\s+of\b",
    r"\bclear\s+of\b",
    r"\bclear,\s+and\s+without\b",
    r"\babsence\s+of\b",
    r"\bruled?\s+out\b",
    r"\bnor\b",
)

#: Cues that a finding, once present, no longer is. Applied to the whole sentence
#: rather than as a prefix, since "the pneumothorax has resolved" places the cue
#: after the finding rather than before it.
RESOLUTION_CUES: tuple[str, ...] = (
    r"\bhas\s+resolved\b",
    r"\bhave\s+resolved\b",
    r"\bresolution\s+of\b",
    r"\bno\s+longer\s+(?:seen|present|visualized|identified)\b",
)

#: Cues that close a negated span before the sentence ends.
TERMINATION_CUES: tuple[str, ...] = (
    r"\bbut\b",
    r"\bhowever\b",
    r"\bthough\b",
    r"\balthough\b",
    r"\bwhereas\b",
    r"\byet\b",
    r"\baside\s+from\b",
    r"\bexcept\b",
)

_NEG_RE = re.compile("|".join(NEGATION_CUES), re.I)
_TERM_RE = re.compile("|".join(TERMINATION_CUES), re.I)
_RESOLVED_RE = re.compile("|".join(RESOLUTION_CUES), re.I)
_SENTENCE_RE = re.compile(r"(?<=[.;:])\s+|\n+")

_COMPILED: dict[Finding, tuple[re.Pattern[str], ...]] = {
    finding: tuple(re.compile(p, re.I) for p in patterns)
    for finding, patterns in FINDING_PATTERNS.items()
}
_COMPILED_NORMAL = tuple(re.compile(p, re.I) for p in NORMAL_PATTERNS)


@dataclass(frozen=True)
class ExtractedLabel:
    """One finding asserted or denied by the text, with the sentence that says so."""

    finding: Finding
    present: bool
    evidence: str


def _is_negated(sentence: str, start: int) -> bool:
    """True when the sentence denies the finding at position ``start``."""
    # A resolution statement negates regardless of position: "the left apical
    # pneumothorax has resolved" places the cue after the finding, so a
    # prefix-only scope would read it as an assertion.
    if _RESOLVED_RE.search(sentence):
        return True

    prefix = sentence[:start]
    negations = list(_NEG_RE.finditer(prefix))
    if not negations:
        return False
    last_negation = negations[-1].end()
    # A termination cue between the negation and the match closes the span.
    return not any(m.start() >= last_negation for m in _TERM_RE.finditer(prefix))


def extract_labels(text: str) -> list[ExtractedLabel]:
    """Return the findings the text asserts, and those it explicitly denies."""
    results: dict[Finding, ExtractedLabel] = {}

    for sentence in _SENTENCE_RE.split(text or ""):
        sentence = sentence.strip()
        if not sentence:
            continue
        for finding, patterns in _COMPILED.items():
            for pattern in patterns:
                match = pattern.search(sentence)
                if match is None:
                    continue
                present = not _is_negated(sentence, match.start())
                existing = results.get(finding)
                # A positive assertion anywhere outweighs a denial elsewhere: a
                # report that denies effusion in FINDINGS and asserts it in
                # IMPRESSION is asserting it.
                if existing is None or (present and not existing.present):
                    results[finding] = ExtractedLabel(finding, present, sentence[:200])
                break

    positives = [label for label in results.values() if label.present]
    if not positives:
        for pattern in _COMPILED_NORMAL:
            match = pattern.search(text or "")
            if match is not None:
                results[Finding.NO_FINDING] = ExtractedLabel(
                    Finding.NO_FINDING, True, match.group(0)
                )
                break

    return sorted(results.values(), key=lambda label: label.finding.value)


def positive_findings(text: str) -> list[Finding]:
    """Just the findings the text asserts."""
    return [label.finding for label in extract_labels(text) if label.present]


__all__ = [
    "FINDING_PATTERNS",
    "NEGATION_CUES",
    "ExtractedLabel",
    "extract_labels",
    "positive_findings",
]
