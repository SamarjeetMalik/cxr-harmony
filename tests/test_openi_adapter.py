"""Open-i MeSH parsing, and one claim about it that had to be measured.

Open-i annotates with slash-qualified MeSH terms, and the adapter scores against
the **head** term only. That looked wrong: the corpus also contains terms whose
head is anatomy and whose qualifier carries the finding — ``Lung/hypoinflation``,
``Thoracic Vertebrae/degenerative``. On the face of it, head-only parsing throws
the finding away on 4,882 mentions across 2,266 of 3,927 studies.

Measured, it throws nothing away. **No qualifier anywhere in the corpus names a
canonical finding that its head missed**, so head-only and head-plus-qualifier
parsing produce identical reference sets and identical out-of-vocabulary
verdicts. The suspicion was reasonable and the data refuted it.

That equivalence is asserted here rather than written down as a comment, for two
reasons: it is a property of *this* corpus and may not hold on another, and it is
exactly the kind of claim that gets re-litigated by the next reader who notices
the same thing. If it ever stops holding, this fails and says so.

The qualifiers are still parsed, by :func:`mesh_concepts`, because they matter
for *describing* the out-of-vocabulary tail even though they do not affect
scoring — see its docstring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cxr_harmony.adapters.openi import (
    MESH_MODIFIERS,
    MESH_TO_FINDING,
    NORMAL_TERMS,
    load_corpus,
    mesh_concepts,
)
from cxr_harmony.schema.vocab import Finding

CORPUS = Path(__file__).resolve().parents[1] / "realdata" / "ecgen-radiology"


# --- mesh_concepts, no corpus needed ----------------------------------------


def test_qualifier_naming_a_finding_is_recovered():
    """The case that motivated the function: head is anatomy, qualifier is not."""
    assert mesh_concepts("Thoracic Vertebrae/degenerative/mild") == [
        "thoracic vertebrae",
        "degenerative",
    ]


def test_grading_qualifiers_are_dropped():
    """`mild` is not a finding, and counting it as one inflates the tail."""
    for term in ("Cardiomegaly/mild", "Cardiomegaly/moderate", "Cardiomegaly/severe"):
        assert mesh_concepts(term) == ["cardiomegaly"]


def test_laterality_qualifiers_are_dropped():
    assert mesh_concepts("Pleural Effusion/left/small") == ["pleural effusion"]


def test_repeated_concepts_collapse():
    assert mesh_concepts("Lung/lung/hypoinflation") == ["lung", "hypoinflation"]


def test_head_is_always_kept_even_when_it_is_a_modifier_word():
    """Only *qualifiers* are filtered. A head term is the term, always."""
    assert mesh_concepts("Small/mild") == ["small"]


@pytest.mark.parametrize("term", ["", "   ", "///"])
def test_degenerate_terms_yield_nothing(term):
    assert mesh_concepts(term) == []


def test_modifiers_do_not_collide_with_the_canonical_vocabulary():
    """A modifier that was also a finding name would silently delete findings."""
    overlap = MESH_MODIFIERS & set(MESH_TO_FINDING)
    assert not overlap, f"these are treated as both modifier and finding: {overlap}"


def test_the_plural_of_atelectasis_maps():
    """The singular and the plural of pleural effusion both map; this did not."""
    assert MESH_TO_FINDING["atelectases"] is Finding.ATELECTASIS


# --- corpus-wide equivalence, skipped without the data ----------------------

requires_corpus = pytest.mark.skipif(
    not CORPUS.exists(),
    reason="realdata/ecgen-radiology absent (gitignored; fetch with `make realdata`)",
)


def _reference(report, *, use_qualifiers: bool) -> frozenset:
    findings, normal = set(), False
    for term in report.mesh_terms:
        parts = [p.strip().lower() for p in term.split("/")]
        for concept in (parts if use_qualifiers else parts[:1]):
            if concept in NORMAL_TERMS:
                normal = True
                continue
            mapped = MESH_TO_FINDING.get(concept)
            if mapped is not None and mapped is not Finding.NO_FINDING:
                findings.add(mapped)
    if normal and not findings:
        findings.add(Finding.NO_FINDING)
    return frozenset(findings)


@requires_corpus
def test_qualifier_parsing_does_not_change_scoring():
    """The measured claim: qualifiers add no canonical finding the head missed.

    If this fails, the adapter's head-only rule has started losing findings on
    this corpus and scoring must move to :func:`mesh_concepts`. The failure
    message names the terms responsible so the decision can be made on evidence.
    """
    reports = load_corpus(CORPUS)
    assert reports, "corpus present but empty"

    culprits = set()
    for report in reports:
        if _reference(report, use_qualifiers=False) != _reference(
            report, use_qualifiers=True
        ):
            for term in report.mesh_terms:
                parts = [p.strip().lower() for p in term.split("/")]
                if parts[0] not in MESH_TO_FINDING and any(
                    p in MESH_TO_FINDING for p in parts[1:]
                ):
                    culprits.add(term.strip())

    assert not culprits, (
        "a qualifier now names a canonical finding its head missed, so head-only "
        f"scoring is losing findings: {sorted(culprits)[:10]}. Move the reference "
        "in parse_report onto mesh_concepts()."
    )
