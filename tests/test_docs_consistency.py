"""Guard the claim that ``docs/RESULTS.md`` opens with.

That page begins: *"Every number on this page is read from a JSON under
``docs/results/``… Nothing here is transcribed by hand."* It was transcribed by
hand, and it drifted. Throughput was regenerated and the docs kept quoting the
superseded figure in three places, until an outside reader found it by reading
the JSON.

A one-off correction leaves the same failure available tomorrow, so the rule is
enforced here instead of promised in prose. Two checks, doing different jobs:

``test_headline_figures_match_source``
    Each headline is pinned to the exact JSON key it comes from. This is the one
    that catches a reversion: restore the old throughput number and it goes red,
    naming the key that disagrees.

``test_every_number_resolves_to_a_source``
    Sweeps every number in the docs and requires each to appear in some results
    JSON, or in :data:`ALLOWED`. The allowlist is the interesting part — it is
    the list of numbers nobody is checking, so it is kept short and every entry
    carries its reason. A number that quietly grows this list is a number that
    quietly stopped being traceable.

Neither check reruns the pipeline. They compare prose against the JSON on disk,
which is exactly the join that broke.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "docs" / "results"
RESULTS_MD = ROOT / "docs" / "RESULTS.md"
README = ROOT / "README.md"


def _load(name: str) -> dict:
    return json.loads((RESULTS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _dig(obj, pointer: str):
    """Resolve a ``/a/b[0]/c`` pointer, so failures can name their own source."""
    for part in pointer.strip("/").split("/"):
        match = re.fullmatch(r"([^\[]+)((?:\[\d+\])*)", part)
        assert match, f"bad pointer segment {part!r}"
        obj = obj[match.group(1)]
        for index in re.findall(r"\[(\d+)\]", match.group(2)):
            obj = obj[int(index)]
    return obj


#: ``(description, json file, pointer, rendered form)``. The rendered form is the
#: literal substring the docs must contain — matching the rendering, not just the
#: value, is what makes a thousands separator or a lost decimal place fail.
HEADLINES = [
    # Throughput. Pinned to *both* median and slowest: the slowest is what the
    # target is judged on, and quoting only the median is how a capacity claim
    # that holds on a good day gets published as a capacity claim.
    (
        "throughput median, real",
        "benchmark",
        "/runs[1]/stages[2]/studies_per_hour_median",
        "88,742",
    ),
    (
        "throughput slowest, real",
        "benchmark",
        "/runs[1]/stages[2]/studies_per_hour_slowest",
        "60,531",
    ),
    (
        "throughput median, synthetic",
        "benchmark",
        "/runs[0]/stages[2]/studies_per_hour_median",
        "70,816",
    ),
    # Label extraction on real radiologist prose, and the pre-real-data baseline
    # it is quoted against. Both, because an improvement is a pair of numbers.
    ("micro F1, held-out", "openi_heldout", "/micro/f1", "0.901"),
    ("micro F1, baseline", "openi_heldout_baseline", "/micro/f1", "0.822"),
    ("pooled kappa", "openi_heldout", "/kappa/pooled/kappa", "0.897"),
    ("pooled kappa, baseline", "openi_heldout_baseline", "/kappa/pooled/kappa", "0.814"),
    ("n held-out reports", "openi_heldout", "/n_reports", "1,965"),
    # Normal-study detection, strict and vocabulary-adjusted, with the baselines.
    ("normal detection strict", "openi_heldout", "/normal_detection/f1", "0.854"),
    (
        "normal detection adjusted",
        "openi_heldout",
        "/normal_detection_vocabulary_adjusted/f1",
        "0.932",
    ),
    (
        "normal detection strict, baseline",
        "openi_heldout_baseline",
        "/normal_detection/f1",
        "0.452",
    ),
    # Real archive: the redaction count and the photometric split, which is the
    # finding synthetic data structurally could not have produced.
    ("burned-in text found", "real_dicom", "/n_redacted", "71"),
    ("MONOCHROME1 count", "real_dicom", "/photometric_before/MONOCHROME1", "372"),
    ("MONOCHROME2 count", "real_dicom", "/photometric_before/MONOCHROME2", "28"),
]

#: Numbers that legitimately are not readings from a committed results JSON.
#: Every entry states why. The list is deliberately awkward to grow: an
#: unexplained exemption is indistinguishable from an unnoticed drift, so a
#: number that arrives here without a reason should be treated as a bug report
#: against whoever added it.
ALLOWED = {
    # -- Targets from the project proposal. Set by the proposal, not measured.
    500: "ingestion target, studies/hour",
    99.2: "PHI removal recall target, %",
    0.80: "Cohen's kappa target",
    0.89: "MAE linear-probe AUC target (out of scope: no model is trained here)",
    # -- Structural facts about the corpora, fixed by construction, not measured.
    145: "synthetic identifier strings planted; ground truth for the recall claim",
    8: "cross-site patients planted in the synthetic corpus",
    400: "real UNIFESP objects sampled",
    3955: "Open-i radiologist reports in the full corpus",
    3: "contributing sites in the worked example",
    256: "UNIFESP image edge, px — explains why the real corpus benchmarks faster",
    512: "synthetic image edge, px",
    # -- Demo transcript in the README. Reproduced verbatim from `make demo`, so
    # it is regenerable, but by rerunning the demo rather than from a results
    # JSON. If it drifts, the demo output no longer matches the README.
    74: "studies in the `make demo` transcript",
    20260731: "`make demo` seed",
    # -- Error decomposition behind the normal-study fix. One-off analysis of a
    # single evaluation run; the counts were reported, not serialised. They
    # cannot drift because nothing regenerates them.
    577: "false positives decomposed in the normal-detection investigation",
    244: "distinct out-of-vocabulary MeSH terms found",
    2125: "mentions of those out-of-vocabulary terms",
    181: "granuloma mentions",
    134: "degenerative-change mentions",
    101: "calcinosis mentions",
    # -- Per-defect F1 deltas from the four real-data corrections, quoted in
    # prose. Same status: reported once, not serialised.
    0.42: "pneumothorax F1 gain from the `clear of` negation cue",
    0.46: "consolidation F1 gain from resolution-as-assertion",
    0.674: (
        "normal-study F1 after the four real-data fixes but BEFORE `OTHER` "
        "detection. A superseded intermediate, kept because the before/after "
        "table documents two separate improvements and collapsing them would "
        "overstate what the negation cues alone achieved."
    ),
    # -- Measurements from the two rejected burned-in-text filters. Recorded so
    # the same rejected approach is not proposed again; no JSON, because a
    # reverted filter has no results file.
    53: "images still carrying PHI under the rejected glyph sub-structure filter",
    0.73: "edge density of merged text lines, lower bound",
    0.16: "edge density of non-text components, lower bound",
    0.38: "edge density of non-text components, upper bound",
    0.36: "the gap between them",
    60: "images the edge-density separation was measured over",
    80: "images the rejected filter was measured over",
    # -- The throughput correction record. These two superseded figures are
    # quoted once each, in the section that explains the mistake. They must stay
    # quotable there: a page that hides its own corrections is worth less than
    # one that shows them. The headline pins above are what stop them returning
    # to the results table.
    103013: "superseded single-run throughput, quoted in the correction note",
    55569: "the second superseded single-run figure",
    46: "the apparent %% regression between them, which was noise",
    0.537: "per-stage scaling factor that showed the slowdown was machine-wide",
    0.542: "the other end of that range",
    # -- Test count. Checked by test_readme_test_count_is_current, not here.
    385: "tests claimed in the README; verified by its own check below",
    94: "coverage on `deid`, %",
    # -- Stratum labels, standards, statute references. Not measurements.
    59: "upper bound of the 40-59 age band label",
    79: "upper bound of the 60-79 age band label",
    40: "age band boundaries",
    3.15: "DICOM PS3.15",
    4: "FHIR R4",
    2023: "DPDP Act 2023",
    15: "DPDP s.15",
    16: "DPDP s.16, cross-border transfer",
    14: "digits in an ABHA number",
    # 1.0 and 1.00 are the same float, so this entry has to carry both readings.
    1.0: "edge density of merged text lines, upper bound; also release v1.0.0",
    113100: "DCM de-identification method code",
    113101: "DCM de-identification method code",
    113107: "DCM de-identification method code",
    113108: "DCM de-identification method code",
}

#: Section and list numbering, table pipes, and markdown syntax produce digits
#: that are not claims. Stripped before the sweep rather than allowlisted, since
#: allowlisting them would swallow real values that happen to collide.
_STRIP = [
    re.compile(r"^#{1,6} \d+\..*$", re.MULTILINE),  # "## 1. Against the targets"
    re.compile(r"^\s*\d+\.\s", re.MULTILINE),  # ordered list markers
    re.compile(r"```.*?```", re.DOTALL),  # code blocks: commands, not results
    re.compile(r"\]\([^)]*\)"),  # link targets, incl. figures/results_*.png
    re.compile(r"§\S+"),  # section cross-references
]

_NUMBER = re.compile(r"\d[\d,]*\.?\d*")


def _numbers(text: str) -> set[float]:
    for pattern in _STRIP:
        text = pattern.sub(" ", text)
    found = set()
    for raw in _NUMBER.findall(text):
        try:
            found.add(float(raw.replace(",", "")))
        except ValueError:
            continue
    return found


def _all_source_values() -> set[float]:
    """Every numeric value in every results JSON, in the forms docs quote them.

    A value written as ``0.9013`` is quoted as ``0.901``, as ``90.1``, and as
    ``90``. All those renderings are admitted, because rounding for prose is not
    drift. Changing the underlying value is, and that is what the headline check
    catches.
    """
    values: set[float] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
        elif isinstance(node, bool) or node is None:
            return
        elif isinstance(node, (int, float)):
            for scaled in (float(node), float(node) * 100):
                for places in range(5):
                    values.add(round(scaled, places))

    for path in sorted(RESULTS_DIR.glob("*.json")):
        walk(json.loads(path.read_text(encoding="utf-8")))
    return values


@pytest.mark.parametrize(
    "description,source,pointer,rendered",
    HEADLINES,
    ids=[h[0].replace(" ", "_").replace(",", "") for h in HEADLINES],
)
def test_headline_figures_match_source(description, source, pointer, rendered):
    """Each headline must be the JSON value, rendered as the docs render it."""
    try:
        raw = _dig(_load(source), pointer)
    except (KeyError, IndexError, FileNotFoundError) as exc:  # pragma: no cover
        pytest.fail(f"{description}: source {source}.json{pointer} missing ({exc})")

    quoted = float(rendered.replace(",", ""))
    # Tolerance comes from how the docs render it: a figure written to three
    # decimals may differ from the source by half of the last place and no more.
    places = len(rendered.partition(".")[2])
    assert quoted == pytest.approx(raw, abs=0.5 * 10**-places), (
        f"{description}: docs say {rendered}, but {source}.json{pointer} is {raw}. "
        "Regenerate the docs from the JSON rather than editing the number."
    )

    docs = RESULTS_MD.read_text(encoding="utf-8") + README.read_text(encoding="utf-8")
    assert rendered in docs, (
        f"{description}: {source}.json{pointer} is {raw}, rendered {rendered!r}, "
        "which appears in neither docs/RESULTS.md nor README.md. Either the docs "
        "still quote a superseded figure, or this headline moved and the pin here "
        "was not moved with it."
    )


#: Figures that were once published and have been superseded, with the one place
#: each is still allowed to appear. Anywhere else, they are a reversion.
#:
#: This is the check that actually catches the drift. The headline pins above
#: assert the *correct* number is present, which a partial revert survives: put
#: the old figure back in the results table while a corrected copy still stands
#: in the targets table, and a presence check sees nothing wrong. Asserting the
#: superseded figure is *absent* has no such hole.
RETIRED = {
    "103,013": "first single-run throughput, published as though it were the throughput",
    "55,569": "second single-run throughput, misread as a 46% regression from the first",
}

#: The only section of RESULTS.md permitted to name a retired figure: the one
#: that exists to record the correction. Bounded at the next same-level heading.
CORRECTION_SECTION = "### Why this is reported as a range, and a mistake that is worth recording"


def _correction_section(text: str) -> str:
    start = text.index(CORRECTION_SECTION)
    rest = text[start + len(CORRECTION_SECTION) :]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


@pytest.mark.parametrize("figure,why", sorted(RETIRED.items()), ids=sorted(RETIRED))
def test_retired_figures_are_not_quoted_as_current(figure, why):
    """A superseded figure may be discussed, never restated as a result."""
    readme = README.read_text(encoding="utf-8")
    assert figure not in readme, (
        f"README.md quotes {figure}, a retired figure ({why}). The README carries "
        "no correction record, so any appearance of this number there is a "
        "reversion. Read the current value from docs/results/benchmark.json."
    )

    results = RESULTS_MD.read_text(encoding="utf-8")
    total = results.count(figure)
    if total == 0:
        return  # The correction record was removed entirely; that is allowed.

    inside = _correction_section(results).count(figure)
    assert total == inside, (
        f"docs/RESULTS.md names {figure} ({why}) {total - inside} time(s) outside "
        f"{CORRECTION_SECTION!r}. That section is the only place a retired figure "
        "may appear. Everywhere else it is a result, and this one is superseded."
    )


@pytest.mark.parametrize("doc", [RESULTS_MD, README], ids=["RESULTS.md", "README.md"])
def test_every_number_resolves_to_a_source(doc):
    """No number in the docs without a JSON behind it or a stated exemption."""
    orphans = sorted(
        _numbers(doc.read_text(encoding="utf-8")) - _all_source_values() - set(ALLOWED)
    )
    assert not orphans, (
        f"{doc.name} quotes numbers that are in no docs/results/*.json: {orphans}. "
        "Either they came from a run whose JSON was never committed, or they were "
        "typed by hand. If a number genuinely is not a reading, add it to ALLOWED "
        "with the reason."
    )


def test_readme_test_count_is_current():
    """The README states a test count; it drifts every time a test is added."""
    match = re.search(r"\|\s*Tests\s*\|\s*\*\*(\d+)\*\*\s*\|", README.read_text(encoding="utf-8"))
    assert match, "README results table no longer has a Tests row"
    claimed = int(match.group(1))

    collected = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        collected += len(re.findall(r"^def test_", path.read_text(encoding="utf-8"), re.MULTILINE))
    assert collected > 0

    # Approximate: parametrised tests multiply at collection, so this is a floor
    # check for gross drift, not an exact count. It catches "we added a module of
    # tests and never updated the README", which is the drift that actually
    # happens.
    assert claimed >= collected, (
        f"README claims {claimed} tests but {collected} test functions are defined "
        "before parametrisation; the claim is now below the floor. Rerun pytest and "
        "update the README."
    )
