"""Cohen's kappa, checked against hand-worked tables.

The rare-class tests matter most. Kappa collapsing at low support is a property
of the statistic, and code that quietly reports a swinging number as if it were
stable is how a weak result gets published as a strong one.
"""

from __future__ import annotations

import pytest

from cxr_harmony.qc.agreement import (
    ContingencyTable,
    agreement_by_label,
    cohens_kappa,
    pooled_kappa,
    table_from_sets,
)


def test_known_contingency_table():
    """Worked by hand.

    n=100, both_pos=20, both_neg=60, only_first=10, only_second=10.
      p_o = (20+60)/100 = 0.80
      first positive  = 30/100 = 0.30
      second positive = 30/100 = 0.30
      p_e = 0.30*0.30 + 0.70*0.70 = 0.09 + 0.49 = 0.58
      kappa = (0.80 - 0.58) / (1 - 0.58) = 0.22 / 0.42 = 0.5238...
    """
    result = cohens_kappa(ContingencyTable(20, 60, 10, 10))
    assert result.observed_agreement == pytest.approx(0.80)
    assert result.expected_agreement == pytest.approx(0.58)
    assert result.kappa == pytest.approx(0.5238, abs=1e-4)


def test_perfect_agreement_is_one():
    assert cohens_kappa(ContingencyTable(30, 70, 0, 0)).kappa == pytest.approx(1.0)


def test_chance_level_agreement_is_zero():
    """Raters independent at 50% each: observed agreement equals expected."""
    result = cohens_kappa(ContingencyTable(25, 25, 25, 25))
    assert result.kappa == pytest.approx(0.0)


def test_systematic_disagreement_is_negative():
    """Worse than chance, which kappa can express and accuracy cannot."""
    assert cohens_kappa(ContingencyTable(0, 0, 50, 50)).kappa < 0


def test_unanimous_negative_returns_zero_not_nan():
    """Both raters said no to everything; chance explains it all.

    The denominator is zero here, so a naive implementation emits nan and poisons
    every downstream average.
    """
    result = cohens_kappa(ContingencyTable(0, 100, 0, 0))
    assert result.kappa == 0.0
    assert result.expected_agreement == pytest.approx(1.0)


def test_empty_table_does_not_divide_by_zero():
    assert cohens_kappa(ContingencyTable(0, 0, 0, 0)).kappa == 0.0


# --- Support and reliability ------------------------------------------------


def test_support_counts_reference_positives_not_predictions():
    """Support is a property of the ground truth, so a wild extractor cannot inflate it."""
    result = cohens_kappa(
        ContingencyTable(both_positive=5, both_negative=90, only_first=400, only_second=5)
    )
    assert result.support == 10


def test_low_support_is_flagged_unreliable():
    """A finding with 3 positives in 1,965 studies, as tuberculosis actually is here."""
    result = cohens_kappa(
        ContingencyTable(both_positive=3, both_negative=1961, only_first=1, only_second=0)
    )
    assert result.support == 3
    assert not result.is_reliable
    assert result.interpretation() == "insufficient support"


def test_adequate_support_is_not_flagged():
    result = cohens_kappa(
        ContingencyTable(both_positive=200, both_negative=1700, only_first=30, only_second=35)
    )
    assert result.is_reliable
    assert result.interpretation() in {"substantial", "almost perfect", "moderate"}


def test_kappa_swings_on_a_single_disagreement_at_low_support():
    """Demonstrates why the support must be reported alongside the value."""
    tight = cohens_kappa(ContingencyTable(3, 1961, 0, 1))
    looser = cohens_kappa(ContingencyTable(3, 1958, 3, 1))
    assert abs(tight.kappa - looser.kappa) > 0.20


@pytest.mark.parametrize(
    ("kappa_table", "expected"),
    [
        (ContingencyTable(200, 1700, 5, 5), "almost perfect"),
        (ContingencyTable(150, 1600, 60, 60), "substantial"),
        # Same proportions as the hand-worked table above (kappa 0.524), scaled up
        # so support clears the reliability floor.
        (ContingencyTable(200, 600, 100, 100), "moderate"),
        (ContingencyTable(100, 1400, 200, 200), "slight"),
    ],
)
def test_interpretation_bands(kappa_table, expected):
    assert cohens_kappa(kappa_table).interpretation() == expected


# --- Building tables from label sets ----------------------------------------


def test_table_from_paired_label_sets():
    predicted = [{"A"}, {"A", "B"}, set(), {"B"}]
    reference = [{"A"}, {"A"}, {"A"}, set()]
    table = table_from_sets(predicted, reference, "A")
    assert table.both_positive == 2
    assert table.only_second == 1  # reference said A, extractor missed it
    assert table.only_first == 0
    assert table.both_negative == 1


def test_mismatched_lengths_are_refused():
    """Silently zipping to the shorter list would score a subset and report it as the whole."""
    with pytest.raises(ValueError):
        table_from_sets([{"A"}], [{"A"}, {"A"}], "A")


def test_agreement_by_label_covers_every_requested_label():
    predicted = [{"A"}, {"B"}]
    reference = [{"A"}, {"A"}]
    result = agreement_by_label(predicted, reference, ["A", "B", "C"])
    assert set(result) == {"A", "B", "C"}
    assert result["C"].support == 0


# --- Pooling ----------------------------------------------------------------


def test_pooled_kappa_sums_the_tables():
    agreements = agreement_by_label(
        [{"A", "B"}, {"A"}, set()],
        [{"A", "B"}, {"B"}, set()],
        ["A", "B"],
    )
    pooled = pooled_kappa(agreements)
    assert pooled.table.total == 6  # 3 studies x 2 labels
    assert pooled.label == "pooled"


def test_pooling_is_not_macro_averaging():
    """A rare label handled badly must not drag the headline as hard as a common one.

    Label COMMON: 200 positives, near-perfect agreement.
    Label RARE:   2 positives, both missed.
    Macro-averaging would halve the headline; pooling barely moves it.
    """
    common = cohens_kappa(ContingencyTable(200, 1760, 2, 3), "COMMON")
    rare = cohens_kappa(ContingencyTable(0, 1963, 0, 2), "RARE")
    agreements = {"COMMON": common, "RARE": rare}

    pooled = pooled_kappa(agreements)
    macro = (common.kappa + rare.kappa) / 2

    assert pooled.kappa > 0.90
    assert macro < 0.60
    assert pooled.kappa > macro
