"""Inter-rater agreement between extracted labels and a reference annotation.

F1 answers "how often is the extractor right about the positives". It does not
answer "how much better than guessing is it", and for a cohort where 63% of
studies are normal those are very different questions: an extractor that never
fires at all scores F1 0.0 but looks respectable on accuracy.

Cohen's kappa corrects observed agreement for the agreement expected by chance:

    kappa = (p_o - p_e) / (1 - p_e)

The project proposal sets a label-harmonisation target of kappa > 0.80, which is
the conventional threshold for "substantial" agreement (Landis & Koch, 1977).

**Kappa is deflated by rare classes and this must not be hidden.** When a finding
appears in 3 of 1,965 studies, p_e is very close to 1, the denominator collapses,
and a single disagreement swings kappa violently. That is a property of the
statistic, not evidence about the extractor. Every value returned here therefore
carries its support and prevalence, and :func:`pooled_kappa` is computed over the
pooled contingency table rather than by averaging per-finding kappas — averaging
would let a finding with support 3 weigh as heavily as one with support 210.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ContingencyTable:
    """Binary agreement counts for one label between two raters."""

    both_positive: int
    both_negative: int
    only_first: int
    only_second: int

    @property
    def total(self) -> int:
        return self.both_positive + self.both_negative + self.only_first + self.only_second

    def to_dict(self) -> dict:
        return {
            "both_positive": self.both_positive,
            "both_negative": self.both_negative,
            "only_first": self.only_first,
            "only_second": self.only_second,
        }


@dataclass(frozen=True)
class Agreement:
    """Kappa for one label, with the context needed to read it honestly."""

    label: str
    kappa: float
    observed_agreement: float
    expected_agreement: float
    support: int
    prevalence: float
    table: ContingencyTable

    @property
    def is_reliable(self) -> bool:
        """Whether the estimate rests on enough positives to mean anything.

        Not a statement about the extractor. Below roughly 30 positives the
        confidence interval on kappa is wide enough that the point estimate should
        not be quoted without it.
        """
        return self.support >= 30

    def interpretation(self) -> str:
        """Landis & Koch (1977) bands, for readers who want the conventional label."""
        if not self.is_reliable:
            return "insufficient support"
        for threshold, name in (
            (0.81, "almost perfect"),
            (0.61, "substantial"),
            (0.41, "moderate"),
            (0.21, "fair"),
            (0.01, "slight"),
        ):
            if self.kappa >= threshold:
                return name
        return "poor"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "kappa": round(self.kappa, 4),
            "observed_agreement": round(self.observed_agreement, 4),
            "expected_agreement": round(self.expected_agreement, 4),
            "support": self.support,
            "prevalence": round(self.prevalence, 4),
            "reliable": self.is_reliable,
            "interpretation": self.interpretation(),
            "table": self.table.to_dict(),
        }


def cohens_kappa(table: ContingencyTable, label: str = "") -> Agreement:
    """Compute Cohen's kappa from a 2x2 contingency table."""
    n = table.total
    if n == 0:
        return Agreement(label, 0.0, 0.0, 0.0, 0, 0.0, table)

    observed = (table.both_positive + table.both_negative) / n

    # Marginals: the probability each rater says positive, and says negative.
    first_positive = (table.both_positive + table.only_first) / n
    second_positive = (table.both_positive + table.only_second) / n
    expected = first_positive * second_positive + (1 - first_positive) * (1 - second_positive)

    if expected >= 1.0:
        # Both raters were unanimous on every study, so chance alone explains the
        # agreement and kappa is undefined. Reporting 0.0 is the conservative read.
        kappa = 0.0
    else:
        kappa = (observed - expected) / (1 - expected)

    support = table.both_positive + table.only_second  # positives per the reference
    return Agreement(
        label=label,
        kappa=kappa,
        observed_agreement=observed,
        expected_agreement=expected,
        support=support,
        prevalence=support / n,
        table=table,
    )


def table_from_sets(
    predicted: Iterable[set[str]],
    reference: Iterable[set[str]],
    label: str,
) -> ContingencyTable:
    """Build a contingency table for one label from paired sets of labels per study."""
    both_pos = both_neg = only_pred = only_ref = 0
    for pred, ref in zip(predicted, reference, strict=True):
        in_pred, in_ref = label in pred, label in ref
        if in_pred and in_ref:
            both_pos += 1
        elif in_pred:
            only_pred += 1
        elif in_ref:
            only_ref += 1
        else:
            both_neg += 1
    return ContingencyTable(both_pos, both_neg, only_pred, only_ref)


def agreement_by_label(
    predicted: list[set[str]],
    reference: list[set[str]],
    labels: Iterable[str],
) -> dict[str, Agreement]:
    """Per-label kappa across a cohort."""
    return {
        label: cohens_kappa(table_from_sets(predicted, reference, label), label)
        for label in labels
    }


def pooled_kappa(agreements: dict[str, Agreement]) -> Agreement:
    """Kappa over the pooled contingency table across all labels.

    Pooled rather than macro-averaged: averaging per-label kappas would give a
    finding with three positives the same weight as one with two hundred, which
    for a statistic already unstable at low support produces a headline number
    driven by its least trustworthy component.
    """
    pooled = ContingencyTable(
        both_positive=sum(a.table.both_positive for a in agreements.values()),
        both_negative=sum(a.table.both_negative for a in agreements.values()),
        only_first=sum(a.table.only_first for a in agreements.values()),
        only_second=sum(a.table.only_second for a in agreements.values()),
    )
    return cohens_kappa(pooled, label="pooled")


__all__ = [
    "Agreement",
    "ContingencyTable",
    "agreement_by_label",
    "cohens_kappa",
    "pooled_kappa",
    "table_from_sets",
]
