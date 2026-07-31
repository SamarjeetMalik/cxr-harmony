"""Equity audit of the cohort.

The project this pipeline serves calls its output *equitable*, and an adjective
that is never measured is decoration. This module measures it — but at the level
a data backbone can honestly speak to, which is narrower than it first appears
and the distinction matters:

* **What is measured here is a property of the dataset.** Whether every stratum is
  represented, whether disease prevalence differs across strata, and how badly the
  cohort departs from a target population.
* **What is *not* measured here is a property of a model.** Demographic parity and
  equalised odds are defined over a classifier's predictions. No classifier is
  trained in this repository, so those cannot be computed, and reporting a
  dataset-level number under a model-level name would be a category error that
  flatters the result.

The relationship is one-way. A balanced cohort does not guarantee a fair model —
"Hidden Stratification" (Oakden-Rayner et al., 2020) is precisely the observation
that scale and diversity of data do not imply fairness of the resulting model
unless the model is itself audited. But an *imbalanced* cohort makes an unfair
model very likely, and that is detectable here, before a GPU is ever booked.

The release gate mirrors the model-level gate the project defines — no release if
any stratum falls below the threshold — with representation substituted for AUC,
since representation is what exists at this stage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..schema.models import CanonicalDataset

#: Age bands used for stratification. Coarse on purpose: a fine grid produces
#: strata with single-digit counts, where every rate estimate is noise and the
#: bands themselves start to be quasi-identifying.
AGE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("0-17", 0, 17),
    ("18-39", 18, 39),
    ("40-59", 40, 59),
    ("60-79", 60, 79),
    ("80+", 80, 200),
)

#: Below this many studies a stratum's rates are not worth quoting.
MIN_STRATUM_SIZE = 30

#: Sampling weights are capped at this ratio. An uncapped weight lets a stratum
#: with three studies dominate a reweighted estimate, which converts a
#: representation problem into a variance problem rather than solving it.
MAX_WEIGHT = 10.0


def age_band(age: int | None) -> str:
    if age is None:
        return "unknown"
    for name, low, high in AGE_BANDS:
        if low <= age <= high:
            return name
    return "unknown"


@dataclass
class Stratum:
    """One cell of the stratification, and what the cohort holds for it."""

    key: str
    n_patients: int
    n_studies: int
    findings: dict[str, int] = field(default_factory=dict)

    def prevalence(self, finding: str) -> float:
        return self.findings.get(finding, 0) / self.n_studies if self.n_studies else 0.0

    @property
    def is_adequate(self) -> bool:
        return self.n_studies >= MIN_STRATUM_SIZE

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "n_patients": self.n_patients,
            "n_studies": self.n_studies,
            "findings": dict(sorted(self.findings.items())),
        }


@dataclass
class ParityGap:
    """The spread in a finding's prevalence across strata."""

    finding: str
    gap: float
    highest_stratum: str
    highest_rate: float
    lowest_stratum: str
    lowest_rate: float

    def to_dict(self) -> dict:
        return {
            "finding": self.finding,
            "gap": round(self.gap, 4),
            "highest": {"stratum": self.highest_stratum, "rate": round(self.highest_rate, 4)},
            "lowest": {"stratum": self.lowest_stratum, "rate": round(self.lowest_rate, 4)},
        }


@dataclass
class EquityReport:
    strata: list[Stratum] = field(default_factory=list)
    parity_gaps: list[ParityGap] = field(default_factory=list)
    underrepresented: list[str] = field(default_factory=list)
    missing_sex: int = 0
    missing_age: int = 0

    @property
    def n_strata(self) -> int:
        return len(self.strata)

    @property
    def worst_gap(self) -> ParityGap | None:
        return self.parity_gaps[0] if self.parity_gaps else None

    def to_dict(self) -> dict:
        return {
            "n_strata": self.n_strata,
            "n_underrepresented": len(self.underrepresented),
            "underrepresented": self.underrepresented,
            "missing_sex": self.missing_sex,
            "missing_age": self.missing_age,
            "min_stratum_size": MIN_STRATUM_SIZE,
            "strata": [s.to_dict() for s in self.strata],
            "parity_gaps": [g.to_dict() for g in self.parity_gaps],
        }


def audit(
    dataset: CanonicalDataset,
    *,
    min_stratum_size: int = MIN_STRATUM_SIZE,
) -> EquityReport:
    """Stratify the cohort and measure representation and prevalence spread."""
    patients = {p.pseudo_id: p for p in dataset.patients}

    findings_by_study: dict[str, set[str]] = defaultdict(set)
    for label in dataset.labels:
        if label.present:
            findings_by_study[label.study_uid].add(label.finding.value)

    cells: dict[str, dict] = defaultdict(
        lambda: {"patients": set(), "studies": 0, "findings": Counter()}
    )
    missing_sex = missing_age = 0

    for study in dataset.studies:
        patient = patients.get(study.pseudo_patient_id)
        if patient is None:
            continue
        sex = patient.sex.value
        band = age_band(patient.age_years)
        if sex == "U":
            missing_sex += 1
        if band == "unknown":
            missing_age += 1

        key = f"{study.site_id} | {sex} | {band}"
        cell = cells[key]
        cell["patients"].add(patient.pseudo_id)
        cell["studies"] += 1
        for finding in findings_by_study.get(study.study_uid, ()):
            cell["findings"][finding] += 1

    strata = sorted(
        (
            Stratum(
                key=key,
                n_patients=len(cell["patients"]),
                n_studies=cell["studies"],
                findings=dict(cell["findings"]),
            )
            for key, cell in cells.items()
        ),
        key=lambda s: (-s.n_studies, s.key),
    )

    underrepresented = [s.key for s in strata if s.n_studies < min_stratum_size]

    # Prevalence spread is computed over adequate strata only. Including a
    # stratum of four studies would report a 0% or 100% rate as though it were an
    # estimate, and the resulting "gap" would be an artefact of sample size.
    adequate = [s for s in strata if s.n_studies >= min_stratum_size]
    all_findings = sorted({f for s in strata for f in s.findings})

    gaps: list[ParityGap] = []
    if len(adequate) >= 2:
        for finding in all_findings:
            rates = [(s.key, s.prevalence(finding)) for s in adequate]
            high = max(rates, key=lambda kv: kv[1])
            low = min(rates, key=lambda kv: kv[1])
            gaps.append(
                ParityGap(
                    finding=finding,
                    gap=high[1] - low[1],
                    highest_stratum=high[0],
                    highest_rate=high[1],
                    lowest_stratum=low[0],
                    lowest_rate=low[1],
                )
            )
    gaps.sort(key=lambda g: -g.gap)

    return EquityReport(
        strata=strata,
        parity_gaps=gaps,
        underrepresented=underrepresented,
        missing_sex=missing_sex,
        missing_age=missing_age,
    )


def raw_sampling_weights(
    observed: dict[str, int],
    target: dict[str, float],
) -> dict[str, float]:
    """Uncapped weights ``w_s = P_target(s) / P_dataset(s)``.

    These are already correctly scaled and need no further normalisation:

        sum_s w_s * n_s  =  sum_s (P_target(s)/P_dataset(s)) * n_s
                         =  sum_s P_target(s) * N
                         =  N

    so the reweighted cohort has the same effective size as the real one. An
    earlier version of this function rescaled the weights to have mean 1, which
    is wrong twice over: it breaks that identity, and because the mean is itself
    dominated by the largest weight, a stratum needing a 2,500x correction came
    out at 2.0 and slid under the cap unnoticed. Extreme weights have to stay
    extreme so that capping can catch them.

    A stratum absent from the cohort gets no weight. No amount of reweighting
    conjures data that was never collected.
    """
    total = sum(observed.values())
    if total == 0:
        return {}

    target_total = sum(target.values())
    if target_total <= 0:
        raise ValueError("target distribution must sum to a positive value")

    return {
        stratum: (target.get(stratum, 0.0) / target_total) / (count / total)
        for stratum, count in observed.items()
        if count > 0
    }


def sampling_weights(
    observed: dict[str, int],
    target: dict[str, float],
    *,
    max_weight: float = MAX_WEIGHT,
) -> dict[str, float]:
    """Sampling weights toward ``target``, capped.

    An uncapped weight on a stratum holding a handful of studies makes the
    reweighted estimate a restatement of those few studies carrying enormous
    variance — the reweighted number looks like a population estimate and is
    really an echo of two patients. Capping bounds the variance and, in exchange,
    biases the estimate. That trade is only defensible if it is visible, so
    :func:`capped_strata` names every stratum affected.
    """
    return {k: min(v, max_weight) for k, v in raw_sampling_weights(observed, target).items()}


def capped_strata(
    observed: dict[str, int],
    target: dict[str, float],
    *,
    max_weight: float = MAX_WEIGHT,
) -> list[str]:
    """Strata whose weight hit the cap, and whose reweighted rates are therefore biased."""
    try:
        raw = raw_sampling_weights(observed, target)
    except ValueError:
        return []
    return sorted(k for k, v in raw.items() if v > max_weight)


__all__ = [
    "AGE_BANDS",
    "MAX_WEIGHT",
    "MIN_STRATUM_SIZE",
    "EquityReport",
    "ParityGap",
    "Stratum",
    "age_band",
    "audit",
    "capped_strata",
    "raw_sampling_weights",
    "sampling_weights",
]
