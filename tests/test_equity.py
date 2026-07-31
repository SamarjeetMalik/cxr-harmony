"""Cohort-level equity audit.

Every check here is tested against a cohort deliberately built to trip it. An
equity audit that has only ever been run on balanced data is decoration, and
decoration is worse than nothing because it is mistaken for evidence.
"""

from __future__ import annotations

import pytest

from cxr_harmony.qc.equity import (
    MAX_WEIGHT,
    MIN_STRATUM_SIZE,
    age_band,
    audit,
    capped_strata,
    raw_sampling_weights,
    sampling_weights,
)
from cxr_harmony.schema.models import CanonicalDataset, Label, Patient, Study
from cxr_harmony.schema.vocab import Finding, LabelSource, Sex


def _cohort(spec: list[tuple[str, str, int, int, int]]) -> CanonicalDataset:
    """Build a cohort from ``(site, sex, age, n_studies, n_with_finding)`` tuples."""
    patients, studies, labels = [], [], []
    counter = 0
    for site, sex, age, n_studies, n_positive in spec:
        for i in range(n_studies):
            counter += 1
            pid = f"{counter:016x}"
            uid = f"2.25.{counter}"
            patients.append(Patient(pseudo_id=pid, sex=Sex(sex), age_years=age))
            studies.append(
                Study(
                    study_uid=uid,
                    pseudo_patient_id=pid,
                    site_id=site,
                    modality="DX",
                )
            )
            if i < n_positive:
                labels.append(
                    Label(
                        study_uid=uid,
                        finding=Finding.PLEURAL_EFFUSION,
                        present=True,
                        source=LabelSource.SITE_STRUCTURED,
                    )
                )
    return CanonicalDataset(patients=patients, studies=studies, labels=labels)


# --- Stratification ---------------------------------------------------------


@pytest.mark.parametrize(
    ("age", "expected"),
    [(3, "0-17"), (17, "0-17"), (18, "18-39"), (55, "40-59"), (79, "60-79"), (89, "80+")],
)
def test_age_bands(age, expected):
    assert age_band(age) == expected


def test_missing_age_is_its_own_band_not_silently_bucketed():
    """Folding unknowns into a real band would corrupt that band's rates."""
    assert age_band(None) == "unknown"


def test_strata_are_site_by_sex_by_age():
    report = audit(_cohort([("SITE_A", "M", 45, 40, 0), ("SITE_B", "F", 70, 40, 0)]))
    assert {s.key for s in report.strata} == {"SITE_A | M | 40-59", "SITE_B | F | 60-79"}


def test_stratum_counts_are_right():
    report = audit(_cohort([("SITE_A", "M", 45, 40, 12)]))
    stratum = report.strata[0]
    assert stratum.n_studies == 40
    assert stratum.findings["PLEURAL_EFFUSION"] == 12
    assert stratum.prevalence("PLEURAL_EFFUSION") == pytest.approx(0.30)


# --- Representation gate ----------------------------------------------------


def test_under_represented_strata_are_flagged():
    report = audit(_cohort([("SITE_A", "M", 45, 100, 0), ("SITE_B", "F", 70, 5, 0)]))
    assert "SITE_B | F | 60-79" in report.underrepresented
    assert "SITE_A | M | 40-59" not in report.underrepresented


def test_a_fully_represented_cohort_flags_nothing():
    report = audit(_cohort([("SITE_A", "M", 45, 50, 0), ("SITE_B", "F", 70, 50, 0)]))
    assert report.underrepresented == []


def test_the_threshold_is_configurable():
    cohort = _cohort([("SITE_A", "M", 45, 20, 0)])
    assert audit(cohort, min_stratum_size=10).underrepresented == []
    assert audit(cohort, min_stratum_size=50).underrepresented != []


# --- Prevalence parity ------------------------------------------------------


def test_a_prevalence_gap_is_detected():
    """Effusion in 80% of one stratum and 5% of another is a confound, not a finding."""
    report = audit(
        _cohort([("SITE_A", "M", 45, 100, 80), ("SITE_B", "F", 70, 100, 5)])
    )
    worst = report.worst_gap
    assert worst is not None
    assert worst.finding == "PLEURAL_EFFUSION"
    assert worst.gap == pytest.approx(0.75, abs=0.01)
    assert worst.highest_stratum == "SITE_A | M | 40-59"
    assert worst.lowest_stratum == "SITE_B | F | 60-79"


def test_a_balanced_cohort_has_a_small_gap():
    report = audit(
        _cohort([("SITE_A", "M", 45, 100, 30), ("SITE_B", "F", 70, 100, 32)])
    )
    assert report.worst_gap.gap == pytest.approx(0.02, abs=0.01)


def test_tiny_strata_are_excluded_from_the_gap():
    """A stratum of 3 studies reads 0% or 100%, which is sample size, not prevalence.

    Including it would manufacture a 100% 'gap' out of noise.
    """
    report = audit(
        _cohort(
            [
                ("SITE_A", "M", 45, 100, 30),
                ("SITE_B", "F", 70, 100, 32),
                ("SITE_C", "M", 8, 3, 3),  # 100% prevalence, n=3
            ]
        )
    )
    assert report.worst_gap.gap < 0.10
    assert "SITE_C | M | 0-17" in report.underrepresented


def test_no_gap_reported_when_fewer_than_two_adequate_strata():
    report = audit(_cohort([("SITE_A", "M", 45, 100, 30)]))
    assert report.parity_gaps == []
    assert report.worst_gap is None


def test_gaps_are_sorted_worst_first():
    report = audit(
        _cohort([("SITE_A", "M", 45, 100, 90), ("SITE_B", "F", 70, 100, 10)])
    )
    gaps = [g.gap for g in report.parity_gaps]
    assert gaps == sorted(gaps, reverse=True)


# --- Missing demographics ---------------------------------------------------


def test_missing_sex_and_age_are_counted():
    patients = [Patient(pseudo_id=f"{i:016x}", sex=Sex.UNKNOWN) for i in range(5)]
    studies = [
        Study(
            study_uid=f"2.25.{i}",
            pseudo_patient_id=p.pseudo_id,
            site_id="SITE_A",
            modality="DX",
        )
        for i, p in enumerate(patients)
    ]
    report = audit(CanonicalDataset(patients=patients, studies=studies))
    assert report.missing_sex == 5
    assert report.missing_age == 5


# --- Sampling weights -------------------------------------------------------


def test_weights_correct_toward_the_target():
    """Under-represented strata get weight above 1, over-represented below."""
    observed = {"A": 900, "B": 100}
    target = {"A": 0.5, "B": 0.5}
    weights = sampling_weights(observed, target)
    assert weights["B"] > weights["A"]


def test_weights_preserve_the_effective_cohort_size():
    """sum(w_s * n_s) must equal N, or reweighted counts are not comparable to raw."""
    observed = {"A": 700, "B": 300}
    weights = sampling_weights(observed, {"A": 0.5, "B": 0.5})
    reweighted = sum(weights[s] * n for s, n in observed.items())
    assert reweighted == pytest.approx(sum(observed.values()))


def test_an_extreme_weight_is_not_cancelled_by_normalisation():
    """Regression: normalising by the mean let the largest weight deflate itself.

    A stratum needing a 2,500x correction came out at 2.0 and slid under the cap.
    """
    raw = raw_sampling_weights({"A": 9998, "B": 2}, {"A": 0.5, "B": 0.5})
    assert raw["B"] > 100


def test_extreme_weights_are_capped():
    """Uncapped, a stratum of 2 becomes a restatement of those 2 with huge variance."""
    weights = sampling_weights({"A": 9998, "B": 2}, {"A": 0.5, "B": 0.5})
    assert weights["B"] == pytest.approx(MAX_WEIGHT)


def test_capping_is_reported_not_silent():
    """The caller must know the reweighted estimate is biased."""
    capped = capped_strata({"A": 9998, "B": 2}, {"A": 0.5, "B": 0.5})
    assert capped == ["B"]
    assert capped_strata({"A": 500, "B": 500}, {"A": 0.5, "B": 0.5}) == []


def test_a_stratum_absent_from_the_cohort_gets_no_weight():
    """Reweighting cannot conjure data that was never collected."""
    weights = sampling_weights({"A": 100}, {"A": 0.5, "B": 0.5})
    assert "B" not in weights


def test_empty_cohort_returns_no_weights():
    assert sampling_weights({}, {"A": 1.0}) == {}


def test_a_degenerate_target_is_refused():
    with pytest.raises(ValueError):
        sampling_weights({"A": 10}, {"A": 0.0})


# --- Integration with the QC report ----------------------------------------


def test_equity_checks_appear_in_the_qc_report():
    from cxr_harmony.qc import run_checks

    report = run_checks(_cohort([("SITE_A", "M", 45, 100, 90), ("SITE_B", "F", 70, 100, 5)]))
    names = {c.name for c in report.checks}
    assert {"strata_adequately_represented", "finding_prevalence_parity"} <= names


def test_a_skewed_cohort_trips_the_parity_warning():
    from cxr_harmony.qc import run_checks

    report = run_checks(_cohort([("SITE_A", "M", 45, 100, 95), ("SITE_B", "F", 70, 100, 2)]))
    assert any(c.name == "finding_prevalence_parity" for c in report.warnings)


def test_a_balanced_cohort_does_not_trip_it():
    from cxr_harmony.qc import run_checks

    report = run_checks(_cohort([("SITE_A", "M", 45, 100, 30), ("SITE_B", "F", 70, 100, 31)]))
    assert not any(c.name == "finding_prevalence_parity" for c in report.warnings)


def test_small_cohort_reports_representation_as_info_not_warning():
    """Below the size where balance is achievable, this is a scale fact, not a skew."""
    from cxr_harmony.qc import run_checks
    from cxr_harmony.qc.checks import Severity

    report = run_checks(_cohort([("SITE_A", "M", 45, 4, 0), ("SITE_B", "F", 70, 4, 0)]))
    check = next(c for c in report.checks if c.name == "strata_adequately_represented")
    assert check.severity is Severity.INFO
    assert "cohort size" in check.message


def test_large_imbalanced_cohort_reports_representation_as_warning():
    from cxr_harmony.qc import run_checks
    from cxr_harmony.qc.checks import Severity

    report = run_checks(
        _cohort([("SITE_A", "M", 45, 500, 0), ("SITE_B", "F", 70, 2, 0)])
    )
    check = next(c for c in report.checks if c.name == "strata_adequately_represented")
    assert check.severity is Severity.WARN
    assert not check.passed
    assert MIN_STRATUM_SIZE == 30
