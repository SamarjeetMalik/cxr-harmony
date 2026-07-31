"""Quality-control checks over the harmonised cohort.

The checks are grouped by what a failure would mean, because that determines who
has to do something about it:

* **Integrity** — the catalogue contradicts itself. Always a bug here, never the
  site's fault.
* **Completeness** — a field the cohort needs is missing for some fraction of
  studies. Usually a config gap or a genuinely absent value at source.
* **Distribution** — the sites disagree with each other in a way that will show
  up as a confound. Not an error at all, but the thing most worth looking at.

That last category is the one this project exists for. A model trained on a cohort
where one hospital contributes almost all the portable AP films and another
almost all the erect PA films can learn to recognise the hospital instead of the
disease, and it will score well on a random split while doing it. The severity
here is a warning rather than a failure precisely because the correct response is
a judgement about study design, not a code change.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum

from ..schema.models import CanonicalDataset
from ..schema.vocab import Laterality, Sex, ViewPosition
from .equity import MIN_STRATUM_SIZE, audit


class Severity(str, Enum):
    FAIL = "fail"
    WARN = "warn"
    INFO = "info"


@dataclass
class Check:
    """One check, its verdict, and enough detail to act on it."""

    name: str
    severity: Severity
    passed: bool
    message: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "passed": self.passed,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class QCReport:
    checks: list[Check] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity is Severity.FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity is Severity.WARN]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_failures": len(self.failures),
            "n_warnings": len(self.warnings),
            "stats": self.stats,
            "checks": [c.to_dict() for c in self.checks],
        }


def _pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def run_checks(
    dataset: CanonicalDataset,
    *,
    unmapped: list[dict] | None = None,
    quarantined: list[dict] | None = None,
    view_imbalance_threshold: float = 0.35,
    parity_gap_threshold: float = 0.25,
    body_part_unverified_threshold: float = 0.01,
) -> QCReport:
    """Run every check and return the report."""
    report = QCReport()
    unmapped = unmapped or []
    quarantined = quarantined or []

    patients = {p.pseudo_id for p in dataset.patients}
    studies = {s.study_uid: s for s in dataset.studies}
    series_by_study: dict[str, list] = defaultdict(list)
    for s in dataset.series:
        series_by_study[s.study_uid].append(s)

    n_studies = len(dataset.studies)

    # --- Integrity ---------------------------------------------------------
    orphan_studies = [s.study_uid for s in dataset.studies if s.pseudo_patient_id not in patients]
    report.checks.append(
        Check(
            "study_patient_integrity",
            Severity.FAIL,
            not orphan_studies,
            f"{len(orphan_studies)} studies reference an unknown patient",
            {"examples": orphan_studies[:5]},
        )
    )

    orphan_series = [s.series_uid for s in dataset.series if s.study_uid not in studies]
    report.checks.append(
        Check(
            "series_study_integrity",
            Severity.FAIL,
            not orphan_series,
            f"{len(orphan_series)} series reference an unknown study",
            {"examples": orphan_series[:5]},
        )
    )

    orphan_labels = [
        label.finding.value for label in dataset.labels if label.study_uid not in studies
    ]
    report.checks.append(
        Check(
            "label_study_integrity",
            Severity.FAIL,
            not orphan_labels,
            f"{len(orphan_labels)} labels reference an unknown study",
            {},
        )
    )

    studies_without_series = [uid for uid in studies if not series_by_study.get(uid)]
    report.checks.append(
        Check(
            "studies_have_images",
            Severity.FAIL,
            not studies_without_series,
            f"{len(studies_without_series)} studies have no series",
            {"examples": studies_without_series[:5]},
        )
    )

    photometric = Counter(i.photometric_interpretation for i in dataset.instances)
    report.checks.append(
        Check(
            "photometric_interpretation_uniform",
            Severity.FAIL,
            len(photometric) <= 1,
            "the cohort mixes greyscale conventions: "
            + ", ".join(f"{k}={v}" for k, v in sorted(photometric.items()))
            + ". MONOCHROME1 renders inverted, so a model would see part of the "
            "cohort as a photographic negative",
            {"counts": dict(sorted(photometric.items()))},
        )
    )

    digests = [i.sha256 for i in dataset.instances]
    duplicate_digests = [d for d, n in Counter(digests).items() if n > 1]
    report.checks.append(
        Check(
            "no_duplicate_pixel_content",
            Severity.WARN,
            not duplicate_digests,
            f"{len(duplicate_digests)} pixel digests appear on more than one instance",
            {"examples": duplicate_digests[:5]},
        )
    )

    # --- Completeness ------------------------------------------------------
    undated = [s.study_uid for s in dataset.studies if s.study_date is None]
    report.checks.append(
        Check(
            "study_date_present",
            Severity.WARN,
            not undated,
            f"{len(undated)} of {n_studies} studies have no parseable date "
            f"({_pct(len(undated), n_studies)}%)",
            {"examples": undated[:5]},
        )
    )

    unknown_view = [s.series_uid for s in dataset.series if s.view_position is ViewPosition.UNKNOWN]
    report.checks.append(
        Check(
            "projection_resolved",
            Severity.WARN,
            not unknown_view,
            f"{len(unknown_view)} of {len(dataset.series)} series have an unresolved projection",
            {"examples": unknown_view[:5]},
        )
    )

    # Ingest tolerates an empty BodyPartExamined, because a populated mismatch is
    # disqualifying but an absent value is merely uninformative. On a real archive
    # that tolerance has teeth: a 400-object hospital export was surveyed with the
    # tag empty on every single object, so the filter admitted abdominal and pelvic
    # studies into what was meant to be a chest cohort. Nothing upstream can catch
    # that, so it is surfaced here as an explicit unverified-scope warning.
    unattested = [s.study_uid for s in dataset.studies if not s.body_part]
    unattested_fraction = len(unattested) / n_studies if n_studies else 0.0
    # Release-blocking above the threshold. A warning was the right severity while
    # nothing could be done about it; now that a classifier can be plugged into
    # ingest, a cohort that is overwhelmingly unverified is a decision someone has
    # to take deliberately rather than scroll past.
    report.checks.append(
        Check(
            "body_part_scope_verified",
            Severity.FAIL,
            unattested_fraction <= body_part_unverified_threshold,
            f"{_pct(len(unattested), n_studies)}% of studies have unverified anatomical "
            f"scope (threshold {body_part_unverified_threshold:.0%}); supply a body-part "
            "classifier to ingest, or accept that non-chest studies may be present",
            {
                "n_unattested": len(unattested),
                "fraction": round(unattested_fraction, 4),
                "threshold": body_part_unverified_threshold,
            },
        )
    )
    report.checks.append(
        Check(
            "body_part_attested",
            Severity.WARN,
            not unattested,
            f"{len(unattested)} of {n_studies} studies carry no BodyPartExamined "
            f"({_pct(len(unattested), n_studies)}%), so anatomical scope is unverified; "
            "a chest cohort built from these is taking the sender's word for it",
            {"examples": unattested[:5]},
        )
    )

    unknown_sex = [p.pseudo_id for p in dataset.patients if p.sex is Sex.UNKNOWN]
    report.checks.append(
        Check(
            "sex_resolved",
            Severity.WARN,
            not unknown_sex,
            f"{len(unknown_sex)} of {len(dataset.patients)} patients have unknown sex",
            {"examples": unknown_sex[:5]},
        )
    )

    missing_age = [p.pseudo_id for p in dataset.patients if p.age_years is None]
    report.checks.append(
        Check(
            "age_present",
            Severity.WARN,
            not missing_age,
            f"{len(missing_age)} of {len(dataset.patients)} patients have no age",
            {"examples": missing_age[:5]},
        )
    )

    labelled = {label.study_uid for label in dataset.labels}
    unlabelled = [uid for uid in studies if uid not in labelled]
    report.checks.append(
        Check(
            "studies_labelled",
            Severity.WARN,
            not unlabelled,
            f"{len(unlabelled)} of {n_studies} studies carry no label",
            {"examples": unlabelled[:5]},
        )
    )

    report.checks.append(
        Check(
            "all_values_mapped",
            Severity.WARN,
            not unmapped,
            f"{sum(u.get('count', 0) for u in unmapped)} site values had no mapping",
            {"values": unmapped[:20]},
        )
    )

    # --- Distribution ------------------------------------------------------
    per_site = Counter(s.site_id for s in dataset.studies)
    view_by_site: dict[str, Counter] = defaultdict(Counter)
    for series in dataset.series:
        study = studies.get(series.study_uid)
        if study is not None:
            view_by_site[study.site_id][series.view_position.value] += 1

    ap_fraction = {
        site: round(counts.get("AP", 0) / sum(counts.values()), 3)
        for site, counts in view_by_site.items()
        if sum(counts.values())
    }
    spread = (max(ap_fraction.values()) - min(ap_fraction.values())) if ap_fraction else 0.0
    report.checks.append(
        Check(
            "projection_balance_across_sites",
            Severity.WARN,
            spread <= view_imbalance_threshold,
            f"AP-film fraction varies by {spread:.2f} across sites "
            f"(threshold {view_imbalance_threshold}); a model can learn the site from this",
            {"ap_fraction_by_site": ap_fraction},
        )
    )

    label_by_site: dict[str, Counter] = defaultdict(Counter)
    for label in dataset.labels:
        study = studies.get(label.study_uid)
        if study is not None and label.present:
            label_by_site[study.site_id][label.finding.value] += 1

    prevalence = {
        site: {
            finding: _pct(count, per_site[site])
            for finding, count in sorted(counts.items())
        }
        for site, counts in sorted(label_by_site.items())
    }

    report.checks.append(
        Check(
            "label_prevalence_recorded",
            Severity.INFO,
            True,
            "per-site label prevalence recorded for review",
            {"prevalence_pct": prevalence},
        )
    )

    # --- Equity ------------------------------------------------------------
    # Properties of the cohort, not of a model. See cxr_harmony.qc.equity for why
    # that distinction is kept sharp rather than borrowing the model-level names.
    equity = audit(dataset)

    # A cohort smaller than (strata x minimum) cannot fill its strata however
    # evenly it is drawn, so under-representation there is a statement about size,
    # not about balance. Warning at that scale is crying wolf, and a QC report that
    # always warns is one nobody reads. Above that scale, balance is a real choice
    # and imbalance is a real finding.
    balance_is_achievable = n_studies >= equity.n_strata * MIN_STRATUM_SIZE
    report.checks.append(
        Check(
            "strata_adequately_represented",
            Severity.WARN if balance_is_achievable else Severity.INFO,
            not equity.underrepresented,
            (
                f"{len(equity.underrepresented)} of {equity.n_strata} strata hold fewer "
                f"than {MIN_STRATUM_SIZE} studies; rates for those strata are not "
                "estimable and a model would be least reliable exactly where it is "
                "least tested"
                if balance_is_achievable
                else f"{len(equity.underrepresented)} of {equity.n_strata} strata are "
                f"below {MIN_STRATUM_SIZE} studies, but {n_studies} studies cannot fill "
                f"{equity.n_strata} strata at any balance; this reflects cohort size "
                "rather than skew and is recorded for scale planning"
            ),
            {
                "underrepresented": equity.underrepresented[:10],
                "balance_achievable_at_this_size": balance_is_achievable,
                "studies_needed_for_full_strata": equity.n_strata * MIN_STRATUM_SIZE,
            },
        )
    )

    worst = equity.worst_gap
    report.checks.append(
        Check(
            "finding_prevalence_parity",
            Severity.WARN,
            worst is None or worst.gap <= parity_gap_threshold,
            (
                f"largest prevalence gap is {worst.gap:.1%} for {worst.finding} "
                f"({worst.highest_stratum} {worst.highest_rate:.1%} vs "
                f"{worst.lowest_stratum} {worst.lowest_rate:.1%}); a gap this wide means "
                "the finding and the stratum are confounded in training data"
                if worst is not None
                else "too few adequately sized strata to compare prevalence"
            ),
            {"gaps": [g.to_dict() for g in equity.parity_gaps[:8]]},
        )
    )

    report.checks.append(
        Check(
            "demographics_recorded",
            Severity.WARN,
            equity.missing_sex == 0 and equity.missing_age == 0,
            f"{equity.missing_sex} studies lack sex and {equity.missing_age} lack age, "
            "so those studies cannot enter any stratified analysis",
            {"missing_sex": equity.missing_sex, "missing_age": equity.missing_age},
        )
    )

    cross_site = _cross_site_patients(dataset)
    report.checks.append(
        Check(
            "cross_site_patients_linked",
            Severity.INFO,
            True,
            f"{len(cross_site)} patients were imaged at more than one site and "
            "collapsed to a single identity",
            {"examples": cross_site[:5]},
        )
    )

    redacted = sum(1 for i in dataset.instances if i.pixel_redacted)
    report.checks.append(
        Check(
            "pixel_redaction_recorded",
            Severity.INFO,
            True,
            f"{redacted} of {len(dataset.instances)} images had burned-in annotation removed "
            f"({_pct(redacted, len(dataset.instances))}%)",
            {},
        )
    )

    if quarantined:
        reasons = Counter(q.get("reason", "UNKNOWN") for q in quarantined)
        report.checks.append(
            Check(
                "ingest_attrition_accounted",
                Severity.INFO,
                True,
                f"{len(quarantined)} objects were quarantined at ingest",
                {"by_reason": dict(sorted(reasons.items()))},
            )
        )

    report.stats = {
        "n_patients": len(dataset.patients),
        "n_studies": n_studies,
        "n_series": len(dataset.series),
        "n_instances": len(dataset.instances),
        "n_reports": len(dataset.reports),
        "n_labels": len(dataset.labels),
        "studies_per_site": dict(sorted(per_site.items())),
        "view_by_site": {k: dict(sorted(v.items())) for k, v in sorted(view_by_site.items())},
        "sex_distribution": dict(sorted(Counter(p.sex.value for p in dataset.patients).items())),
        "laterality_distribution": dict(
            sorted(Counter(s.laterality.value for s in dataset.series).items())
        ),
        "n_cross_site_patients": len(cross_site),
        "equity": equity.to_dict(),
        "label_prevalence_pct": prevalence,
    }
    return report


def _cross_site_patients(dataset: CanonicalDataset) -> list[str]:
    sites: dict[str, set[str]] = defaultdict(set)
    for study in dataset.studies:
        sites[study.pseudo_patient_id].add(study.site_id)
    return sorted(pid for pid, s in sites.items() if len(s) > 1)


__all__ = ["Check", "QCReport", "Severity", "run_checks", "Laterality"]
