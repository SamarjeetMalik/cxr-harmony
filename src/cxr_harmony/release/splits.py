"""Assignment of patients to train, validation and test partitions.

Two decisions here, both of which are easy to get wrong in ways that do not show
up until the model is in front of a clinician.

**Splits are assigned per patient, never per study.** A patient with a baseline
and two follow-up films contributes three studies. Split those independently and
the same chest — same habitus, same old rib fracture, same implanted device —
appears in training and in test. The model recognises the patient rather than the
pathology, the test score is inflated, and nothing in a random-split evaluation
reveals it. This is the single most common serious error in medical imaging
datasets, and the cross-site linkage in :mod:`cxr_harmony.deid` exists partly to
make the guarantee hold across hospitals as well as within one.

**Assignment is by hash threshold, not by shuffling.** A shuffle reassigns
everyone whenever the cohort grows. For a collection that will receive deliveries
for years, that is fatal: patients who were in training last quarter land in test
this quarter, so every model ever trained has seen part of the new test set. A
hash threshold has the property that an existing patient's assignment never
changes when new patients arrive, because it depends on that patient alone.

The price is that proportions only approach the target asymptotically. At a few
dozen patients the realised split can be several points off, so the achieved
proportions are recorded in the datasheet rather than assumed.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

from ..schema.vocab import Split

#: Resolution of the hash threshold. Finer than any realistic cohort needs.
_BUCKETS = 1_000_000


@dataclass(frozen=True)
class SplitRatios:
    """Target proportions. Must sum to one."""

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        if min(self.train, self.val, self.test) < 0:
            raise ValueError("split ratios must be non-negative")


def _bucket(pseudo_id: str, salt: str) -> int:
    """Map a patient to a bucket in ``[0, _BUCKETS)``.

    Salted with the release's split seed so that a deliberate re-split is
    possible, while an unchanged seed reproduces the assignment exactly.
    """
    digest = hashlib.sha256(f"{salt}\x1f{pseudo_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _BUCKETS


def assign_split(pseudo_id: str, *, salt: str, ratios: SplitRatios) -> Split:
    """Return the partition for one patient. Depends on that patient alone."""
    bucket = _bucket(pseudo_id, salt)
    train_cut = int(ratios.train * _BUCKETS)
    val_cut = train_cut + int(ratios.val * _BUCKETS)
    if bucket < train_cut:
        return Split.TRAIN
    if bucket < val_cut:
        return Split.VAL
    return Split.TEST


def assign_all(
    pseudo_ids: list[str],
    *,
    salt: str = "cxr-harmony-v1",
    ratios: SplitRatios | None = None,
) -> dict[str, Split]:
    """Assign every patient. Order-independent by construction."""
    ratios = ratios or SplitRatios()
    return {
        pseudo_id: assign_split(pseudo_id, salt=salt, ratios=ratios)
        for pseudo_id in sorted(set(pseudo_ids))
    }


def assign_stratified(
    strata: dict[str, list[str]],
    *,
    salt: str = "cxr-harmony-v1",
    ratios: SplitRatios | None = None,
) -> dict[str, Split]:
    """Assign per patient, with proportional allocation inside each stratum.

    ``strata`` maps a stratum key to the patients in it. Every patient must appear
    in exactly one stratum, which is why the caller supplies the grouping rather
    than this module inferring it — a patient imaged at two sites belongs to one
    stratum, and only the caller knows which.

    The threshold is applied *within* each stratum rather than globally. Both
    properties that matter survive: assignment still depends on the patient alone,
    so it is stable when the cohort grows, and it is still per patient, so no
    patient can straddle two splits.

    **Proportional, not Neyman, allocation.** Neyman allocation sizes strata by
    ``N_h * S_h``, to minimise the variance of an estimate of a population
    quantity. That is a survey-sampling objective and it is the wrong one here: it
    would deliberately put *more* test data in high-variance strata, which does not
    make a test set more representative, it makes it differently biased. A split
    wants each stratum represented in proportion to its size, which is what this
    does.
    """
    ratios = ratios or SplitRatios()
    assignments: dict[str, Split] = {}
    for stratum, patients in sorted(strata.items()):
        # Salting per stratum decorrelates the thresholds, so a small stratum is
        # not systematically all-train just because its members happen to hash low.
        stratum_salt = f"{salt}\x1f{stratum}"
        for pseudo_id in sorted(set(patients)):
            assignments[pseudo_id] = assign_split(
                pseudo_id, salt=stratum_salt, ratios=ratios
            )
    return assignments


def strata_from_dataset(dataset) -> dict[str, list[str]]:
    """Group patients by site x sex x age band, one stratum each.

    A patient imaged at more than one site is assigned to the stratum of the site
    they appear at first in sorted order — arbitrary, but it must be *some* single
    stratum or the leakage guarantee fails at the seam.
    """
    from ..qc.equity import age_band

    patients = {p.pseudo_id: p for p in dataset.patients}
    site_of: dict[str, str] = {}
    for study in sorted(dataset.studies, key=lambda s: (s.pseudo_patient_id, s.site_id)):
        site_of.setdefault(study.pseudo_patient_id, study.site_id)

    strata: dict[str, list[str]] = {}
    for pseudo_id, patient in patients.items():
        key = (
            f"{site_of.get(pseudo_id, 'UNKNOWN')} | {patient.sex.value} | "
            f"{age_band(patient.age_years)}"
        )
        strata.setdefault(key, []).append(pseudo_id)
    return strata


def realised_proportions(assignments: dict[str, Split]) -> dict[str, float]:
    """What the split actually came out as, which is not quite the target."""
    counts = Counter(split.value for split in assignments.values())
    total = sum(counts.values())
    return {
        split: round(counts.get(split, 0) / total, 4) if total else 0.0
        for split in ("train", "val", "test")
    }


__all__ = [
    "SplitRatios",
    "assign_all",
    "assign_split",
    "assign_stratified",
    "realised_proportions",
    "strata_from_dataset",
]
