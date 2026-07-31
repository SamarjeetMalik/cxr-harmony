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


def realised_proportions(assignments: dict[str, Split]) -> dict[str, float]:
    """What the split actually came out as, which is not quite the target."""
    counts = Counter(split.value for split in assignments.values())
    total = sum(counts.values())
    return {
        split: round(counts.get(split, 0) / total, 4) if total else 0.0
        for split in ("train", "val", "test")
    }


__all__ = ["SplitRatios", "assign_all", "assign_split", "realised_proportions"]
