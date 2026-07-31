"""Stratified allocation must keep both properties the plain split has.

Stratifying is easy to get wrong in a way that silently breaks the leakage
guarantee — assign within strata but let a patient belong to two of them, and the
same chest lands in train and in test with nothing to show for it.
"""

from __future__ import annotations

import pytest

from cxr_harmony.release import (
    SplitRatios,
    assign_stratified,
    realised_proportions,
    strata_from_dataset,
)
from cxr_harmony.schema.models import CanonicalDataset, Patient, Study
from cxr_harmony.schema.vocab import Sex, Split


def _strata(n_per: int, n_strata: int = 4) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    counter = 0
    for s in range(n_strata):
        members = []
        for _ in range(n_per):
            counter += 1
            members.append(f"{counter:016x}")
        out[f"STRATUM_{s}"] = members
    return out


def test_every_patient_gets_exactly_one_split():
    strata = _strata(50)
    assignments = assign_stratified(strata)
    everyone = [p for members in strata.values() for p in members]
    assert set(assignments) == set(everyone)
    assert all(isinstance(v, Split) for v in assignments.values())


def test_no_patient_can_straddle_two_splits():
    """The guarantee the whole release stage exists to provide."""
    assignments = assign_stratified(_strata(100))
    assert len(assignments) == len(set(assignments))


def test_assignment_is_stable_when_the_cohort_grows():
    """Existing patients must not move, or every model ever trained has seen
    part of the new test set."""
    small = _strata(40)
    before = assign_stratified(small)

    grown = {k: list(v) for k, v in small.items()}
    extra = 10_000
    for key in grown:
        for _ in range(200):
            extra += 1
            grown[key].append(f"{extra:016x}")
    after = assign_stratified(grown)

    for pid, split in before.items():
        assert after[pid] == split, pid


def test_each_stratum_is_split_in_proportion():
    """The point of stratifying: no stratum is disproportionately in one split."""
    assignments = assign_stratified(_strata(400, n_strata=5))
    strata = _strata(400, n_strata=5)
    for members in strata.values():
        counts = {s: 0 for s in ("train", "val", "test")}
        for pid in members:
            counts[assignments[pid].value] += 1
        assert abs(counts["train"] / len(members) - 0.70) < 0.08
        assert abs(counts["test"] / len(members) - 0.15) < 0.06


def test_a_small_stratum_is_not_systematically_all_one_split():
    """Without per-stratum salting, a stratum whose members happen to hash low
    lands entirely in train."""
    all_train = 0
    for trial in range(40):
        stratum = {f"S{trial}": [f"{trial * 1000 + i:016x}" for i in range(8)]}
        assignments = assign_stratified(stratum)
        if len({v.value for v in assignments.values()}) == 1:
            all_train += 1
    assert all_train < 20, "strata are collapsing to a single split too often"


def test_overall_proportions_still_approach_the_target():
    assignments = assign_stratified(_strata(2000, n_strata=6))
    proportions = realised_proportions(assignments)
    assert abs(proportions["train"] - 0.70) < 0.03
    assert abs(proportions["val"] - 0.15) < 0.03


def test_custom_ratios_are_honoured():
    assignments = assign_stratified(
        _strata(2000, n_strata=4), ratios=SplitRatios(train=0.5, val=0.25, test=0.25)
    )
    assert abs(realised_proportions(assignments)["train"] - 0.50) < 0.03


def test_changing_the_salt_reassigns():
    strata = _strata(300)
    a = assign_stratified(strata, salt="one")
    b = assign_stratified(strata, salt="two")
    moved = sum(1 for pid in a if a[pid] != b[pid])
    assert moved > 100


# --- Deriving strata from a cohort ------------------------------------------


def _dataset(spec):
    patients, studies = [], []
    for i, (site, sex, age) in enumerate(spec, start=1):
        pid = f"{i:016x}"
        patients.append(Patient(pseudo_id=pid, sex=Sex(sex), age_years=age))
        studies.append(
            Study(
                study_uid=f"2.25.{i}",
                pseudo_patient_id=pid,
                site_id=site,
                modality="DX",
            )
        )
    return CanonicalDataset(patients=patients, studies=studies)


def test_strata_are_site_by_sex_by_age_band():
    strata = strata_from_dataset(_dataset([("SITE_A", "M", 45), ("SITE_B", "F", 70)]))
    assert set(strata) == {"SITE_A | M | 40-59", "SITE_B | F | 60-79"}


def test_a_cross_site_patient_lands_in_exactly_one_stratum():
    """Otherwise the leakage guarantee fails at the seam between strata."""
    pid = f"{1:016x}"
    dataset = CanonicalDataset(
        patients=[Patient(pseudo_id=pid, sex=Sex.MALE, age_years=45)],
        studies=[
            Study(study_uid="2.25.1", pseudo_patient_id=pid, site_id="SITE_B", modality="DX"),
            Study(study_uid="2.25.2", pseudo_patient_id=pid, site_id="SITE_A", modality="DX"),
        ],
    )
    strata = strata_from_dataset(dataset)
    appearances = [k for k, members in strata.items() if pid in members]
    assert len(appearances) == 1

    assignments = assign_stratified(strata)
    assert len(assignments) == 1


def test_every_patient_in_the_cohort_reaches_a_stratum():
    dataset = _dataset([("SITE_A", "M", 45), ("SITE_A", "F", 8), ("SITE_C", "M", 89)])
    strata = strata_from_dataset(dataset)
    assert sum(len(v) for v in strata.values()) == len(dataset.patients)


def test_a_patient_with_no_age_still_gets_a_stratum():
    dataset = CanonicalDataset(
        patients=[Patient(pseudo_id=f"{1:016x}", sex=Sex.UNKNOWN)],
        studies=[
            Study(
                study_uid="2.25.1",
                pseudo_patient_id=f"{1:016x}",
                site_id="SITE_A",
                modality="DX",
            )
        ],
    )
    strata = strata_from_dataset(dataset)
    assert "SITE_A | U | unknown" in strata


@pytest.mark.parametrize("n_strata", [1, 3, 12])
def test_works_across_stratum_counts(n_strata):
    assignments = assign_stratified(_strata(60, n_strata=n_strata))
    assert len(assignments) == 60 * n_strata
