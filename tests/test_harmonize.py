"""Harmonisation, checked against what was actually planted in each site's delivery.

Counting that a value resolved is not the same as checking it resolved to the
right thing. A config that maps every projection to PA produces a clean QC report
and a ruined cohort, so these tests compare against ground truth.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cxr_harmony.deid import deidentify
from cxr_harmony.harmonize import ValueMapping, harmonize, load_site_configs
from cxr_harmony.harmonize.config import SiteConfig
from cxr_harmony.ingest import ingest
from cxr_harmony.reports import process_reports
from cxr_harmony.schema.vocab import Finding, LabelSource, Sex, ViewPosition

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "sites"
KEY = b"harmonise-stage-test-key-32-byte"


@pytest.fixture(scope="module")
def harmonised(tmp_path_factory):
    from cxr_harmony.synth import generate_corpus
    from cxr_harmony.workspace import Workspace

    src = tmp_path_factory.mktemp("src")
    truth = generate_corpus(src, seed=616, n_patients=36, n_cross_site=7, image_size=128)
    ws = Workspace(tmp_path_factory.mktemp("ws") / "work").ensure()
    ingest(src, ws)
    deidentify(src, ws, key=KEY)
    process_reports(src, ws, key=KEY)
    result = harmonize(src, ws, CONFIGS)
    return src, ws, truth, result


# --- Config loading ---------------------------------------------------------


def test_every_delivered_site_has_a_config():
    configs = load_site_configs(CONFIGS)
    assert set(configs) == {"SITE_A", "SITE_B", "SITE_C"}


def test_a_config_with_an_unknown_key_is_rejected(tmp_path):
    """A typo in a config must fail loudly, not be ignored."""
    (tmp_path / "bad.yaml").write_text(
        yaml.safe_dump(
            {"site_id": "SITE_X", "labels": {"source": "report"}, "vew_position": {}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_site_configs(tmp_path)


def test_an_uncompilable_pattern_is_rejected_at_load():
    with pytest.raises(ValidationError):
        SiteConfig.model_validate(
            {
                "site_id": "SITE_X",
                "labels": {"source": "report"},
                "view_position": {"patterns": [{"match": "[unclosed", "value": "PA"}]},
            }
        )


def test_an_unknown_label_source_is_rejected():
    with pytest.raises(ValidationError):
        SiteConfig.model_validate({"site_id": "SITE_X", "labels": {"source": "telepathy"}})


def test_missing_config_for_a_delivered_site_raises(harmonised, tmp_path):
    src, ws, _, _ = harmonised
    only_a = tmp_path / "configs"
    only_a.mkdir()
    (only_a / "site_a.yaml").write_text(
        (CONFIGS / "site_a.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(KeyError):
        harmonize(src, ws, only_a)


# --- Value resolution -------------------------------------------------------


def test_exact_map_takes_precedence_over_patterns():
    mapping = ValueMapping(
        map={"AP": "AP"},
        patterns=[{"match": ".*", "value": "PA"}],
        default="UNKNOWN",
    )
    assert mapping.resolve("AP") == ("AP", True)


def test_patterns_run_in_order():
    """'CHEST AP SUPINE PORTABLE' must not match a careless PA rule."""
    config = load_site_configs(CONFIGS)["SITE_B"]
    assert config.view_position.resolve("CHEST AP SUPINE PORTABLE")[0] == "AP"
    assert config.view_position.resolve("CHEST PA ERECT")[0] == "PA"
    assert config.view_position.resolve("CHEST LAT")[0] == "LATERAL"


def test_unrecognised_value_falls_back_and_reports_itself():
    mapping = ValueMapping(map={"PA": "PA"}, default="UNKNOWN")
    value, matched = mapping.resolve("OBLIQUE-ISH")
    assert value == "UNKNOWN"
    assert matched is False


def test_empty_value_is_not_counted_as_unmapped():
    mapping = ValueMapping(map={"PA": "PA"}, default="UNKNOWN")
    assert mapping.resolve("") == ("UNKNOWN", False)


def test_legacy_arrow_notation_resolves():
    """Site C's 'P->A' is a beam direction, not a conformant CS value."""
    config = load_site_configs(CONFIGS)["SITE_C"]
    assert config.view_position.resolve("P->A")[0] == "PA"
    assert config.view_position.resolve("A->P")[0] == "AP"


def test_hl7_numeric_sex_codes_resolve():
    """'1' and '2' are valid strings, so an unconfigured reader silently loses sex."""
    config = load_site_configs(CONFIGS)["SITE_C"]
    assert config.sex.resolve("1")[0] == "M"
    assert config.sex.resolve("2")[0] == "F"


# --- Correctness against ground truth ---------------------------------------


def test_projection_matches_what_each_site_actually_sent(harmonised):
    _, ws, truth, result = harmonised
    planted = {s["study_uid"]: s["view"] for s in truth["studies"]}

    uid_map = {}
    import json

    uid_map = json.loads(ws.uid_map.read_text(encoding="utf-8"))
    remapped = {uid_map[k]: v for k, v in planted.items() if k in uid_map}

    checked = 0
    for series in result.dataset.series:
        expected = remapped.get(series.study_uid)
        if expected is None:
            continue
        assert series.view_position is ViewPosition(expected), series.study_uid
        checked += 1
    assert checked == len(result.dataset.series)


def test_no_projection_is_left_unknown(harmonised):
    """All three encodings resolved; an UNKNOWN here means a config gap."""
    _, _, _, result = harmonised
    unknown = [s for s in result.dataset.series if s.view_position is ViewPosition.UNKNOWN]
    assert unknown == []


def test_every_study_has_a_date_including_the_private_block_site(harmonised):
    _, _, _, result = harmonised
    undated = [s for s in result.dataset.studies if s.study_date is None]
    assert undated == []


def test_site_b_dates_parse_from_dd_mm_yyyy(harmonised):
    _, _, _, result = harmonised
    site_b = [s for s in result.dataset.studies if s.site_id == "SITE_B"]
    assert site_b
    assert all(s.study_date is not None for s in site_b)


def test_sex_resolves_across_all_three_encodings(harmonised):
    _, _, _, result = harmonised
    values = {p.sex for p in result.dataset.patients}
    assert values <= {Sex.MALE, Sex.FEMALE, Sex.OTHER, Sex.UNKNOWN}
    assert Sex.UNKNOWN not in values, "a site's sex encoding was not configured"


def test_labels_arrive_from_all_three_channels(harmonised):
    _, _, _, result = harmonised
    by_site: dict[str, set[str]] = {}
    studies = {s.study_uid: s.site_id for s in result.dataset.studies}
    for label in result.dataset.labels:
        by_site.setdefault(studies[label.study_uid], set()).add(label.source.value)

    assert LabelSource.SITE_STRUCTURED.value in by_site["SITE_A"]
    assert LabelSource.SITE_STRUCTURED.value in by_site["SITE_B"]
    assert LabelSource.REPORT_RULE.value in by_site["SITE_C"]


def test_report_derived_labels_are_marked_as_weaker_evidence(harmonised):
    """They must not be laundered into equivalence with a site's structured export."""
    _, _, _, result = harmonised
    studies = {s.study_uid: s.site_id for s in result.dataset.studies}
    for label in result.dataset.labels:
        if studies[label.study_uid] == "SITE_C":
            assert label.source is LabelSource.REPORT_RULE


def test_structured_labels_retain_the_site_native_string(harmonised):
    """'CM' has to stay visible, or a mis-mapping is unauditable after the fact."""
    _, _, _, result = harmonised
    structured = [
        label for label in result.dataset.labels
        if label.source is LabelSource.SITE_STRUCTURED
    ]
    assert structured
    assert all(label.site_native_value for label in structured)


def test_abbreviated_and_english_labels_reach_the_same_canonical_value(harmonised):
    """Site A writes 'Cardiomegaly', site B writes 'CM'; both must converge.

    Asserted over whichever findings the corpus happens to contain rather than a
    named one, since which findings are drawn varies with the seed.
    """
    _, _, _, result = harmonised
    natives_by_finding: dict[Finding, set[str]] = {}
    for label in result.dataset.labels:
        if label.site_native_value:
            natives_by_finding.setdefault(label.finding, set()).add(label.site_native_value)

    converged = {
        finding: natives
        for finding, natives in natives_by_finding.items()
        if len(natives) >= 2
    }
    assert converged, f"no finding received two spellings: {natives_by_finding}"


def test_the_two_label_vocabularies_are_disjoint_but_map_onto_one():
    """The mapping is doing real work: no site-native string is shared."""
    configs = load_site_configs(CONFIGS)
    a = {k.upper() for k in configs["SITE_A"].labels.map}
    b = {k.upper() for k in configs["SITE_B"].labels.map}
    assert a.isdisjoint(b)
    assert set(configs["SITE_A"].labels.map.values()) == set(configs["SITE_B"].labels.map.values())


def test_nothing_is_unmapped_for_the_configured_sites(harmonised):
    _, _, _, result = harmonised
    assert result.unmapped == [], [u.to_dict() for u in result.unmapped]


def test_an_unknown_label_becomes_other_and_is_counted(harmonised, tmp_path):
    """A site that starts sending a new code must surface, not vanish."""
    src, ws, _, _ = harmonised
    configs = tmp_path / "configs"
    configs.mkdir()
    for path in CONFIGS.glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data["site_id"] == "SITE_B":
            data["labels"]["map"].pop("CM", None)
        (configs / path.name).write_text(yaml.safe_dump(data), encoding="utf-8")

    result = harmonize(src, ws, configs)
    unmapped = [u for u in result.unmapped if u.value == "CM"]
    assert unmapped, "dropping a label mapping produced no QC signal"
    assert any(label.finding is Finding.OTHER for label in result.dataset.labels)


# --- Structure --------------------------------------------------------------


def test_cross_site_patients_appear_once_with_studies_at_both(harmonised):
    _, _, truth, result = harmonised
    studies_by_patient: dict[str, set[str]] = {}
    for study in result.dataset.studies:
        studies_by_patient.setdefault(study.pseudo_patient_id, set()).add(study.site_id)

    multi = [pid for pid, sites in studies_by_patient.items() if len(sites) > 1]
    assert len(multi) == len(truth["cross_site_patients"])


def test_referential_integrity_holds(harmonised):
    _, _, _, result = harmonised
    ds = result.dataset
    patients = {p.pseudo_id for p in ds.patients}
    studies = {s.study_uid for s in ds.studies}
    series = {s.series_uid for s in ds.series}

    assert all(s.pseudo_patient_id in patients for s in ds.studies)
    assert all(s.study_uid in studies for s in ds.series)
    assert all(i.series_uid in series for i in ds.instances)
    assert all(label.study_uid in studies for label in ds.labels)
    assert all(r.study_uid in studies for r in ds.reports)


def test_canonical_dataset_is_written_and_reloadable(harmonised):
    from cxr_harmony.schema import CanonicalDataset

    _, ws, _, result = harmonised
    reloaded = CanonicalDataset.model_validate_json(ws.canonical.read_text(encoding="utf-8"))
    assert len(reloaded.studies) == len(result.dataset.studies)
