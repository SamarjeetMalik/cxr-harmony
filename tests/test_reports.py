"""Report sectioning, scrubbing and label extraction.

The negation and section-restriction tests matter most. Both failure modes are
silent: a keyword matcher that ignores negation labels every normal study as
abnormal, and one that reads the indication learns to predict the clinician's
suspicion rather than the image — which inflates validation scores instead of
depressing them, so nothing looks wrong until deployment.
"""

from __future__ import annotations

import pytest

from cxr_harmony.deid import Pseudonymiser, deidentify
from cxr_harmony.ingest import ingest
from cxr_harmony.reports import (
    clinical_text,
    extract_labels,
    parse_sections,
    positive_findings,
    process_reports,
    scrub_report,
)
from cxr_harmony.schema.vocab import Finding, ReportSection

KEY = b"report-stage-test-key-32-bytes!!"


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    from cxr_harmony.synth import generate_corpus
    from cxr_harmony.workspace import Workspace

    src = tmp_path_factory.mktemp("src")
    truth = generate_corpus(src, seed=321, n_patients=30, n_cross_site=6, image_size=128)
    ws = Workspace(tmp_path_factory.mktemp("ws") / "work").ensure()
    ingest(src, ws)
    deidentify(src, ws, key=KEY)
    result = process_reports(src, ws, key=KEY)
    return src, ws, truth, result


# --- Sectioning -------------------------------------------------------------


def test_sections_are_recognised():
    text = """
INDICATION:
Cough for five days.

FINDINGS:
The cardiac silhouette is enlarged.

IMPRESSION:
1. Cardiomegaly.
"""
    sections = parse_sections(text)
    assert set(sections) == {
        ReportSection.INDICATION,
        ReportSection.FINDINGS,
        ReportSection.IMPRESSION,
    }
    assert "enlarged" in sections[ReportSection.FINDINGS]


def test_demographic_header_before_the_first_heading_is_dropped():
    text = "Patient Name : Ravi Sharma\nMRN : SMC-001\n\nFINDINGS:\nClear lungs.\n"
    sections = parse_sections(text)
    assert ReportSection.FINDINGS in sections
    assert "Ravi" not in "".join(sections.values())


def test_trailer_lines_are_not_captured_as_clinical_text():
    text = "FINDINGS:\nClear lungs.\n\nReported by Dr. Menon\nElectronically signed on 01-01-2024\n"
    sections = parse_sections(text)
    assert "Menon" not in sections[ReportSection.FINDINGS]


def test_clinical_text_excludes_the_indication():
    """The indication states the suspicion, not the finding."""
    text = """
INDICATION:
Persistent cough, evaluate for tuberculosis.

FINDINGS:
Both lung fields are clear.
"""
    body = clinical_text(parse_sections(text))
    assert "tuberculosis" not in body.lower()
    assert "clear" in body.lower()


def test_a_suspicion_in_the_indication_does_not_become_a_label():
    """Otherwise the model learns to predict the referrer, not the radiograph."""
    text = """
INDICATION:
Persistent cough for 6 weeks, evaluate for tuberculosis.

FINDINGS:
Both lung fields are clear. The cardiac silhouette is within normal limits.

IMPRESSION:
1. No significant abnormality detected.
"""
    findings = positive_findings(clinical_text(parse_sections(text)))
    assert Finding.TUBERCULOSIS not in findings
    assert findings == [Finding.NO_FINDING]


# --- Negation ---------------------------------------------------------------


def test_denied_findings_are_not_asserted():
    text = "No focal parenchymal opacity, effusion or pneumothorax is identified."
    assert positive_findings(text) == []


def test_negation_scope_ends_at_a_termination_cue():
    text = "There is no effusion but there is dense consolidation in the left mid zone."
    findings = positive_findings(text)
    assert Finding.CONSOLIDATION in findings
    assert Finding.PLEURAL_EFFUSION not in findings


def test_denial_is_recorded_separately_from_silence():
    """'Effusion was looked for and not seen' is different from 'not mentioned'."""
    labels = extract_labels("No pleural effusion is seen.")
    denied = [label for label in labels if not label.present]
    assert any(label.finding is Finding.PLEURAL_EFFUSION for label in denied)


def test_a_positive_elsewhere_outweighs_a_denial():
    text = "FINDINGS:\nNo effusion is seen.\n\nIMPRESSION:\n1. Left pleural effusion.\n"
    assert Finding.PLEURAL_EFFUSION in positive_findings(clinical_text(parse_sections(text)))


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("The cardiac silhouette is enlarged.", Finding.CARDIOMEGALY),
        ("Blunting of the left costophrenic angle is noted.", Finding.PLEURAL_EFFUSION),
        ("Patchy consolidation is seen in the right lower zone.", Finding.CONSOLIDATION),
        (
            "Homogeneous opacity with air bronchograms in the left lung field.",
            Finding.CONSOLIDATION,
        ),
        ("The heart is mildly enlarged.", Finding.CARDIOMEGALY),
        ("Pulmonary vascular congestion is again noted.", Finding.PULMONARY_EDEMA),
        ("A thin visceral pleural line is visible along the left apex.", Finding.PNEUMOTHORAX),
        ("Bilateral perihilar haziness with upper lobe diversion.", Finding.PULMONARY_EDEMA),
        ("There is volume loss in the left lower lobe.", Finding.ATELECTASIS),
        ("A well-circumscribed 12 mm nodular density is noted.", Finding.NODULE),
        ("A cortical break is seen involving the 7th rib.", Finding.FRACTURE),
        ("Fibrocavitary changes are seen in the right upper zone.", Finding.TUBERCULOSIS),
    ],
)
def test_findings_are_recognised_from_radiologist_phrasing(sentence, expected):
    """Reports say 'the cardiac silhouette is enlarged', not 'cardiomegaly'."""
    assert expected in positive_findings(sentence)


def test_volume_loss_as_a_modifier_is_not_an_atelectasis_assertion():
    """'Fibrocavitary changes with volume loss' describes tuberculosis, not collapse."""
    text = (
        "Fibrocavitary changes with volume loss are seen in the right upper zone, "
        "suggestive of post-primary infection."
    )
    findings = positive_findings(text)
    assert Finding.TUBERCULOSIS in findings
    assert Finding.ATELECTASIS not in findings


# --- Scrubbing --------------------------------------------------------------


def test_labelled_header_values_are_removed():
    text = (
        "Patient Name : Ravi Sharma\n"
        "MRN          : SMC-001234\n"
        "ABHA Number  : 99-1111-2222-3333\n"
    )
    out = scrub_report(text).text
    assert "Ravi" not in out
    assert "SMC-001234" not in out
    assert "99-1111-2222-3333" not in out
    assert "Patient Name" in out  # the field label itself is retained


def test_known_names_are_removed_from_prose_not_just_the_header():
    text = "FINDINGS:\nMr Sharma was imaged erect. Compared with the film at Sunrise Hospital.\n"
    out = scrub_report(
        text, known_names=["SHARMA^RAVI"], known_institutions=["Sunrise Hospital"]
    ).text
    assert "Sharma" not in out
    assert "Sunrise" not in out


def test_residual_doctor_names_are_caught_by_the_generic_pattern():
    out = scrub_report("Reported by Dr. Priya Menon\n").text
    assert "Priya" not in out and "Menon" not in out


def test_telephone_numbers_are_removed():
    assert "9945772185" not in scrub_report("Tel: +91-9945772185\n").text


def test_dates_are_shifted_consistently_rather_than_blanked():
    """A shifted date keeps the interval information a blanked one destroys."""
    pseudo = Pseudonymiser(KEY)
    pid = "a" * 16
    text = "COMPARISON:\nCompared with the radiograph dated 12-04-2024.\n"
    out = scrub_report(text, pseudonymiser=pseudo, pseudo_id=pid).text

    assert "12-04-2024" not in out
    assert "[DATE]" not in out  # it was shifted, not redacted
    shifted = pseudo.shift_date(__import__("datetime").date(2024, 4, 12), pid)
    assert shifted.strftime("%d-%m-%Y") in out


def test_dates_fall_back_to_redaction_without_a_pseudonymiser():
    assert "[DATE]" in scrub_report("Dated 12-04-2024.\n").text


def test_exact_age_in_the_identifying_tail_is_capped():
    out = scrub_report("Age / Sex    : 94 Y / M\n").text
    assert "94" not in out
    assert "89+" in out


def test_ordinary_ages_are_left_alone():
    assert "54" in scrub_report("Age / Sex    : 54 Y / F\n").text


def test_adjacent_placeholders_are_collapsed():
    out = scrub_report("Reported by Dr. Priya Menon\n").text
    assert "[NAME] [NAME]" not in out


# --- End to end -------------------------------------------------------------


def test_every_report_is_processed(processed):
    _, _, truth, result = processed
    assert result.n_reports == truth["n_studies"]


def test_no_generated_identifier_survives_in_any_scrubbed_report(processed):
    _, ws, truth, result = processed
    for record in result.records:
        text = (ws.reports_dir / record.relative_path).read_text(encoding="utf-8").upper()
        for phi in truth["phi_values"]:
            token = str(phi).strip().upper()
            if len(token) >= 5:
                assert token not in text, f"{token} survived in {record.relative_path}"


def test_extracted_labels_match_the_planted_findings(processed):
    """Precision and recall against ground truth, measured rather than asserted."""
    _, _, truth, result = processed
    expected = {s["accession"]: set(s["findings"]) - {"NO_FINDING"} for s in truth["studies"]}

    tp = fp = fn = 0
    for record in result.records:
        accession = record.source_path.rsplit("/", 1)[-1].removesuffix(".dcm")
        want = expected[accession]
        got = set(record.findings) - {"NO_FINDING"}
        tp += len(want & got)
        fp += len(got - want)
        fn += len(want - got)

    assert tp > 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    # High because the phrase bank and the generator share an author; this is a
    # consistency check on negation and section handling, not evidence about real prose.
    assert precision >= 0.95, f"precision {precision:.3f}"
    assert recall >= 0.95, f"recall {recall:.3f}"


def test_normal_studies_are_not_labelled_abnormal(processed):
    _, _, truth, result = processed
    normal = {
        s["accession"] for s in truth["studies"] if s["findings"] == ["NO_FINDING"]
    }
    assert normal
    for record in result.records:
        accession = record.source_path.rsplit("/", 1)[-1].removesuffix(".dcm")
        if accession in normal:
            assert set(record.findings) - {"NO_FINDING"} == set()


def test_redactions_are_counted_for_audit(processed):
    _, _, _, result = processed
    assert result.total_redactions > 0
    assert all(r.redaction_count > 0 for r in result.records)


# --- Behaviours learned from real prose (Open-i corpus) ---------------------


def test_clear_of_is_a_negation():
    """Boilerplate in real normal reports; without this cue it asserts three findings.

    This single construction accounted for the majority of false-positive
    pneumothorax calls before it was handled.
    """
    text = "The lungs are clear of focal airspace disease, pneumothorax, or pleural effusion."
    findings = positive_findings(text)
    assert Finding.PNEUMOTHORAX not in findings
    assert Finding.PLEURAL_EFFUSION not in findings
    assert Finding.CONSOLIDATION not in findings
    # The sentence is itself an assertion of normality, so that is the right label.
    assert findings == [Finding.NO_FINDING]


def test_grossly_clear_of_is_also_a_negation():
    text = (
        "The lungs are grossly clear of focal airspace disease, "
        "pneumothorax or pleural effusion."
    )
    assert Finding.PNEUMOTHORAX not in positive_findings(text)


def test_a_resolved_finding_is_not_asserted():
    """The cue follows the finding, so a prefix-only negation scope misses it."""
    assert Finding.PNEUMOTHORAX not in positive_findings(
        "The left apical pneumothorax has resolved."
    )
    assert Finding.PLEURAL_EFFUSION not in positive_findings(
        "Interval resolution of the right pleural effusion."
    )


@pytest.mark.parametrize(
    "sentence",
    [
        "The heart is mildly enlarged.",
        "The heart is again mildly enlarged.",
        "Heart size is moderately enlarged.",
        "There is stable enlargement of the cardiac silhouette.",
        "Mild cardiac enlargement is noted.",
    ],
)
def test_cardiomegaly_is_recognised_as_radiologists_write_it(sentence):
    """Real reports say 'the heart is enlarged', not 'cardiomegaly'."""
    assert Finding.CARDIOMEGALY in positive_findings(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "Pulmonary vascular congestion again noted.",
        "There is mild vascular prominence.",
        "Mild diffuse interstitial prominence suggestive of edema.",
    ],
)
def test_congestion_language_maps_to_oedema(sentence):
    """Open-i indexes 'Pulmonary Congestion' as oedema; reports rarely say oedema."""
    assert Finding.PULMONARY_EDEMA in positive_findings(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "Lungs are clear bilaterally.",
        "The lungs are well expanded and clear.",
        "No acute cardiopulmonary disease.",
        "No acute cardiopulmonary abnormality.",
        "Normal chest radiograph.",
    ],
)
def test_normal_studies_are_recognised_as_radiologists_write_them(sentence):
    """The original cues required 'lung fields are clear', which reports rarely write."""
    assert positive_findings(sentence) == [Finding.NO_FINDING]


def test_opacity_descriptors_are_not_called_consolidation():
    """'Airspace disease' is a descriptor; consolidation is a diagnosis.

    Conflating them scored recall 0.98 but precision 0.31 against radiologist
    annotation. Real annotators index opacity under a separate heading.
    """
    for descriptor in (
        "There is focal airspace disease in the right middle lobe.",
        "Streaky and patchy bibasilar opacities are noted.",
        "There is a vague increased opacity within the left lower lobe.",
    ):
        assert Finding.CONSOLIDATION not in positive_findings(descriptor)


def test_consolidation_proper_is_still_recognised():
    for sentence in (
        "There is dense consolidation involving the left mid zone.",
        "Homogeneous opacity with air bronchograms in the right lung field.",
        "Findings are compatible with pneumonia.",
    ):
        assert Finding.CONSOLIDATION in positive_findings(sentence)
