"""General person-span closure and safe clinical controls for Phase 17."""

import re

import pytest
from services.local_model_detectors import SOURCE_GLINER, SOURCE_STANFORD
from services.ner_phi_detector import _explicit_person_context_entities
from services.phi_detection import (
    SOURCE_CONTEXT_RULE,
    SOURCE_STRUCTURED_PATTERN,
    DetectedEntity,
    resolve_overlaps,
)
from services.text_anonymization import (
    _apply_evidence_model,
    _merge_adjacent_same_type,
    _merge_overlapping_name_candidates,
    anonymize_clinical_text,
)


def _words(value):
    return [word.casefold() for word in re.findall(r"[^\W\d_]{4,}", value, re.UNICODE)]


def _assert_names_absent(text, names):
    result = anonymize_clinical_text(
        text, profile="strict", study_salt="phase17-matrix"
    )
    output = result["anonymized_text"].casefold()
    for name in names:
        assert name.casefold() not in output
        assert all(fragment not in output for fragment in _words(name))
    assert result["residual_phi_categories"].get("PERSON", 0) == 0


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("Patient: Alviera Dornethyl", ("Alviera Dornethyl",)),
        ("Dr. Caelen Voss reviewed the chart.", ("Caelen Voss",)),
        ("Patient: J. Voss arrived.", ("J. Voss",)),
        ("Patient: J. R. Voss arrived.", ("J. R. Voss",)),
        ("Patient: Ana-Luisa Morcant arrived.", ("Ana-Luisa Morcant",)),
        ("Patient: Eira O'Valen arrived.", ("Eira O'Valen",)),
        ("Alviera Dornethyl's medication was reviewed.", ("Alviera Dornethyl",)),
        ("Patient: Élodie Vársen arrived.", ("Élodie Vársen",)),
        ("Patient: ALVIERA DORNETHYL", ("ALVIERA DORNETHYL",)),
        ("Patient: alviera dornethyl", ("alviera dornethyl",)),
        ("Patient: Alviera\nDornethyl attended.", ("Alviera Dornethyl",)),
        ("Patient: Alviera   Dornethyl attended.", ("Alviera Dornethyl",)),
        ("Patient: Alviera Dornethyl; visit 12 November 2026.", ("Alviera Dornethyl",)),
        ("Patient: Alviera Dornethyl; MRN FZ814221.", ("Alviera Dornethyl",)),
        ("Patient: Alviera Dornethyl; diagnosis asthma.", ("Alviera Dornethyl",)),
        ("Patient: Alviera Dornethyl; continue ramipril 5 mg.", ("Alviera Dornethyl",)),
        ("Patient: Alviera Dornethyl; sodium 139 mmol/L.", ("Alviera Dornethyl",)),
        (
            "Alviera Dornethyl called. Alviera Dornethyl returned.",
            ("Alviera Dornethyl",),
        ),
        ("Alviera Voss and Caelen Voss attended.", ("Alviera Voss", "Caelen Voss")),
        (
            "Alviera Dornethyl reported pain at the beginning; end: Alviera Dornethyl",
            ("Alviera Dornethyl",),
        ),
    ],
)
def test_name_variation_matrix_removes_complete_names_and_fragments(text, names):
    _assert_names_absent(text, names)


def test_a_short_rule_span_cannot_discard_a_complete_model_person_span():
    text = "Patient: Alviera Dornethyl"
    start = text.index("Alviera")
    complete = DetectedEntity("PERSON", start, len(text), SOURCE_GLINER, 0.99, "person")
    short = DetectedEntity(
        "PERSON",
        start,
        start + len("Alviera"),
        SOURCE_CONTEXT_RULE,
        None,
        "PATIENT_PERSON",
    )
    merged = _merge_overlapping_name_candidates([short, complete], text)
    resolved = _merge_adjacent_same_type(resolve_overlaps(merged, len(text)), text)
    assert [(entity.start, entity.end, entity.entity_type) for entity in resolved] == [
        (start, len(text), "PERSON")
    ]


@pytest.mark.parametrize("detected_part", ["Alviera", "Dornethyl"])
def test_explicit_person_context_closes_a_first_only_or_surname_only_detection(
    detected_part,
):
    text = "Patient: Alviera Dornethyl"
    start = text.index(detected_part)
    partial = DetectedEntity(
        "PERSON", start, start + len(detected_part), SOURCE_GLINER, 0.99, "person"
    )
    contextual = _explicit_person_context_entities(text, [partial])
    assert [(entity.start, entity.end) for entity in contextual] == [
        (text.index("Alviera"), len(text))
    ]


def test_a_complete_misclassified_name_span_is_unioned_with_person_evidence():
    text = "Patient: Alviera Dornethyl"
    start = text.index("Alviera")
    partial = DetectedEntity(
        "PERSON",
        start,
        start + len("Alviera"),
        SOURCE_CONTEXT_RULE,
        None,
        "PATIENT_PERSON",
    )
    misclassified = DetectedEntity(
        "ORGANIZATION", start, len(text), SOURCE_GLINER, 0.99, "organization"
    )
    merged = _merge_overlapping_name_candidates([partial, misclassified], text)
    assert [(entity.start, entity.end, entity.entity_type) for entity in merged] == [
        (start, len(text), "PERSON")
    ]


def test_name_union_keeps_a_provider_title_for_role_classification():
    text = "Dr. Shalini Muthukrishnan signed the note."
    model = DetectedEntity(
        "PERSON",
        text.index("Dr."),
        text.index(" signed"),
        SOURCE_GLINER,
        0.99,
        "person",
    )
    context = DetectedEntity(
        "PERSON",
        text.index("Shalini"),
        text.index(" signed"),
        SOURCE_CONTEXT_RULE,
        None,
        "TITLE_PERSON",
    )
    merged = _merge_overlapping_name_candidates([model, context], text)
    assert [(entity.start, entity.end) for entity in merged] == [
        (text.index("Shalini"), text.index(" signed"))
    ]
    result = anonymize_clinical_text(
        text, profile="strict", study_salt="phase17-provider"
    )
    assert "Dr. <PROVIDER_" in result["anonymized_text"]


def test_an_ambiguous_inner_reading_does_not_hold_an_exact_enclosing_identifier():
    text = "Email: user4821@example.invalid"
    start = text.index("user4821")
    exact = DetectedEntity(
        "EMAIL_ADDRESS",
        start,
        len(text),
        SOURCE_STRUCTURED_PATTERN,
        None,
        "email_address",
    )
    uncertain = DetectedEntity(
        "PHONE_NUMBER",
        start,
        start + len("user4821"),
        SOURCE_STANFORD,
        0.8,
        "PHONE",
    )
    kept, review = _apply_evidence_model([exact, uncertain], text)
    assert review == []
    assert any(entity.entity_type == "EMAIL_ADDRESS" for entity in kept)


@pytest.mark.parametrize(
    ("text", "preserved"),
    [
        ("Parkinson's disease remained stable.", "Parkinson's disease"),
        ("Alzheimer disease was discussed.", "Alzheimer disease"),
        ("Hodgkin lymphoma is in remission.", "Hodgkin lymphoma"),
        ("Crohn disease remained quiescent.", "Crohn disease"),
        ("Type 2 Diabetes was reviewed.", "Diabetes"),
        ("Ramipril 5 mg daily was continued.", "Ramipril 5 mg"),
        ("Budesonide 200 mcg was continued.", "Budesonide 200 mcg"),
        ("The pacemaker device was checked.", "pacemaker"),
        ("The nursing team reviewed the plan.", "nursing team"),
        ("The attending physician reviewed the plan.", "attending physician"),
        ("The left iliac crest was tender.", "iliac crest"),
        ("The right femoral head was painless.", "femoral head"),
        ("Sodium was 139 mmol/L.", "139 mmol/L"),
        ("Temperature was 37.2°C.", "37.2°C"),
        ("eGFR was 52 mL/min/1.73m2.", "mL/min"),
        ("Diagnosis N18.3 remained active.", "N18.3"),
        ("ASSESSMENT: asthma remains controlled.", "asthma remains controlled"),
        ("PLAN: continue inhaled therapy.", "continue inhaled therapy"),
        ("The multidisciplinary team agreed.", "multidisciplinary team"),
        ("Monday 12 November 2026 was the genuine visit date.", "genuine visit date"),
    ],
)
def test_safe_clinical_controls_do_not_gain_a_person_redaction(text, preserved):
    result = anonymize_clinical_text(
        text, profile="strict", study_salt="phase17-controls"
    )
    assert result["detected_entities"].get("PERSON", 0) == 0
    assert preserved.casefold() in result["anonymized_text"].casefold()
