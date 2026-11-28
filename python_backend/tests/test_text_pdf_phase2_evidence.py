"""Phase 2 evidence rules for text and PDF, one decision at a time.

Each case states the construction in plain words and asserts what the evidence
model does with one proposed span. None of these names a fixture or a value
from a test pack: the strings are minimal constructions of the general rule.
"""

from __future__ import annotations

import pytest
from services import detection_evidence as de
from services.phi_detection import (
    SOURCE_CONTEXT_RULE,
    SOURCE_NER,
    SOURCE_STRICT_PROPER_NOUN,
    StructuredPatternDetector,
    fold_dashes,
)

STANFORD = "stanford_deidentifier"
GLINER = "gliner_multi_pii"


def _assess(text, span, category, source, agreeing=1, proper_noun=None, modifier=False):
    start = text.index(span)
    return de.assess_finding(
        text,
        start,
        start + len(span),
        category,
        source,
        agreeing_detectors=agreeing,
        contains_proper_noun=proper_noun,
        opening_modifier=modifier,
    )


def _kept(assessment):
    return assessment.action == "preserve"


# -- headings: capitals that come from layout ----------------------------------------------


def test_an_all_capitals_banner_gives_a_capitalisation_detector_no_evidence():
    text = "SYNTHETIC DEMO RECORD - NOT A REAL PATIENT\nPatient: Someone Else\n"
    assert _kept(_assess(text, "SYNTHETIC", "PERSON", SOURCE_STRICT_PROPER_NOUN))


def test_a_title_case_heading_of_three_words_is_not_a_name():
    text = "Clinical Discharge Summary\nPatient: Someone Else\n"
    assert _kept(_assess(text, "Clinical Discharge", "ORGANIZATION", SOURCE_NER))


def test_a_two_word_title_case_line_is_how_a_name_is_written():
    text = "Someone Else\nDiagnosis: asthma\n"
    result = _assess(text, "Someone Else", "PERSON", SOURCE_STRICT_PROPER_NOUN)
    assert result.rationale != "heading_capitalisation_carries_no_case_evidence"


def test_a_label_value_line_is_never_a_heading():
    text = "Patient: Someone Else\n"
    assert not de.line_carries_no_case_evidence("Patient: Someone Else")
    result = _assess(text, "Someone Else", "PERSON", SOURCE_NER)
    assert result.action == "redact"


def test_a_pinned_model_reading_a_banner_is_still_assessed_on_its_own_claim():
    text = "CLINIC LETTER\nPatient: Someone Else\n"
    result = _assess(text, "CLINIC LETTER", "PERSON", STANFORD)
    assert result.rationale != "heading_capitalisation_carries_no_case_evidence"


# -- spelled-out letters, field labels, document titles ------------------------------------


def test_letters_spelled_one_by_one_are_an_acronym_not_a_name():
    text = "G M C reg: 7011932\n"
    assert _kept(_assess(text, "G M C", "PERSON", SOURCE_STRICT_PROPER_NOUN))


def test_spelled_letters_after_a_naming_title_are_still_a_person():
    text = "Seen by Dr J R Smith today\n"
    assert not _kept(_assess(text, "J R", "PERSON", SOURCE_STRICT_PROPER_NOUN))


@pytest.mark.parametrize("key", ["Workstation address", "Home address", "GMC reg"])
def test_the_key_of_a_key_value_line_is_a_label(key):
    text = f"{key}: some value\n"
    assert _kept(_assess(text, key, "ADDRESS", GLINER))


def test_a_phrase_that_is_not_the_key_of_its_line_is_not_excused():
    text = "She moved to Workstation address, Springfield\n"
    assert not _kept(_assess(text, "Workstation address", "ADDRESS", GLINER))


def test_part_of_a_key_is_judged_as_the_whole_key():
    text = "GMC reg: 7011932\n"
    assert _kept(_assess(text, "GMC", "FACILITY", STANFORD, agreeing=2))


@pytest.mark.parametrize(
    "title", ["Referral Form", "Discharge Summary", "Progress Report"]
)
def test_a_title_ending_in_a_kind_of_document_is_not_a_name(title):
    text = f"{title}\nPatient: Someone Else\n"
    assert _kept(_assess(text, title, "PERSON", SOURCE_STRICT_PROPER_NOUN))


def test_a_name_span_that_takes_the_next_word_in_running_text_is_still_a_name():
    text = "Seen in clinic. Zelquin Vantreske Plan: continue the dose.\n"
    result = _assess(text, "Zelquin Vantreske Plan", "PERSON", SOURCE_NER)
    assert result.action == "redact"


def test_a_heading_with_a_single_letter_mark_beside_a_name_is_not_a_heading():
    assert not de.line_carries_no_case_evidence("H&E Jordan Fictional")
    assert not de.line_carries_no_case_evidence("Idris Nakashima-Holt")
    assert not de.line_carries_no_case_evidence("Dr 3 L Harrow")
    assert de.line_carries_no_case_evidence("Outpatient Discharge Summary")


# -- opening words, services, ward designators ---------------------------------------------


def test_an_opening_modifier_claimed_by_capitalisation_alone_is_kept():
    text = "Metabolic screen raised a question.\n"
    assert _kept(
        _assess(text, "Metabolic", "PERSON", SOURCE_NER, agreeing=1, modifier=True)
    )


def test_without_the_parse_showing_a_modifier_nothing_changes():
    text = "Metabolic screen raised a question.\n"
    result = _assess(text, "Metabolic", "PERSON", SOURCE_NER, agreeing=1)
    assert result.rationale != "opening_word_claimed_by_capitalisation_alone"


def test_the_same_word_with_a_second_detector_agreeing_is_not_excused_here():
    text = "Metabolic screen raised a question.\n"
    result = _assess(text, "Metabolic", "PERSON", SOURCE_NER, agreeing=2, modifier=True)
    assert result.rationale != "opening_word_claimed_by_capitalisation_alone"


def test_a_capital_in_the_middle_of_a_sentence_is_not_explained_by_position():
    text = "The patient was seen by Metabolic yesterday.\n"
    result = _assess(text, "Metabolic", "PERSON", SOURCE_NER, agreeing=1, modifier=True)
    assert result.rationale != "opening_word_claimed_by_capitalisation_alone"


def test_an_opening_word_no_pinned_model_named_is_kept_only_when_they_ran():
    text = "Haemoglobin and white cell count were within range.\n"
    start = text.index("Haemoglobin")
    ran = de.assess_finding(
        text,
        start,
        start + len("Haemoglobin"),
        "PERSON",
        SOURCE_NER,
        agreeing_detectors=1,
        opening_unconfirmed=True,
    )
    not_ran = de.assess_finding(
        text,
        start,
        start + len("Haemoglobin"),
        "PERSON",
        SOURCE_NER,
        agreeing_detectors=1,
        opening_unconfirmed=False,
    )
    assert (
        _kept(ran) and ran.rationale == "opening_word_claimed_by_capitalisation_alone"
    )
    assert not_ran.rationale != "opening_word_claimed_by_capitalisation_alone"


def test_a_pinned_model_claim_on_an_opening_word_is_never_excused_by_this_rule():
    text = "Kartik went home after lunch.\n"
    start = text.index("Kartik")
    result = de.assess_finding(
        text,
        start,
        start + len("Kartik"),
        "PERSON",
        STANFORD,
        agreeing_detectors=1,
        opening_unconfirmed=True,
        opening_modifier=True,
    )
    assert result.rationale != "opening_word_claimed_by_capitalisation_alone"


@pytest.mark.parametrize(
    "text,label",
    [
        ("Fax <PHONE_1>   Page 5\n", "Page"),
        ("Footer Pg 2 of 7\n", "Pg"),
        ("Page 3/9\n", "Page"),
    ],
)
def test_the_word_that_labels_a_page_number_is_structure(text, label):
    start = text.index(label)
    result = de.assess_finding(
        text, start, start + len(label), "PERSON", SOURCE_STRICT_PROPER_NOUN
    )
    assert _kept(result) and result.rationale == "page_number_label"


def test_page_without_a_number_after_it_is_not_a_page_label():
    text = "Mr Page attended clinic.\n"
    start = text.index("Page")
    result = de.assess_finding(
        text, start, start + 4, "PERSON", SOURCE_STRICT_PROPER_NOUN
    )
    assert result.rationale != "page_number_label"


def test_the_parse_finds_a_modifier_before_a_noun_and_not_a_name_before_a_verb():
    from services.ner_phi_detector import configured_model_name, noun_evidence

    text = "Metabolic screen raised a question.\nKartik went home after lunch.\n"
    _proper, modifiers = noun_evidence(text, configured_model_name())
    starts = {low for low, _high in modifiers}
    assert text.index("Metabolic") in starts
    assert text.index("Kartik") not in starts


def test_a_clinic_named_by_subject_words_is_a_department():
    text = "Genetics clinic review\n"
    assert _kept(
        _assess(text, "Genetics", "ORGANIZATION", GLINER, agreeing=2, proper_noun=False)
    )


def test_a_clinic_with_a_proper_noun_in_it_is_an_organisation():
    text = "Riverbend clinic review\n"
    assert not _kept(
        _assess(text, "Riverbend", "ORGANIZATION", GLINER, agreeing=2, proper_noun=True)
    )


def test_ward_designators_with_numbers_are_not_a_name():
    text = "Bed 14, Cubicle 3, Ward 5 - take 4 tablets\n"
    assert _kept(_assess(text, "Bed 14, Cubicle 3, Ward 5", "PERSON", GLINER))


# -- times: durations, offsets and clock times ---------------------------------------------


@pytest.mark.parametrize(
    "sentence,span",
    [
        ("Review in 6 weeks.", "6 weeks"),
        ("Seen 2 days ago.", "2 days ago"),
        ("Post-operative day 5, afebrile.", "day 5"),
        ("Symptoms started 3 months before the visit.", "3 months before"),
    ],
)
def test_a_relative_offset_has_no_calendar_content(sentence, span):
    assert _kept(_assess(sentence + "\n", span, "DATE_TIME", SOURCE_NER))


def test_an_offset_past_an_age_is_not_relative():
    result = _assess("Born 93 years ago.\n", "93 years ago", "DATE_TIME", SOURCE_NER)
    assert result.action == "redact"


def test_a_time_of_day_without_a_date_is_clinical_sequencing():
    text = "Capillary glucose 208 mg/dL at 14:00, repeat at 18:00.\n"
    assert _kept(_assess(text, "14:00", "TIME", "structured_pattern"))


def test_a_time_beside_a_date_keeps_the_handling_it_had():
    text = "Admitted 11/02/2026 at 14:00 to the ward.\n"
    assert _assess(text, "14:00", "TIME", "structured_pattern").action == "redact"


# -- token shapes in the input -------------------------------------------------------------


def test_a_model_claim_on_input_shaped_like_our_token_is_replaced_not_escalated():
    text = "Mail <EMAIL_58> or someone.\n"
    result = _assess(text, "<EMAIL_58>", "EMAIL_ADDRESS", GLINER, agreeing=2)
    assert result.action == "redact" and result.justified
    assert result.rationale == "token_shaped_source_text"


# -- eponyms and named instruments ---------------------------------------------------------


@pytest.mark.parametrize(
    "sentence,span,category",
    [
        ("A Foley catheter in situ.", "Foley", "PERSON"),
        ("Stool chart Bristol stool chart type 4.", "Bristol", "ORGANIZATION"),
        (
            "Conscious level Glasgow Coma Scale 14.",
            "Glasgow Coma Scale",
            "ORGANIZATION",
        ),
        ("Language centres: Broca area intact.", "Broca", "PERSON"),
    ],
)
def test_an_eponymous_instrument_or_scale_is_clinical(sentence, span, category):
    assert _kept(_assess(sentence + "\n", span, category, SOURCE_NER, agreeing=1))


def test_a_place_followed_by_area_is_not_an_eponym():
    result = _assess(
        "She lives in the Jordan area now.\n", "Jordan", "LOCATION", SOURCE_NER
    )
    assert not _kept(result)


# -- structured grammar: dashes, national numbers -------------------------------------------


def test_dash_variants_fold_to_a_hyphen_without_moving_an_offset():
    text = "call 206­555–0148 now; or 12;"
    folded = fold_dashes(text)
    assert len(folded) == len(text)
    assert folded == "call 206-555-0148 now; or 12;"


@pytest.mark.parametrize(
    "number",
    [
        "0161 496 0731",
        "020 7946 0112",
        "07700 900412",
        "+44 20 7946 0873",
        "206­555­0148",
    ],
)
def test_non_north_american_and_dash_variant_numbers_are_phone_numbers(number):
    found = StructuredPatternDetector().detect(f"Contact on {number} please.")
    assert any(entity.entity_type == "PHONE_NUMBER" for entity in found), number


@pytest.mark.parametrize(
    "text",
    [
        "Take 0.5 mg twice daily.",
        "Weight 0161 kg",
        "Range 10 to 20 mg",
        "Dose 0150 0200 0300 mg",
    ],
)
def test_amounts_are_not_taken_for_national_numbers(text):
    found = StructuredPatternDetector().detect(text)
    assert not any(entity.entity_type == "PHONE_NUMBER" for entity in found), text


def test_the_independent_residual_scan_reads_spaced_national_numbers_too():
    from services.residual_identifier_scan import scan

    found = scan("Direct dial 0161 496 0731 for results")
    assert any(entity.entity_type == "PHONE_NUMBER" for entity in found)
    assert not any(
        entity.entity_type == "PHONE_NUMBER"
        for entity in scan("Counts 0161 496 mg and 0731 doses")
    )


# -- the utility measure counts occurrences, not values ------------------------------------


def test_a_removed_name_equal_to_an_eponym_does_not_charge_the_disease():
    from services.text_utility import measure_text_utility

    original = "Seen by Dr Hannah Crohn. Known Crohn disease."
    start = original.index("Hannah")
    anonymized = "Seen by Dr <PROVIDER_1>. Known Crohn disease."
    metrics = measure_text_utility(
        original, anonymized, redacted_spans=[(start, start + len("Hannah Crohn"))]
    )
    assert metrics["clinical_term_preservation"] == 1.0


def test_a_kept_occurrence_that_is_lost_is_still_charged():
    from services.text_utility import measure_text_utility

    original = "Seen by Dr Hannah Crohn. Known Crohn disease."
    start = original.index("Hannah")
    anonymized = "Seen by Dr <PROVIDER_1>. Known <PERSON_1> disease."
    metrics = measure_text_utility(
        original, anonymized, redacted_spans=[(start, start + len("Hannah Crohn"))]
    )
    assert metrics["clinical_term_preservation"] < 1.0


# -- overlap typing ------------------------------------------------------------------------


def test_the_longest_reading_names_what_a_merged_span_is():
    from services.phi_detection import DetectedEntity
    from services.text_anonymization import _merge_overlapping_name_candidates

    text = "Address: 14 Marlowe Terrace, Ashbourne, Derbyshire\n"
    address = DetectedEntity("ADDRESS", 9, 49, GLINER, 0.9, "address")
    person = DetectedEntity("PERSON", 12, 19, SOURCE_NER, None, "PERSON")
    merged = _merge_overlapping_name_candidates([address, person], text)
    assert [entity.entity_type for entity in merged] == ["ADDRESS"]


def test_a_naming_context_rule_still_makes_the_merged_span_a_person():
    from services.phi_detection import DetectedEntity
    from services.text_anonymization import _merge_overlapping_name_candidates

    text = "Seen by Dr Avery Morgan-Hale today\n"
    context = DetectedEntity("PERSON", 11, 20, SOURCE_CONTEXT_RULE, None, "ctx")
    longer = DetectedEntity("ORGANIZATION", 8, 28, GLINER, 0.9, "org")
    merged = _merge_overlapping_name_candidates([context, longer], text)
    assert [entity.entity_type for entity in merged] == ["PERSON"]


# -- the PDF read-back: noise is not hidden text -------------------------------------------


# -- labels left in front of a replaced value -----------------------------------------------


@pytest.mark.parametrize(
    "text, span, category",
    [
        (
            "Record. Account # <ACCOUNT_ID_1>; seen today.",
            "Account # <ACCOUNT_ID_1>",
            "PERSON",
        ),
        (
            "Record. Driver licence <LICENSE_ID_1>; BP stable.",
            "Driver licence",
            "DRIVER_LICENSE",
        ),
        (
            "Record. pacemaker serial <DEVICE_ID_1>; BP stable.",
            "pacemaker serial",
            "DEVICE_ID",
        ),
    ],
)
def test_the_label_left_in_front_of_a_replaced_value_is_a_field_name(
    text, span, category
):
    result = _assess(text, span, category, GLINER)
    assert result.action == "preserve"
    assert result.rationale == "label_of_a_replaced_value"


def test_a_label_with_no_replaced_value_after_it_is_not_excused_by_that_rule():
    text = "Record. Driver licence belongs to the patient.\n"
    assert (
        _assess(text, "Driver licence", "DRIVER_LICENSE", GLINER).rationale
        != "label_of_a_replaced_value"
    )


def test_a_name_in_front_of_a_replaced_value_is_not_a_label():
    text = "Record. Harriet Voss <PERSON_1> was seen.\n"
    assert (
        _assess(text, "Harriet Voss", "PERSON", GLINER).rationale
        != "label_of_a_replaced_value"
    )


def test_a_deterministic_match_is_never_excused_as_a_label():
    text = "Record. Account <ACCOUNT_ID_1> 123456789 noted.\n"
    result = _assess(text, "Account", "ACCOUNT_NUMBER", "structured_pattern")
    assert result.rationale != "label_of_a_replaced_value"


def test_a_numbered_label_before_a_replaced_value_is_recognised_as_one():
    text = "Record. Insurance no: <HEALTH_PLAN_ID_2> noted.\n"
    start = text.index("Insurance no")
    assert de._label_of_a_replaced_value(text, start, start + len("Insurance no"))
