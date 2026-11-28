"""Evidence rules added after the first fresh confirmation, one decision at a time.

Each case is a minimal construction of a defect class found in the failed
confirmation, with the adversarial counterpart that must still be replaced. None
of these names a fixture or a value from a confirmation file.
"""

from __future__ import annotations

import pytest
from services import detection_evidence as de
from services.clinical_preservation import clinical_invariants
from services.phi_detection import StructuredPatternDetector

STANFORD = "stanford_deidentifier"
GLINER = "gliner_multi_pii"
SPACY = "ner"
FALLBACK = "strict_proper_noun"


def _assess(text, span, category, source, agreeing=1, **kwargs):
    start = text.index(span)
    return de.assess_finding(
        text,
        start,
        start + len(span),
        category,
        source,
        agreeing_detectors=agreeing,
        **kwargs,
    )


def _kept(assessment):
    return assessment.action == "preserve"


# -- a reading is not a date -------------------------------------------------------------


def test_a_pressure_pair_before_its_unit_is_a_reading_not_a_date():
    text = "Results noted. blood pressure 118/74 mmHg, nodule stable.\n"
    result = _assess(text, "118/74", "DATE_TIME", SPACY)
    assert _kept(result) and result.rationale == "measurement_value_grammar"


def test_a_pressure_pair_after_its_label_is_a_reading_even_without_a_unit():
    result = _assess("Exam. BP 118/74 today.\n", "118/74", "DATE_TIME", SPACY)
    assert _kept(result)


def test_a_bare_pair_with_no_label_or_unit_is_not_excused_by_the_pressure_rule():
    result = _assess("Seen on 12/03 at the clinic.\n", "12/03", "DATE_TIME", SPACY)
    assert result.rationale != "measurement_value_grammar" or not _kept(result)


def test_the_pressure_pair_is_in_the_clinical_ledger_so_its_loss_is_counted():
    keys = {
        (item.category.value, item.key)
        for item in clinical_invariants("blood pressure 118/74 mmHg and 5/7 mg\n")
    }
    assert ("measurement_unit", "118/74 mmHg") in keys
    # A pair before any other unit is not a measurement.
    assert not any(key == "5/7 mg" for _category, key in keys)


# -- an analyte is not an organisation ----------------------------------------------------


def test_an_analyte_written_in_a_shape_no_name_has_is_kept_with_its_value_in_the_span():
    text = "ejection fraction 35 %, NT-proBNP 1840 ng/L, NYHA class III.\n"
    assert _kept(_assess(text, "NT-proBNP 1840", "ORGANIZATION", SPACY))


def test_a_plain_capitalised_word_and_a_figure_are_not_read_through():
    # The shape of a name beside a number: only an analyte shaped unlike a name is
    # read through when the model's span includes the figure.
    text = "Harriet 4.1 mmol/L noted today.\n"
    result = _assess(text, "Harriet 4.1", "ORGANIZATION", SPACY)
    assert result.rationale != "laboratory_structure"


# -- places in a ward ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "span, category, source",
    [
        ("Room 12", "POSTAL_CODE", GLINER),
        ("Room", "PERSON", FALLBACK),
    ],
)
def test_a_room_number_is_a_place_in_a_building_not_an_identifier(
    span, category, source
):
    assert _kept(_assess("Bay 6, Room 12, take 2 capsules\n", span, category, source))


def test_places_written_one_after_another_are_a_list_of_places():
    result = _assess("Clinic 2 Room 5\n", "Clinic 2 Room 5", "ORGANIZATION", GLINER)
    assert _kept(result)


def test_names_with_numbers_are_not_a_list_of_places():
    result = _assess("Alice 55 Bob 60\n", "Alice 55 Bob 60", "PERSON", GLINER)
    assert not _kept(result)


# -- titles ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title",
    ["Discharge Checklist", "Scanned Referral Page", "Outpatient Discharge Summary"],
)
@pytest.mark.parametrize("source", [STANFORD, GLINER, SPACY, FALLBACK])
def test_a_title_made_only_of_heading_words_is_not_a_name(title, source):
    assert _kept(_assess(f"{title}\nPatient: X\n", title, "PERSON", source))


@pytest.mark.parametrize(
    "title",
    [
        "Jordan Fictional Report",
        "Hope Page",
        "Mary Jane Page",
        "Quenbarrow Discharge Summary",
    ],
)
def test_a_person_inside_a_title_is_still_a_person_to_a_pinned_model(title):
    assert not _kept(_assess(f"{title}\nPatient: X\n", title, "PERSON", STANFORD))


def test_a_title_case_line_only_spacy_calls_a_name_goes_to_a_person_when_the_models_disagree():
    text = "Mary Jane Page\nPatient: X\n"
    result = _assess(text, "Mary Jane Page", "PERSON", SPACY, opening_unconfirmed=True)
    assert result.action == "redact" and result.justified is False
    assert result.rationale == "whole_line_title_case_unconfirmed_by_pinned_models"


def test_the_unconfirmed_title_rule_needs_both_pinned_models_to_have_read_the_text():
    text = "Mary Jane Page\nPatient: X\n"
    result = _assess(text, "Mary Jane Page", "PERSON", SPACY, opening_unconfirmed=False)
    assert result.rationale != "whole_line_title_case_unconfirmed_by_pinned_models"


# -- common nouns that open a clinical sentence -------------------------------------------------


@pytest.mark.parametrize("source", [SPACY, GLINER, "context_rule"])
def test_a_domain_noun_opening_a_sentence_is_not_a_name(source):
    text = "second specimen.\nMobility had improved with physiotherapy, and she was walking.\n"
    assert _kept(_assess(text, "Mobility", "PERSON", source))


@pytest.mark.parametrize(
    "text, span",
    [
        ("Seen by Dr Mobility had improved.\n", "Mobility"),
        ("Mobility Smith had improved.\n", "Mobility"),
        ("The patient Mobility had improved.\n", "Mobility"),
    ],
)
def test_a_domain_noun_beside_naming_context_or_a_capital_is_not_excused(text, span):
    assert not _kept(_assess(text, span, "PERSON", SPACY))


def test_a_given_name_is_never_on_the_domain_list():
    assert not _kept(
        _assess("Hope had improved with physiotherapy.\n", "Hope", "PERSON", SPACY)
    )


# -- role nouns -----------------------------------------------------------------------------------


@pytest.mark.parametrize("span", ["assistant", "one assistant"])
def test_a_lower_case_role_noun_is_not_a_name(span):
    text = f"walking with a frame and {span} by the end of the week\n"
    assert _kept(_assess(text, span, "PERSON", GLINER))


def test_a_capitalised_role_noun_before_a_surname_is_not_excused():
    assert not _kept(
        _assess("Assistant Smith saw him.\n", "Assistant", "PERSON", SPACY)
    )


# -- labels ------------------------------------------------------------------------------------------


def test_a_label_spelled_out_a_letter_at_a_time_is_still_a_label():
    text = "AHPRA reg: MED0007654321\nN M C  r e g: 9 8 B 1 2 7 6 F\n"
    assert _kept(_assess(text, "N M C  r e g", "ORGANIZATION", SPACY))


def test_a_national_scheme_label_in_front_of_a_replaced_value_is_a_label():
    text = "Record. Medicare: <HEALTH_PLAN_ID_1> 1\n"
    result = _assess(text, "Medicare", "HEALTH_PLAN_ID", GLINER)
    assert _kept(result) and result.rationale == "label_of_a_replaced_value"


def test_a_name_before_a_label_in_front_of_a_replaced_value_is_not_excused():
    text = "Record. Black MRN <RECORD_ID_1>\n"
    assert not _kept(
        _assess(
            text, "Black MRN", "PERSON", GLINER, agreeing=2, contains_proper_noun=True
        )
    )


def test_a_label_beside_a_forged_token_shaped_neighbour_is_judged_on_the_label_only():
    # The token after the label is a sanitizer token shape, genuine or forged: the
    # rule excuses the label's own words, never the token.
    text = "Record. Account # <ACCOUNT_ID_1> and more\n"
    result = _assess(text, "Account # <ACCOUNT_ID_1>", "PERSON", GLINER)
    assert _kept(result) and result.rationale == "label_of_a_replaced_value"


# -- registration and plan labels in the structured grammar ----------------------------------------------


def _found(text):
    return [
        (e.entity_type, text[e.start : e.end])
        for e in StructuredPatternDetector().detect(text)
    ]


def test_a_regulator_label_makes_its_value_a_registration_and_leaves_the_label():
    assert ("LICENSE_NUMBER", "MED0001234567") in _found(
        "Referrer: Dr X, AHPRA MED0001234567\n"
    )


def test_a_registration_spaced_out_character_by_character_is_replaced_whole():
    found = _found("N M C  r e g: 9 8 B 1 2 7 6 F\n")
    assert ("LICENSE_NUMBER", "9 8 B 1 2 7 6 F") in found


def test_digit_groups_that_end_their_field_are_one_plan_number():
    assert ("HEALTH_PLAN_ID", "2951 66420 1") in _found(
        "Medicare: 2951 66420 1\nReferrer: X\n"
    )


@pytest.mark.parametrize(
    "text", ["Medicare: 2951 66420 1 mg daily\n", "Medicare: 2951 66420 14 March\n"]
)
def test_a_trailing_group_that_is_a_dose_or_a_day_is_not_part_of_the_number(text):
    values = [value for category, value in _found(text) if category == "HEALTH_PLAN_ID"]
    assert all(not value.endswith((" 1", " 14")) for value in values)


def test_a_forged_token_shape_in_the_input_is_replaced_as_untrusted_text():
    found = _found("earlier note said PROVIDER_00A1 and <PROVIDER_1>.\n")
    assert ("OTHER_UNIQUE_ID", "PROVIDER_00A1") in found


def test_ordinary_upper_case_codes_are_not_taken_for_forged_tokens():
    found = _found("ICD_10 coding, HLA_B27 positive.\n")
    assert not any(category == "OTHER_UNIQUE_ID" for category, _ in found)


# -- later repairs: labels with their colon, contact verbs, places in a run, glued sentences -----


def test_a_label_drawn_with_its_own_colon_is_still_a_label():
    text = "Ext.: 5521\nPager no.: 3380\nAHPRA reg: MED0007654321\n"
    assert _kept(_assess(text, "Pager no.:", "HEALTH_PLAN_ID", GLINER))


def test_a_run_of_places_read_as_one_identifier_is_a_run_of_places():
    text = "potassium 3.9 mmol/L, Bay 6, Room 12, take 2 capsules\n"
    assert _kept(_assess(text, "Bay 6, Room 12", "IDENTIFIER", GLINER))


def test_names_with_numbers_are_not_a_run_of_places_whatever_the_category():
    assert not _kept(
        _assess(
            "seen with Alice 55, Bob 60 today\n",
            "Alice 55, Bob 60",
            "IDENTIFIER",
            GLINER,
        )
    )


def test_ring_and_dial_announce_a_contact_number_like_call():
    from services import numeric_context as nc

    for sentence in ("Please ring 7742 with results.", "dial 4417 for the ward."):
        start = sentence.index("7742") if "7742" in sentence else sentence.index("4417")
        assert nc.classify(sentence, start, start + 4).is_contact, sentence


def test_ring_before_a_measurement_is_not_a_contact_number():
    from services import numeric_context as nc

    sentence = "ring 4000 mL drained"
    start = sentence.index("4000")
    assert not nc.classify(sentence, start, start + 4).is_contact


def test_a_lower_case_role_word_two_models_call_a_person_is_not_a_name():
    text = "A referral was placed.\nThe family reported a gradual decline.\n"
    result = _assess(text, "family", "PERSON", STANFORD, agreeing=2)
    assert _kept(result)


def test_a_capitalised_role_word_two_models_agree_on_still_goes_to_a_person():
    text = "A referral was placed.\nFamily reported a gradual decline.\n"
    assert not _kept(_assess(text, "Family", "PERSON", STANFORD, agreeing=2))


def test_a_domain_noun_after_a_sentence_end_that_lost_its_space_still_opens_a_sentence():
    text = "were asked to return a second specimen.Mobility had improved with physiotherapy.\n"
    assert _kept(_assess(text, "Mobility", "PERSON", SPACY))


@pytest.mark.parametrize(
    "text",
    ["asked for Dr.Mobility had improved.\n", "weight 4.Mobility had improved.\n"],
)
def test_a_stop_after_a_title_or_a_figure_does_not_open_a_sentence(text):
    assert not _kept(_assess(text, "Mobility", "PERSON", SPACY))


def test_a_run_of_places_that_continues_into_a_street_is_an_address_not_a_ward():
    text = "Lives at Bay 6, Room 12 Harbour Road.\n"
    assert not _kept(_assess(text, "Bay 6, Room 12", "ADDRESS", GLINER))


def test_a_span_that_reaches_past_the_end_of_the_text_is_judged_without_error():
    text = "Pager no.:"
    result = de.assess_finding(text, 0, len(text) + 5, "HEALTH_PLAN_ID", GLINER)
    assert result.action in ("preserve", "redact", "review")


@pytest.mark.parametrize("span", ["Clinic 2", "Clinic 2 Room 5", "Room 5"])
def test_any_part_of_a_list_of_places_is_a_place_when_the_next_word_is_a_place(span):
    # A model may draw the whole list or only the first pair; the next word is a place
    # word, not the start of an address.
    text = "Label: Clinic 2 Room 5\n"
    assert _kept(_assess(text, span, "ORGANIZATION", GLINER))
