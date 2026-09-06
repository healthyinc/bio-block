"""The same word in two contexts must get two answers.

Phase 10 protected clinical content with a closed vocabulary: a word in the
list survived, a word outside it did not. That fails in both directions. It
destroys every medication, condition and eponym nobody thought to list, and it
shields a listed word unconditionally - so a patient genuinely surnamed Wilson
was protected by a rule written for Wilson disease.

These tests pin the replacement behaviour: the decision follows the evidence
around the span, with the vocabulary as one source rather than the verdict.
Each pair below is the *same* word, redacted in one sentence and preserved in
the other. The unseen-term cases matter just as much - they are what shows the
list is no longer doing the work.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.detection_evidence import (  # noqa: E402
    EVIDENCE_DETERMINISTIC,
    EVIDENCE_HEURISTIC,
    EVIDENCE_MODEL,
    assess_finding,
    clinical_reading_strength,
    count_agreeing,
)
from services.phi_detection import (  # noqa: E402
    SOURCE_NER,
    SOURCE_STRICT_PROPER_NOUN,
    SOURCE_STRUCTURED_PATTERN,
)
from services.text_anonymization import anonymize_clinical_text  # noqa: E402


def _output(text):
    return anonymize_clinical_text(text, profile="strict", study_salt="study-a")[
        "anonymized_text"
    ]


def _assert_preserved(text, term):
    output = _output(text)
    assert term in output, f"{term!r} was destroyed in: {output!r}"


def _assert_removed(text, term):
    output = _output(text)
    assert term not in output, f"{term!r} survived in: {output!r}"


# -- the same word, read two ways ------------------------------------------


CLINICAL_USE = [
    pytest.param(
        "Parkinson's disease was diagnosed by the neurologist.",
        "Parkinson",
        id="parkinson_disease",
    ),
    pytest.param(
        "Addison's disease is well controlled on replacement therapy.",
        "Addison",
        id="addison_disease",
    ),
    pytest.param(
        "Wilson disease was excluded by copper studies.",
        "Wilson",
        id="wilson_disease",
    ),
    pytest.param(
        "Bell palsy resolved without treatment.",
        "Bell",
        id="bell_palsy",
    ),
    pytest.param(
        "Cushing syndrome was suspected on clinical grounds.",
        "Cushing",
        id="cushing_syndrome",
    ),
    pytest.param(
        "Blood pressure remained stable throughout the admission.",
        "Blood pressure",
        id="blood_pressure_vital_sign",
    ),
]

NAMING_USE = [
    pytest.param("Dr Parkinson reviewed the chart.", "Parkinson", id="dr_parkinson"),
    pytest.param("Addison visited the clinic today.", "Addison", id="addison_visited"),
    pytest.param(
        "Patient Wilson was admitted for observation.", "Wilson", id="patient_wilson"
    ),
    pytest.param("Dr Bell signed the discharge summary.", "Bell", id="dr_bell"),
    pytest.param(
        "Dr Cushing performed the procedure.", "Cushing", id="dr_cushing"
    ),
    pytest.param(
        "Samples were sent to Blood Pressure Diagnostics Ltd.",
        "Blood Pressure Diagnostics",
        id="blood_pressure_as_a_company",
    ),
]


@pytest.mark.parametrize(("text", "term"), CLINICAL_USE)
def test_clinical_use_of_an_ambiguous_word_is_preserved(text, term):
    _assert_preserved(text, term)


@pytest.mark.parametrize(("text", "term"), NAMING_USE)
def test_naming_use_of_the_same_word_is_removed(text, term):
    _assert_removed(text, term)


# -- terms no list contains -------------------------------------------------


def test_unseen_condition_survives_on_grammar_not_membership():
    """An invented eponym is protected by the construction around it.

    "<Word> syndrome" reads as a condition whether or not the word has ever
    been recorded, which is the property a closed list cannot have.
    """
    _assert_preserved("Verrando syndrome was considered unlikely.", "Verrando")


def test_unseen_medication_survives_beside_its_dosage():
    _assert_preserved("Ranolazatib 250 mg twice daily was started.", "Ranolazatib")


def test_unseen_medication_without_dosage_context_is_not_assumed_safe():
    """No dosage, no grammar, no vocabulary: the word is not simply kept.

    Preserving an unrecognised capitalised word on the chance that it is a
    drug is how a surname survives. It is removed or escalated, never waved
    through.
    """
    text = "Ranolazatib was prescribed by Dr Anand Rajagopalan."
    output = _output(text)

    assert "Anand" not in output


def test_anatomy_is_preserved_where_geography_is_removed():
    _assert_preserved("The scan showed a lesion in the temporal lobe.", "temporal lobe")
    _assert_removed("The patient relocated to Coimbatore.", "Coimbatore")


# -- layer ordering ---------------------------------------------------------


def test_layer_one_structured_identifier_outranks_a_clinical_reading():
    """A regex match on identifier syntax is never overridden."""
    text = "Contact bell.palsy@example.org for the report."
    start = text.index("bell.palsy@example.org")
    assessment = assess_finding(
        text,
        start,
        start + len("bell.palsy@example.org"),
        "EMAIL_ADDRESS",
        SOURCE_STRUCTURED_PATTERN,
    )

    assert assessment.action == "redact"
    assert assessment.evidence == EVIDENCE_DETERMINISTIC


def test_layer_four_naming_context_outranks_the_vocabulary():
    """A recorded eponym used as a surname is still a surname."""
    text = "Dr Parkinson reviewed the chart."
    start = text.index("Parkinson")

    reading, rationale = clinical_reading_strength(
        text, start, start + len("Parkinson")
    )

    assert reading == "identifier"
    assert rationale == "person_context_present"


def test_layer_three_clinical_reading_blocks_a_weak_redaction():
    text = "Parkinson's disease was diagnosed by the neurologist."
    start = text.index("Parkinson")
    assessment = assess_finding(
        text, start, start + len("Parkinson"), "PERSON", SOURCE_STRICT_PROPER_NOUN
    )

    assert assessment.action == "preserve"
    assert assessment.evidence == EVIDENCE_HEURISTIC


def test_layer_two_agreeing_detectors_escalate_a_clinical_reading():
    """Two models against the dictionary is not something to decide silently."""
    text = "Parkinson's disease was diagnosed by the neurologist."
    start = text.index("Parkinson")
    assessment = assess_finding(
        text,
        start,
        start + len("Parkinson"),
        "PERSON",
        SOURCE_NER,
        agreeing_detectors=2,
    )

    assert assessment.action == "review"
    assert assessment.evidence == EVIDENCE_MODEL


def test_organisation_designator_is_naming_evidence_without_a_directory():
    text = "Samples were sent to Verrando Pressure Diagnostics Ltd."
    start = text.index("Verrando")

    reading, rationale = clinical_reading_strength(text, start, len(text) - 1)

    assert reading == "identifier"
    assert rationale == "organisation_designator_present"


# -- what counts as agreement ----------------------------------------------


class _Span:
    def __init__(self, start, end, entity_type, source):
        self.start = start
        self.end = end
        self.entity_type = entity_type
        self.source = source


def test_overlap_alone_is_not_agreement():
    """A structured hit beside a label is not a second opinion about the label.

    "MRN: 123456" produces a deterministic record-number match that overlaps a
    model calling the label "MRN" an organisation. Counting that as
    corroboration sent every ordinary field label to manual review.
    """
    entities = [
        _Span(12, 15, "ORGANIZATION", SOURCE_NER),
        _Span(12, 23, "MEDICAL_RECORD_NUMBER", SOURCE_STRUCTURED_PATTERN),
    ]

    agreeing = count_agreeing(
        entities, 12, 15, "ORGANIZATION", exclude_source=SOURCE_NER
    )

    assert agreeing == 1


def test_two_models_reading_a_span_the_same_way_do_agree():
    entities = [
        _Span(0, 9, "PERSON", SOURCE_NER),
        _Span(0, 9, "FACILITY", "gliner_multi_pii"),
    ]

    agreeing = count_agreeing(entities, 0, 9, "PERSON", exclude_source=SOURCE_NER)

    assert agreeing == 2


# -- categories that are not identifiers at all -----------------------------


def test_a_diagnosis_is_never_redacted_however_confident_the_model():
    """A medical condition is not one of the eighteen Safe Harbor identifiers.

    GLiNER is asked for a "medical condition" label because naming it stops
    the model reaching for PERSON when it sees an eponym. Treating what comes
    back as PHI inverted the pipeline: it redacted "Vital signs remained
    stable" and left the note medically empty.
    """
    text = "Vital signs remained stable throughout the admission."
    assessment = assess_finding(
        text, 0, len("Vital signs"), "MEDICAL_CONDITION", "gliner_multi_pii",
        agreeing_detectors=2,
    )

    assert assessment.action == "preserve"
    assert assessment.rationale == "category_is_not_a_safe_harbor_identifier"


def test_a_condition_prediction_protects_an_overlapping_name_guess():
    """One model's clinical reading is evidence against another's name guess.

    This is the corroboration a closed vocabulary cannot supply for a term
    nobody listed, and it arrives from a model rather than a list.
    """
    text = "Fluid balance was maintained overnight."
    assessment = assess_finding(
        text, 0, len("Fluid balance"), "PERSON", "gliner_multi_pii",
        clinical_support=True,
    )

    assert assessment.action == "preserve"
    assert assessment.rationale == "another_detector_read_the_span_as_clinical"


def test_naming_context_still_wins_over_a_condition_prediction():
    text = "Reviewed by Dr Wilson this morning."
    start = text.index("Wilson")
    assessment = assess_finding(
        text, start, start + len("Wilson"), "PERSON", "gliner_multi_pii",
        clinical_support=True,
    )

    assert assessment.action == "redact"


def test_a_single_character_is_not_replaced_with_a_patient_surrogate():
    """The "C" in "37.2°C" was being turned into PATIENT_002."""
    text = "Temperature 37.2°C recorded at intake."
    start = text.index("C", text.index("°"))
    assessment = assess_finding(
        text, start, start + 1, "ORGANIZATION", "gliner_multi_pii"
    )

    assert assessment.action == "preserve"
    assert assessment.rationale == "single_character_span_carries_no_identity"


def test_a_single_initial_in_a_naming_construction_is_still_redacted():
    text = "Seen by Dr R on the ward."
    start = text.index("R", text.index("Dr"))
    assessment = assess_finding(
        text, start, start + 1, "PERSON", "gliner_multi_pii"
    )

    assert assessment.action == "redact"


# -- the tagger as open-vocabulary evidence about names ---------------------


def test_a_span_with_no_proper_noun_is_not_replaced_with_a_surrogate():
    """"nursing staff" identifies nobody, and one model saying PERSON is not
    enough to turn it into a patient surrogate."""
    text = "No adverse reaction was recorded by nursing staff."
    start = text.index("nursing staff")
    assessment = assess_finding(
        text, start, start + len("nursing staff"), "PERSON", "gliner_multi_pii",
        contains_proper_noun=False,
    )

    assert assessment.action == "preserve"
    assert assessment.rationale == "no_proper_noun_token_in_span"


def test_two_agreeing_detectors_outrank_the_absence_of_a_proper_noun():
    """Corroboration exists for the name the tagger got wrong."""
    text = "No adverse reaction was recorded by nursing staff."
    start = text.index("nursing staff")
    assessment = assess_finding(
        text, start, start + len("nursing staff"), "PERSON", "gliner_multi_pii",
        agreeing_detectors=2,
        contains_proper_noun=False,
    )

    assert assessment.action == "redact"


def test_a_failed_parse_exempts_nothing():
    """None and False mean opposite things and must not be confused.

    An unavailable tagger reports None. Treating that as "no proper nouns
    here" would exempt every span in the document at once.
    """
    text = "Reviewed by Anand Rajagopalan on the ward."
    start = text.index("Anand")
    assessment = assess_finding(
        text, start, start + len("Anand Rajagopalan"), "PERSON",
        "gliner_multi_pii", contains_proper_noun=None,
    )

    assert assessment.action == "redact"


def test_a_lower_case_name_is_still_removed():
    """Capitalisation is not the signal; the tagger's reading of the word is.

    A name typed in lower case is exactly what a capitalisation rule misses,
    and the corpus carries the case deliberately.
    """
    _assert_removed(
        "CASE ESCALATED BY rukmini balasubramanian and reviewed later.",
        "balasubramanian",
    )


def test_role_nouns_and_label_words_survive_the_second_pass():
    for text, term in (
        ("No adverse reaction was recorded by nursing staff.", "nursing staff"),
        ("Enrolment completed without difficulty.", "Enrolment"),
    ):
        _assert_preserved(text, term)


# -- Layer 1 belongs to the regex, not to a model's category label ----------


def test_a_model_calling_a_word_a_phone_number_does_not_get_regex_authority():
    """Layer 1 exists for a match on identifier syntax.

    GLiNER labelled the bare word "Fax" a phone number. Extending Layer 1 to
    model findings by category alone gave that prediction the standing of a
    regex match, and it blocked the document with no further examination.
    """
    text = "Fax confirmation to 555-0123."
    assessment = assess_finding(text, 0, 3, "PHONE_NUMBER", "gliner_multi_pii")

    assert assessment.action != "redact"


def test_a_real_structured_span_still_wins_at_layer_one():
    text = "Contact synthetic.person@example.invalid for notes."
    start = text.index("synthetic")
    assessment = assess_finding(
        text, start, start + len("synthetic.person@example.invalid"),
        "EMAIL_ADDRESS", "gliner_multi_pii",
    )

    assert assessment.action == "redact"
    assert assessment.rationale == "exact_structured_match"


def test_a_deterministic_source_needs_no_further_proof():
    """The regex is the evidence; it does not have to argue for itself."""
    assessment = assess_finding(
        "MRN redacted", 0, 3, "MEDICAL_RECORD_NUMBER", SOURCE_STRUCTURED_PATTERN
    )

    assert assessment.action == "redact"
    assert assessment.rationale == "exact_structured_match"


# -- relative time is not a date element ------------------------------------


def test_a_relative_time_reference_is_not_a_date_element():
    """"the morning round" carries no date and removing it buys no privacy."""
    from services.detection_evidence import is_relative_time_expression

    assert is_relative_time_expression("DATE_TIME", "morning") is True
    assert is_relative_time_expression("DATE_TIME", "overnight") is True
    assert is_relative_time_expression("DATE_TIME", "14 March 2021") is False
    assert is_relative_time_expression("DATE_TIME", "Tuesday") is False
    assert is_relative_time_expression("PERSON", "morning") is False


def test_a_dated_reference_is_still_removed():
    _assert_removed("The final review was on 17 June 2011.", "17 June 2011")
