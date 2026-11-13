"""A field heading is not the name of a person.

A clinical record is a sequence of labelled fields, and the label carries
meaning the values depend on. Replacing "Dx" with a person surrogate did not
merely lose a word: it removed the evidence that the code beside it was a
diagnosis, and ``K80.20`` was then read as an account number and blocked the
document. One misread label destroyed a clinical fact and produced a false
hold at the same time.

The rule added for this needs both halves of its evidence, and every test
below is paired accordingly: membership in a closed vocabulary of field heads
is not enough, because several of those words are also surnames, and label
position is not enough either, because anyone's name can precede a colon.
"""

from __future__ import annotations

import pytest
from services.clinical_vocabulary import (
    CLINICAL_FIELD_LABELS,
    is_clinical_field_label,
)
from services.detection_evidence import _is_clinical_field_label, assess_finding

# -- the vocabulary ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "word",
    [
        "Dx",
        "dx",
        "DX",
        "Patient",
        "Diagnosis",
        "Record",
        "Allergies",
        "Impression",
        "Findings",
    ],
)
def test_known_field_heads_are_recognised(word):
    assert is_clinical_field_label(word)


@pytest.mark.parametrize(
    "word",
    [
        "Warrenbeck",
        "Oyelowo",
        "Isolde",
        "Vanterpool",
        "Fairweather",
        "Achebe",
        "Lindqvist",
    ],
)
def test_invented_surnames_are_not_field_heads(word):
    assert not is_clinical_field_label(word)


def test_the_vocabulary_holds_no_obvious_given_name():
    """A given name in this set would exempt a real person from detection."""
    for name in ("isolde", "marisol", "lucian", "thandiwe", "nadia", "omar", "sofia"):
        assert name not in CLINICAL_FIELD_LABELS


# -- membership alone is not enough ---------------------------------------------------------


def test_a_field_head_in_label_position_is_a_label():
    text = "Dx: K80.20 - calculus of gallbladder"
    assert _is_clinical_field_label(text, 0, 2)


def test_the_same_word_in_running_prose_is_not_a_label():
    """The privacy counterexample for membership: no colon, no protection."""
    text = "The patient record attended for review"
    start = text.index("record")
    assert not _is_clinical_field_label(text, start, start + len("record"))


def test_a_field_head_without_a_colon_is_not_a_label():
    text = "Diagnosis K80.20 was made"
    assert not _is_clinical_field_label(text, 0, 9)


# -- label position alone is not enough -----------------------------------------------------


def test_a_surname_before_a_colon_is_not_a_label():
    """The privacy counterexample for position: anyone's name can precede a
    colon, so position without vocabulary protects nothing."""
    text = "Warrenbeck: attended the clinic"
    assert not _is_clinical_field_label(text, 0, 10)


def test_a_provider_name_before_a_colon_is_not_a_label():
    text = "Oyelowo: reviewed the film and agreed"
    assert not _is_clinical_field_label(text, 0, 7)


@pytest.mark.parametrize(
    "text,span",
    [
        ("Dr Baker: seen today", "Baker"),
        ("Reviewed by Dr Achebe: agreed", "Achebe"),
        ("Consultant Lindqvist: signed", "Lindqvist"),
    ],
)
def test_a_name_in_a_naming_construction_is_never_a_label(text, span):
    start = text.index(span)
    assert not _is_clinical_field_label(text, start, start + len(span))


# -- what the rule is for --------------------------------------------------------------------


def test_the_label_is_preserved_rather_than_replaced():
    text = "Dx: K80.20 - calculus of gallbladder"
    assessment = assess_finding(text, 0, 2, "PERSON", "gliner_multi_pii")
    assert assessment.action == "preserve"
    assert assessment.rationale == "clinical_field_label"


def test_an_invented_surname_in_the_same_position_is_still_redacted():
    """The whole point, as a counterexample: the rule must not be a way in."""
    text = "Warrenbeck: attended the clinic for review"
    assessment = assess_finding(text, 0, 10, "PERSON", "gliner_multi_pii")
    assert (
        assessment.action != "preserve"
        or assessment.rationale != "clinical_field_label"
    )


def test_keeping_the_label_keeps_the_code_that_depends_on_it():
    """The cascade this exists to stop.

    With the field head intact the ICD code beside it is governed by an
    explicit diagnosis field, which is what protects it. Remove the label and
    that protection has nothing to stand on.
    """
    intact = "Dx: K80.20 - calculus of gallbladder"
    code_start = intact.index("K80.20")
    kept = assess_finding(
        intact, code_start, code_start + 3, "IDENTIFIER", "stanford_deidentifier"
    )
    assert kept.action == "preserve"

    stripped = "<PERSON_2>: K80.20 - calculus of gallbladder"
    lost_start = stripped.index("K80.20")
    lost = assess_finding(
        stripped, lost_start, lost_start + 3, "IDENTIFIER", "stanford_deidentifier"
    )
    # Without the label the code no longer has the evidence that saved it.
    assert (
        lost.action != "preserve" or lost.rationale != "explicit_diagnosis_code_field"
    )


def test_an_identifier_after_an_administrative_label_is_not_protected():
    """ "Record:" is a label, but the value after it is still an identifier."""
    text = "Record: RT4420518"
    value_start = text.index("RT4420518")
    assessment = assess_finding(
        text, value_start, value_start + 9, "IDENTIFIER", "stanford_deidentifier"
    )
    assert assessment.action != "preserve"


# -- a span with no name in it ----------------------------------------------------------


def test_a_label_and_a_code_together_contain_no_person():
    """Preserving "Dx" moved the problem rather than solving it.

    With the label intact the model proposed the wider span "Dx: K80.20" as a
    person. Nothing in it is one: a field head and a diagnosis code, and no
    third thing.
    """
    from services.detection_evidence import _span_is_labels_and_codes

    assert _span_is_labels_and_codes("Dx: K80.20", 0, 10)


@pytest.mark.parametrize(
    "span",
    [
        "Dx: K80.20",
        "Diagnosis: J45.909",
        "Impression: H35.00",
    ],
)
def test_spans_made_only_of_heads_and_diagnosis_codes_are_preserved(span):
    from services.detection_evidence import _span_is_labels_and_codes

    assert _span_is_labels_and_codes(span, 0, len(span))


@pytest.mark.parametrize(
    "span",
    [
        "Dr Baker: K80.20",
        "Isolde: 12",
        "Patient: Isolde Warrenbeck",
        "Reviewed by: Oyelowo",
        "Warrenbeck",
        "Vanterpool: K80.20",
        # The first version of this rule accounted for "any token with a digit"
        # and would have passed both of these. They are a label and a record
        # number.
        "Record: RT4420518",
        "Account: ACR520563",
        "Record: 12345",
    ],
)
def test_one_unexplained_word_is_enough_to_decline(span):
    """The counterexample set. The test is exhaustive on purpose: a single
    token that is not a head, a code or a number means the span might name
    somebody, and the rule declines rather than guessing."""
    from services.detection_evidence import _span_is_labels_and_codes

    assert not _span_is_labels_and_codes(span, 0, len(span))


def test_the_assessment_preserves_a_label_and_code_span():
    text = "Dx: K80.20 - calculus of gallbladder"
    assessment = assess_finding(text, 0, 10, "PERSON", "gliner_multi_pii")
    assert assessment.action == "preserve"
    assert assessment.rationale == "span_contains_no_name_token"


def test_the_assessment_still_redacts_a_name_beside_a_code():
    text = "Vanterpool: K80.20 noted"
    assessment = assess_finding(text, 0, 18, "PERSON", "gliner_multi_pii")
    assert assessment.rationale != "span_contains_no_name_token"


# -- a label does not reach across a line break -------------------------------------------

NL = chr(10)


def test_a_spent_label_does_not_claim_the_next_field():
    """The last step of the cascade.

    With "Dx" preserved, the model proposed "Dx: K80.20" as a person, and the
    naming-context guard let it through because "Patient:" sat 10 characters
    earlier - on the line above, with its own value already removed. A field
    label introduces the value on its own line; it does not reach past the
    break to claim the next field.
    """
    from services.detection_evidence import _context_says_person

    text = "Patient:" + NL + "Dx: K80.20 - calculus of gallbladder"
    start = text.index("Dx: K80.20")
    assert not _context_says_person(text, start, start + len("Dx: K80.20"))


def test_a_name_on_the_line_below_its_label_keeps_that_context():
    """The counterexample. Names wrap onto their own line all the time, and
    trimming at every break would strip exactly the evidence that catches
    them. Only a span that is itself a field head trims."""
    from services.detection_evidence import _context_says_person

    text = "Patient:" + NL + "Isolde Warrenbeck attended"
    start = text.index("Isolde")
    assert _context_says_person(text, start, start + len("Isolde Warrenbeck"))


def test_a_name_on_the_same_line_as_its_label_keeps_that_context():
    from services.detection_evidence import _context_says_person

    text = "Patient: Isolde Warrenbeck"
    start = text.index("Isolde")
    assert _context_says_person(text, start, start + len("Isolde Warrenbeck"))


@pytest.mark.parametrize(
    "span,expected",
    [
        ("Dx: K80.20", True),
        ("Diagnosis: J45.909", True),
        ("Dr Achebe", False),
        ("Isolde Warrenbeck", False),
        ("Dx K80.20", False),  # no colon: not a field head
    ],
)
def test_only_a_real_field_head_trims_its_context(span, expected):
    from services.detection_evidence import _starts_a_new_labelled_field

    text = "Patient:" + NL + span
    start = text.index(span)
    assert _starts_a_new_labelled_field(text, start, start + len(span)) is expected


def test_the_whole_cascade_now_preserves_the_diagnosis_line():
    text = "Patient:" + NL + "Dx: K80.20 - calculus of gallbladder"
    start = text.index("Dx: K80.20")
    assessment = assess_finding(
        text, start, start + len("Dx: K80.20"), "PERSON", "gliner_multi_pii"
    )
    assert assessment.action == "preserve"


def test_a_patient_name_under_a_spent_label_is_still_redacted():
    """The privacy counterexample for the whole cascade."""
    text = "Record:" + NL + "Isolde Warrenbeck attended the clinic"
    start = text.index("Isolde")
    assessment = assess_finding(
        text, start, start + len("Isolde Warrenbeck"), "PERSON", "gliner_multi_pii"
    )
    assert assessment.action != "preserve"
