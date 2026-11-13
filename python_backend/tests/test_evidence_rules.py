"""What counts as evidence, pinned by the failures that changed the rules.

Each test below corresponds to an attributed cause in the Phase 16 diagnosis
of the spent Phase 15 corpus, where five rules between them destroyed 1,164
clinical tokens across 216 adversarial and 72 identifier-free notes, held 32
clean notes, and released a biometric identifier twice:

* a "medical condition" prediction was counted as a second detector agreeing
  that the span was a name, so the detector arguing for preservation supplied
  the corroboration for removal;
* a category a name is not confused with decided on its own, with no clinical
  reading consulted, and spaCy's DATE_TIME over "HbA1c 58" removed the result;
* an analyte had to be in a list to be recognised, so every laboratory result
  nobody listed was collateral;
* the denominator of a compound unit was proposed as a name 250 times; and
* a code written inside a span a model called clinical was shielded by that
  reading.

The pairs matter as much as the single cases: each rule has to keep letting
through the thing it was written to catch.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.detection_evidence import (  # noqa: E402
    EVIDENCE_HEURISTIC,
    EVIDENCE_MODEL,
    assess_finding,
    clinical_reading_strength,
    count_agreeing,
    identifier_token_spans,
)
from services.phi_detection import (  # noqa: E402
    SOURCE_NER,
    SOURCE_STRICT_PROPER_NOUN,
    SOURCE_STRUCTURED_PATTERN,
)

GLINER = "gliner_multi_pii"


class _Span:
    def __init__(self, start, end, entity_type, source):
        self.start = start
        self.end = end
        self.entity_type = entity_type
        self.source = source


def _at(text, term, category, source, **kwargs):
    start = text.index(term)
    return assess_finding(text, start, start + len(term), category, source, **kwargs)


# -- agreement means a compatible identifier claim --------------------------


def test_a_condition_label_is_not_a_detector_agreeing_it_is_a_name():
    """GLiNER is asked for "medical condition" so a clinical span gets a
    clinical label. Counting that label as agreement with PERSON turned the
    one prediction arguing for preservation into the second vote for removal,
    and "Haemoglobin 118 g/L" went to manual review on the strength of it."""
    entities = [
        _Span(0, 19, "PERSON", SOURCE_NER),
        _Span(0, 19, "MEDICAL_CONDITION", GLINER),
    ]

    assert count_agreeing(entities, 0, 19, "PERSON", exclude_source=SOURCE_NER) == 1


def test_two_identifier_claims_still_agree():
    entities = [
        _Span(0, 9, "PERSON", SOURCE_NER),
        _Span(0, 9, "FACILITY", GLINER),
    ]

    assert count_agreeing(entities, 0, 9, "PERSON", exclude_source=SOURCE_NER) == 2


def test_a_clinical_reading_survives_a_lone_condition_label():
    text = "Haemoglobin 118 g/L and ferritin 14 ug/L confirm iron deficiency."

    assessment = _at(
        text,
        "Haemoglobin 118 g/L",
        "PERSON",
        SOURCE_NER,
        agreeing_detectors=1,
        clinical_support=True,
    )

    assert assessment.action == "preserve"


def test_naming_context_still_wins_over_a_condition_label():
    """The fix must not shield a name. A title in front of the span returns an
    identifier reading before any of the corroboration rules are reached."""
    text = "Discussed with Dr Ravelstoke before discharge."

    assessment = _at(
        text,
        "Ravelstoke",
        "PERSON",
        SOURCE_NER,
        agreeing_detectors=1,
        clinical_support=True,
    )

    assert assessment.action == "redact"


# -- a clinical reading applies to every model claim ------------------------


def test_a_models_date_claim_over_a_prescription_keeps_the_prescription():
    text = "eGFR 52 mL/min/1.73m2 is stable; continue ramipril 5 mg once daily."

    assessment = _at(text, "ramipril 5", "DATE_TIME", SOURCE_NER)

    assert assessment.action == "preserve"
    assert assessment.reading == "clinical"


def test_a_models_date_claim_over_a_real_date_still_removes_it():
    """The exemption is for spans with no calendar content in them. A month
    name or a calendar figure is a date element whatever else the span reads
    as, so a model span that swallows a drug name and a date loses the date."""
    text = "Started ramipril 5 March 2024 after review."

    assessment = _at(text, "ramipril 5 March 2024", "DATE_TIME", SOURCE_NER)

    assert assessment.action == "redact"


@pytest.mark.parametrize("term", ["12/03/2024", "2024-01-05", "5 Jan"])
def test_calendar_content_is_never_exempted(term):
    text = f"Sample taken on {term} in clinic."

    assessment = _at(text, term, "DATE", SOURCE_NER)

    assert assessment.action == "redact"


# -- an analyte does not have to be in a list -------------------------------


@pytest.mark.parametrize(
    "sentence,term",
    [
        ("Haemoglobin 118 g/L confirms iron deficiency.", "Haemoglobin 118 g/L"),
        ("Repeat ferritin 14 ug/L was requested.", "ferritin 14 ug/L"),
        ("HbA1c 58 mmol/mol is unchanged.", "HbA1c 58 mmol/mol"),
        ("Serum tryptasine 8.2 nmol/L was normal.", "tryptasine 8.2 nmol/L"),
    ],
)
def test_a_word_a_value_and_a_unit_reads_as_a_measurement(sentence, term):
    """The last one names nothing real. That is the point: a closed vocabulary
    protects the analytes somebody thought of and loses every other result."""
    start = sentence.index(term)

    reading, _rationale = clinical_reading_strength(sentence, start, start + len(term))

    assert reading == "clinical"


def test_a_span_that_stops_before_its_unit_is_still_a_measurement():
    """A model does not draw its span where a reader would. spaCy proposed
    "Haemoglobin 118" without the "g/L" that makes it a result, in 106 of the
    288 documents, so the unit is looked for a little past the span's end."""
    sentence = "Haemoglobin 118 g/L confirms iron deficiency."

    reading, _rationale = clinical_reading_strength(sentence, 0, len("Haemoglobin 118"))

    assert reading == "clinical"


def test_a_measurement_does_not_shield_a_name_that_follows_it():
    """The measurement has to account for the whole span, not merely open it.
    A rule that read only the beginning would release the name after it."""
    sentence = "Haemoglobin 118 g/L and Orsolyne Marchbanke reviewed it."
    term = "Haemoglobin 118 g/L and Orsolyne Marchbanke"

    reading, _rationale = clinical_reading_strength(sentence, 0, len(term))

    assert reading != "clinical"


def test_a_surname_with_a_dose_after_it_is_not_a_measurement():
    """The leading word has to be one a surname is unlikely to be mistaken
    for: lower case, carrying a digit, or opening a sentence."""
    sentence = "Reviewed by Saltonstall 12 mg later that day."
    start = sentence.index("Saltonstall 12 mg")

    reading, _rationale = clinical_reading_strength(sentence, start, start + 17)

    assert reading != "clinical"


# -- a unit fragment is not a name ------------------------------------------


@pytest.mark.parametrize(
    "sentence,term",
    [
        ("HbA1c 58 mmol/mol is unchanged.", "mol"),
        ("Peak flow improved to 420 L/min today.", "min"),
        ("eGFR 52 mL/min/1.73m2 is stable.", "min"),
    ],
)
def test_the_denominator_of_a_unit_is_preserved(sentence, term):
    start = sentence.index("/" + term) + 1

    assessment = assess_finding(
        sentence,
        start,
        start + len(term),
        "PERSON",
        SOURCE_STRICT_PROPER_NOUN,
    )

    assert assessment.action == "preserve"
    assert assessment.rationale == "unit_expression_fragment"


def test_a_name_after_a_solidus_is_not_a_unit():
    sentence = "Referred by Marchbanke/Nettlecombe for review."
    start = sentence.index("Nettlecombe")

    assessment = assess_finding(
        sentence,
        start,
        start + len("Nettlecombe"),
        "PERSON",
        SOURCE_NER,
    )

    assert assessment.action != "preserve"


# -- a condition label does not shield a code -------------------------------


def test_a_code_inside_a_clinical_span_is_found():
    text = "Enrolled with palm-vein template PVT123456 on file."

    spans = identifier_token_spans(
        text, text.index("palm-vein"), text.index(" on file")
    )

    assert [text[start:end] for start, end in spans] == ["PVT123456"]


@pytest.mark.parametrize(
    "category", ["IDENTIFIER", "MEDICAL_RECORD_NUMBER", "DATE_TIME"]
)
def test_a_model_diagnosis_code_in_an_explicit_field_is_preserved(category):
    text = "Assessment\nDiagnosis: J45.909. Asthma remains controlled."
    start = text.index("J45.909")

    assessment = assess_finding(
        text,
        start,
        start + len("J45.909"),
        category,
        GLINER,
    )

    assert assessment.action == "preserve"
    assert assessment.rationale == "explicit_diagnosis_code_field"


@pytest.mark.parametrize("label", ["MRN", "Account", "Patient ID", "Reference"])
def test_an_icd_shaped_value_outside_a_diagnosis_field_is_not_exempt(label):
    text = f"{label}: J45.909"
    start = text.index("J45.909")

    assessment = assess_finding(
        text,
        start,
        start + len("J45.909"),
        "IDENTIFIER",
        GLINER,
    )

    assert assessment.action != "preserve"


def test_a_deterministic_identifier_match_still_wins_in_a_diagnosis_field():
    text = "Diagnosis: J45.909"
    start = text.index("J45.909")

    assessment = assess_finding(
        text,
        start,
        start + len("J45.909"),
        "MEDICAL_RECORD_NUMBER",
        SOURCE_STRUCTURED_PATTERN,
    )

    assert assessment.action == "redact"


@pytest.mark.parametrize(
    "text,term",
    [
        ("Diagnosis: J45 unspecified.", "J45"),
        ("Dx: j45.909 reviewed.", "j45.909"),
        ("Diagnosis: J45.909 (asthma, unspecified).", "J45.909"),
        ("Primary Diagnosis: J45.909", "J45.909"),
        ("Discharge Diagnoses: J45.909", "J45.909"),
        ("ICD-10: J45.909 asthma", "J45.909"),
        ("Assessment\nDiagnosis: J45.909\nPlan: continue inhaler.", "J45.909"),
        ("Diagnosis: J45.909, stable.", "J45.909"),
    ],
)
def test_a_governed_diagnosis_code_is_preserved_however_it_is_written(text, term):
    """A code with no decimal, in lower case, beside its description, under a
    qualified or plural label, in a labelled section, or followed by sentence
    punctuation is the same code each time."""
    start = text.index(term)

    assessment = assess_finding(text, start, start + len(term), "IDENTIFIER", GLINER)

    assert assessment.action == "preserve"
    assert assessment.rationale == "explicit_diagnosis_code_field"


def test_the_same_code_twice_in_one_note_is_preserved_both_times():
    text = "Diagnosis: J45.909 and Dx: J45.909 again."

    for start in (text.index("J45.909"), text.rindex("J45.909")):
        assessment = assess_finding(
            text,
            start,
            start + len("J45.909"),
            "IDENTIFIER",
            GLINER,
        )

        assert assessment.action == "preserve"


@pytest.mark.parametrize(
    "text,term",
    [
        ("Diagnosis: J45.909X7 extended.", "J45.909"),
        ("Diagnosis: J45.909-22 on file.", "J45.909"),
    ],
)
def test_an_icd_shaped_prefix_of_a_longer_token_is_not_a_diagnosis_code(text, term):
    """The exemption is judged on the whole token. A model that proposes only
    the ICD-looking head of an account number gets no protection for it,
    because the token it sits in is not a code."""
    start = text.index(term)

    assessment = assess_finding(text, start, start + len(term), "IDENTIFIER", GLINER)

    assert assessment.action == "redact"


@pytest.mark.parametrize("term", [".909", "909"])
def test_a_model_span_inside_a_diagnosis_code_keeps_the_whole_code(term):
    """Stanford's deidentifier called ".909" an identifier inside "G43.909",
    which left "G43" joined to a surrogate and destroyed the diagnosis: the
    span it offered was a fragment, and a rule that judged the fragment could
    only decline to protect it. The enclosing token is what is read."""
    text = "Diagnosis: G43.909. Impression consistent with migraine."
    start = text.index(term)

    assessment = assess_finding(text, start, start + len(term), "IDENTIFIER", GLINER)

    assert assessment.action == "preserve"
    assert assessment.rationale == "explicit_diagnosis_code_field"


@pytest.mark.parametrize("term", [".909", "909"])
def test_a_fragment_under_an_identifier_head_is_still_removed(term):
    """The enclosing token is read, not trusted: under MRN it is an MRN."""
    text = f"MRN: G43{term if term.startswith('.') else '.' + term}"
    start = text.rindex(term)

    assessment = assess_finding(text, start, start + len(term), "IDENTIFIER", GLINER)

    assert assessment.action == "redact"


def test_a_four_character_fraction_is_a_valid_code_shape():
    """ICD-10-CM allows three characters, a point and up to four more, so
    "J45.9091" is a code and not evidence of a longer identifier."""
    text = "Diagnosis: J45.9091 recorded."
    start = text.index("J45.9091")

    assessment = assess_finding(
        text, start, start + len("J45.9091"), "IDENTIFIER", GLINER
    )

    assert assessment.action == "preserve"


@pytest.mark.parametrize(
    "text,term",
    [
        ("Diagnosis: A12.34567", "A12.34567"),
        ("Diagnosis: AB.1234", "AB.1234"),
        ("Diagnosis: 1145.909", "1145.909"),
        ("Diagnosis: JJ5.909", "JJ5.909"),
        ("The diagnosis was confirmed. J45.909", "J45.909"),
    ],
)
def test_a_value_that_is_not_an_icd_code_gets_no_diagnosis_exemption(text, term):
    """A too-long fraction, an arbitrary alphanumeric with a decimal in it, a
    leading digit and a second leading letter all fail the code shape; the
    last case fails the field test instead, because a sentence mentioning the
    word is not a field governing the value."""
    start = text.index(term)

    assessment = assess_finding(text, start, start + len(term), "IDENTIFIER", GLINER)

    assert assessment.action == "redact"


@pytest.mark.parametrize(
    "text",
    [
        "Ulcerative colitis J45.909 is in remission.",
        "HbA1c 58 mmol/mol and eGFR 52 mL/min/1.73m2 are stable.",
        "Blood pressure 132/84 was recorded twice.",
        "Reviewed on 12/03/2024 in clinic.",
    ],
)
def test_clinical_content_carries_no_identifier_syntax(text):
    """The syntax test is a letter and at least four digits. A diagnosis code,
    a reading, a ratio and a date all fail it, so nothing clinical is removed
    by this rule."""
    assert identifier_token_spans(text, 0, len(text)) == []


def test_a_code_a_model_called_a_condition_is_removed_and_the_words_are_kept():
    """The span itself still goes through the evidence model, which preserves
    it as the clinical reading it is; only the code is taken out of it.

    Driven from a constructed condition prediction rather than from a model
    run: which detector applies which label is the models' business, and this
    is about what the pipeline does once one of them says "condition"."""
    from services.phi_detection import DetectedEntity
    from services.text_anonymization import _apply_evidence_model

    text = "Enrolled with palm-vein template PVT123456 on file."
    span = DetectedEntity(
        entity_type="MEDICAL_CONDITION",
        start=text.index("palm-vein"),
        end=text.index(" on file"),
        source=GLINER,
        score=0.9,
    )

    kept, review = _apply_evidence_model([span], text)

    assert review == []
    assert [(text[e.start : e.end], e.entity_type) for e in kept] == [
        ("PVT123456", "OTHER_UNIQUE_ID")
    ]


# -- the heuristic still catches what it exists for -------------------------


def test_an_unexplained_name_shaped_span_is_still_replaced():
    """Every rule above widens what is kept. None of them may reopen the
    fail-closed path the proper-noun fallback exists for."""
    text = "The referral says 'Quenbridge' in full."
    start = text.index("Quenbridge")

    assessment = assess_finding(
        text,
        start,
        start + len("Quenbridge"),
        "PERSON",
        SOURCE_STRICT_PROPER_NOUN,
    )

    assert assessment.evidence == EVIDENCE_HEURISTIC
    assert assessment.action == "redact"


def test_a_model_name_with_no_clinical_reading_is_still_replaced():
    text = "The referral says 'Thornwaithe' in full."
    start = text.index("Thornwaithe")

    assessment = assess_finding(
        text,
        start,
        start + len("Thornwaithe"),
        "PERSON",
        SOURCE_NER,
    )

    assert assessment.evidence == EVIDENCE_MODEL
    assert assessment.action == "redact"


# -- a unit named after a person is still a unit ----------------------------


@pytest.mark.parametrize(
    "sentence,term",
    [
        ("A temperature of 38.2 degrees Celsius was obtained.", "Celsius"),
        ("Recorded 99.1 degrees Fahrenheit on arrival.", "Fahrenheit"),
        ("Visual acuity 12 Snellen metres recorded.", "Snellen"),
        ("Bilirubin 28 umol/L today.", "umol"),
        ("Dose 5 Gray delivered to the field.", "Gray"),
        ("Attenuation 45 Hounsfield units.", "Hounsfield"),
        ("Drain size 14 French inserted.", "French"),
    ],
)
def test_a_unit_named_after_its_deviser_is_preserved(sentence, term):
    """Two losses in one family. "38.2 degrees Celsius" came back as "38.2
    degrees <PERSON_2>", and "12 Snellen metres" lost the scale it was
    measured on. Anders Celsius and Herman Snellen were people, so the shape
    is genuinely a name; what settles it is the company the word keeps."""
    start = sentence.index(term)

    assessment = assess_finding(
        sentence,
        start,
        start + len(term),
        "PERSON",
        SOURCE_STRICT_PROPER_NOUN,
    )

    assert assessment.action == "preserve"
    assert assessment.rationale == "measurement_unit_in_use"


@pytest.mark.parametrize(
    "sentence,term",
    [
        ("Reviewed by Celsius in clinic.", "Celsius"),
        ("Referred to Dr Snellen today.", "Snellen"),
        ("Patient: Kelvin Marchbanke", "Kelvin"),
        ("Contact Mr French about the referral.", "French"),
        ("The referral says 'Celsius' in full.", "Celsius"),
        ("Signed Snellen.", "Snellen"),
        ("Ward list: Kelvin, Gray, French", "Gray"),
    ],
)
def test_the_same_word_as_a_name_is_still_removed(sentence, term):
    """Membership in the lexicon protects nothing on its own. Without a number,
    a unit word or a solidus beside it, the word is read as what it looks
    like, and a patient called Kelvin keeps their privacy."""
    start = sentence.index(term)

    assessment = assess_finding(
        sentence,
        start,
        start + len(term),
        "PERSON",
        SOURCE_STRICT_PROPER_NOUN,
    )

    assert assessment.action != "preserve"


@pytest.mark.parametrize(
    "sentence,term",
    [
        ("Account ACR520563 closed.", "ACR520563"),
        ("Record RT4420518 filed.", "RT4420518"),
        ("Initials K.M. noted.", "K.M."),
    ],
)
def test_an_identifier_beside_a_number_is_not_a_unit(sentence, term):
    """The rule reads a lexicon, not a shape, so nothing becomes a unit by
    standing near a digit."""
    start = sentence.index(term)

    assessment = assess_finding(
        sentence,
        start,
        start + len(term),
        "PERSON",
        SOURCE_STRICT_PROPER_NOUN,
    )

    assert assessment.action != "preserve"
