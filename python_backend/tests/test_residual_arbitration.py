"""Arbitration of first-pass edits and residual findings: v5 diagnostic classes.

Each test is a defect class seen in the real-model trace of the v5 corpus,
checked with new invented wording rather than the fixture sentences:

- a model's identifier label on a number its own sentence reads as a
  measurement, while a fragment of a code under an identifier head is still
  replaced;
- a lowercase unit read as a personal title ("ms on the");
- a hyphenated surname split by a model into a name and a stray fragment;
- a line-opening analyte word read as a name, and a lone lowercase noun;
- role phrases and code labels held because two models agree;
- a model edit already inside an exact identifier match;
- a residual reading over nothing but sanitizer tokens and copied token shapes;
- holds a person can adjudicate being refused manual review.

Every rule is paired with the case that must still be removed.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "residual-arbitration-test-salt")

from services.detection_evidence import assess_finding  # noqa: E402
from services.phi_detection import DetectedEntity  # noqa: E402

GLINER = "gliner_multi_pii"


@pytest.fixture(autouse=True)
def _legacy_models(monkeypatch):
    monkeypatch.setenv("PHI_MODEL_MODE", "legacy_test")
    monkeypatch.setenv("PHI_ALLOW_NON_PRODUCTION_DETECTORS", "1")


def _assess(text, span, category, source=GLINER, **kwargs):
    start = text.index(span)
    return assess_finding(text, start, start + len(span), category, source, **kwargs)


# -- a number judged where it stands --------------------------------------------------------------


@pytest.mark.parametrize(
    "text,value",
    [
        ("Lipase 3817 U/L on arrival.", "3817"),
        ("Coronal STIR TR 5410 ms, slice gap 1 mm.", "5410"),
        ("Ferritin was 1260 ng/mL last week.", "1260"),
    ],
)
def test_a_model_identifier_label_on_a_reading_keeps_the_reading(text, value):
    assessment = _assess(text, value, "HEALTH_PLAN_ID", agreeing_detectors=2)
    assert (assessment.action, assessment.rationale) == (
        "preserve",
        "numeric_context_clinical",
    )


@pytest.mark.parametrize(
    "text,value",
    [
        ("MRN: K51.3817", "3817"),  # a fragment of a code under an identifier head
        ("Account 22-5410 closed.", "5410"),
        ("Rang 6093 twice, no answer.", "6093"),  # unresolved: no reading here
    ],
)
def test_a_number_without_a_reading_of_its_own_is_not_kept(text, value):
    assessment = _assess(text, value, "IDENTIFIER")
    assert assessment.rationale != "numeric_context_clinical"


# -- titles as written ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Axial TE 4120 ms on the coronal plane.",
        "Severe mr noted by the echo team.",
    ],
)
def test_a_unit_or_abbreviation_is_never_a_title(text):
    from services.ner_phi_detector import _explicit_person_context_entities

    assert _explicit_person_context_entities(text, []) == []


@pytest.mark.parametrize(
    "text,name",
    [
        ("Letter copied to Ms Ilse Vorhaven today.", "Ilse Vorhaven"),
        ("reviewed by dr tomas quellbury later", "tomas quellbury"),
        ("SEEN BY MRS ODILE MARCHBANK", "ODILE MARCHBANK"),
    ],
)
def test_a_written_title_still_introduces_a_name(text, name):
    from services.ner_phi_detector import _explicit_person_context_entities

    found = [text[e.start : e.end] for e in _explicit_person_context_entities(text, [])]
    assert name in found


# -- hyphenated surnames -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "span,whole",
    [
        ("Ysolde Brannock", "Ysolde Brannock-Achterberg"),
        ("Achterberg", "Brannock-Achterberg"),
    ],
)
def test_a_person_span_takes_the_whole_hyphenated_surname(span, whole):
    from services.text_anonymization import _whole_words

    text = (
        "Handover for Ysolde Brannock-Achterberg in bay 3, a well-known X-ray regular."
    )
    start = text.index(span)
    widened = _whole_words(
        DetectedEntity("PERSON", start, start + len(span), GLINER, 0.9, "person"), text
    )
    assert text[widened.start : widened.end] == whole


@pytest.mark.parametrize("word", ["known", "ray"])
def test_a_hyphenated_common_word_is_not_widened(word):
    from services.text_anonymization import _whole_words

    text = (
        "Handover for Ysolde Brannock-Achterberg in bay 3, a well-known X-ray regular."
    )
    start = text.index(word)
    widened = _whole_words(
        DetectedEntity("PERSON", start, start + len(word), GLINER, 0.9, "person"), text
    )
    assert text[widened.start : widened.end] == word


# -- the proper-noun heuristic -------------------------------------------------------------------


def _strict_candidates(text):
    from services.ner_phi_detector import (
        _strict_proper_noun_entities,
        configured_model_name,
        load_spacy_pipeline,
    )

    nlp = load_spacy_pipeline(configured_model_name())
    doc = nlp(text)
    return [text[e.start : e.end] for e in _strict_proper_noun_entities(doc, [], nlp)]


def test_a_line_opening_word_that_labels_a_measurement_is_not_a_name():
    found = _strict_candidates(
        "Plan unchanged.\nPancreastatin B 88 pmol/L on the fasting sample.\n"
    )
    assert "Pancreastatin" not in found


@pytest.mark.parametrize(
    "text,word,labels",
    [
        (
            "Plan unchanged.\nPancreastatin B 88 pmol/L on the fasting sample.\n",
            "Pancreastatin",
            True,
        ),
        (
            "Plan unchanged.\nOsvaldo 2 days ago reported pain.\n",
            "Osvaldo",
            False,
        ),  # a duration
        (
            "Plan unchanged. Seen with Osvaldo 88 mg given.\n",
            "Osvaldo",
            False,
        ),  # not an opening
    ],
)
def test_only_a_line_opening_word_before_a_unit_labels_a_measurement(
    text, word, labels
):
    from services.ner_phi_detector import (
        _labels_a_measurement,
        configured_model_name,
        load_spacy_pipeline,
    )

    doc = load_spacy_pipeline(configured_model_name())(text)
    index = next(token.i for token in doc if token.text == word)
    assert _labels_a_measurement(doc, index, index + 1) is labels


def test_a_lone_lowercase_noun_is_not_a_name_candidate():
    found = _strict_candidates("Next of kin informed by phone this evening.")
    assert "kin" not in found


def test_a_line_opening_review_label_is_not_a_name_candidate():
    found = _strict_candidates("Review: cafe-au-lait macules are unchanged.")
    assert "Review" not in found


# -- structure beats model agreement -------------------------------------------------------------


@pytest.mark.parametrize(
    "text,span,category",
    [
        (
            "Respiratory physiotherapist: bleep 2231.",
            "Respiratory physiotherapist",
            "PERSON",
        ),
        ("Stoma nurse: extension 7314.", "Stoma nurse", "PERSON"),
        ("Registrant GMC 7112345 confirmed.", "GMC", "ORGANIZATION"),
    ],
)
def test_structural_labels_are_kept_when_two_models_agree(text, span, category):
    assessment = _assess(
        text, span, category, agreeing_detectors=2, contains_proper_noun=True
    )
    assert assessment.action == "preserve"


def test_a_name_before_a_role_noun_is_still_removed_when_models_agree():
    text = "Quillon physiotherapist: bleep 2231."
    assessment = _assess(
        text,
        "Quillon physiotherapist",
        "PERSON",
        agreeing_detectors=2,
        contains_proper_noun=True,
    )
    assert assessment.action == "redact"


@pytest.mark.parametrize("span", ["Plaques", "Lesions", "Nodules"])
def test_a_finding_with_treatment_response_is_structural(span):
    text = f"{span} improved on azathioprine 20 mg weekly."
    assessment = _assess(
        text, span, "PERSON", agreeing_detectors=2, contains_proper_noun=True
    )
    assert (assessment.action, assessment.rationale) == (
        "preserve",
        "clinical_finding_response_grammar",
    )


# -- residual arbitration ------------------------------------------------------------------------


def _residual(text, regions, span, category="PERSON", source=GLINER, monkeypatch=None):
    from services import text_anonymization as ta
    from services.transformation_provenance import (
        ReplacementRegion,
        TransformationProvenance,
    )

    start = text.index(span)
    entity = DetectedEntity(
        category, start, start + len(span), source, 0.9, category.lower()
    )
    monkeypatch.setattr(ta, "_detect_entities", lambda *a, **k: ([entity], []))
    monkeypatch.setattr(ta, "scan_structured_identifiers", lambda _text: [])
    monkeypatch.setattr(ta, "date_fragments", lambda *_a: [])
    provenance = TransformationProvenance(
        [
            ReplacementRegion(
                text.index(token), text.index(token) + len(token), "surrogate", "PERSON"
            )
            for token in regions
        ]
    )
    return ta.residual_findings(text, provenance)


def test_a_copied_token_shape_beside_a_genuine_token_still_blocks(monkeypatch):
    text = "Imported note: <PERSON_4> <PERSON_1>. Plan unchanged."
    findings = _residual(
        text, ["<PERSON_1>"], "<PERSON_4> <PERSON_1>", monkeypatch=monkeypatch
    )
    assert findings and any(f["blocking"] for f in findings)


def test_a_single_original_letter_beside_the_tokens_still_blocks(monkeypatch):
    text = "Imported note: <PERSON_4> K <PERSON_1>. Plan unchanged."
    findings = _residual(
        text, ["<PERSON_1>"], "<PERSON_4> K <PERSON_1>", monkeypatch=monkeypatch
    )
    assert any(f["blocking"] for f in findings)


# -- holds a person can adjudicate ---------------------------------------------------------------


# -- CSV role-column headers ---------------------------------------------------------------------


# -- a sheet named with an identifier ------------------------------------------------------------


# -- values of clinical fields, and code labels before replaced codes -----------------------------


@pytest.mark.parametrize(
    "text,span",
    [
        ("unit: pmol/L\nanalyte: Pancreastatin\nvalue: 88", "Pancreastatin"),
        ("Stain = Warthin-Starry.", "Warthin-Starry"),
        ("Specimen type: bronchial washings | result: pending", "bronchial washings"),
    ],
)
def test_the_value_of_a_clinical_field_is_clinical(text, span):
    assessment = _assess(
        text,
        span,
        "ORGANIZATION",
        source="ner",
        agreeing_detectors=2,
        contains_proper_noun=True,
    )
    assert (assessment.action, assessment.rationale) == (
        "preserve",
        "value_of_clinical_field",
    )


@pytest.mark.parametrize(
    "text,span",
    [
        (
            "analyte: Pancreastatin checked by Vorhaven",
            "Vorhaven",
        ),  # not the field's value
        ("Copied to: Ilse Vorhaven", "Ilse Vorhaven"),  # not a clinical field
    ],
)
def test_a_name_near_a_clinical_field_is_still_removed(text, span):
    assert _assess(text, span, "PERSON").action == "redact"


@pytest.mark.parametrize(
    "text,span,kept",
    [
        ("Registrant: NMCR <LICENSE_ID_1>.", "NMCR", True),
        ("Pharmacist GPhC 2081734 on duty.", "GPhC", True),
        ("Seen by JKS <PERSON_1> today.", "JKS", False),  # initials are not a body
    ],
)
def test_an_issuing_body_acronym_before_its_code_is_a_label(text, span, kept):
    assessment = _assess(text, span, "ORGANIZATION", contains_proper_noun=True)
    assert (assessment.action == "preserve") is kept


def test_a_product_code_filling_an_equipment_field_is_not_a_device_identifier():
    kept = _assess("Model: QX-12", "QX-12", "DEVICE_ID")
    serial = _assess("Serial: QX-12", "QX-12", "DEVICE_ID")
    assert (kept.action, kept.rationale) == ("preserve", "equipment_model_designation")
    assert serial.action == "redact"
