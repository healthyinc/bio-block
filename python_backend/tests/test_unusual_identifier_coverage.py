"""Unusual structured identifiers: delimited codes a model calls a phone number.

A seven-digit minimum stops a model's phone-number claim from removing a
blood pressure, and it must stay. It also stopped removing a six-digit
delimited reference code a model labelled a phone number: with no phone
authority left, the span reached an exemption written for names, and a
tagger that saw no proper noun in it kept it in the output. Two failures
followed from one gap - no grammar recognised the code as an identifier in
its own right, and a structured claim that failed its syntax check fell
through to a rule meant for words.

The fix recognises such codes as OTHER_UNIQUE_ID from their own grammar -
an announcing word, or distinctive delimiters - and a structured claim
without its syntax now fails closed. Every value here is invented; none is
drawn from a corpus.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "unusual-identifier-test-salt")

from services.detection_evidence import assess_finding  # noqa: E402
from services.phi_detection import (  # noqa: E402
    DetectedEntity,
    StructuredPatternDetector,
    expand_partial_identifiers,
    propagate_repeated_identifiers,
)
from services.residual_identifier_scan import scan as residual_scan  # noqa: E402

UNIQUE = "OTHER_UNIQUE_ID"


@pytest.fixture(autouse=True)
def _legacy_models(monkeypatch):
    monkeypatch.setenv("PHI_MODEL_MODE", "legacy_test")


def _found(text, category=UNIQUE):
    return [
        text[e.start : e.end]
        for e in StructuredPatternDetector().detect(text)
        if e.entity_type == category
    ]


def _anonymize(text):
    from services import text_anonymization as ta

    ta._build_detectors.cache_clear()
    return ta.anonymize_clinical_text(text, profile="strict")


class _FixedSpanModel:
    """A stand-in model proposing one fixed span wherever it appears."""

    def __init__(self, source, category, span):
        self.source, self.category, self.span = source, category, span

    def detect(self, value):
        if self.span not in value:
            return []
        start = value.index(self.span)
        return [
            DetectedEntity(
                entity_type=self.category,
                start=start,
                end=start + len(self.span),
                source=self.source,
                score=0.5,
            )
        ]


def _anonymize_with(monkeypatch, text, model):
    from services import text_anonymization as ta

    monkeypatch.setattr(
        ta, "_detectors", lambda *_args, **_kwargs: (StructuredPatternDetector(), model)
    )
    return ta.anonymize_clinical_text(text, profile="strict")


def _release(outcome):
    from services.ingestion import release_decision_for
    from services.utility_contract import contract_for

    contract = contract_for("text")
    handler = {
        "anonymized_text": outcome["anonymized_text"],
        "residual_phi_categories": outcome["residual_phi_categories"],
        "utility_metrics": outcome.get("utility_metrics", {}),
        "utility_verdict": (
            contract.evaluate(outcome.get("utility_metrics", {})).to_public_dict()
            if contract
            else None
        ),
        "review_required_reasons": outcome.get("review_required_reasons", []),
        "anonymization_status": outcome["anonymization_status"],
    }
    return release_decision_for("text", handler, "unusual-identifier-test")


# -- the two regression shapes, invented -------------------------------------------


@pytest.mark.parametrize(
    "text,value",
    [
        ("Local reference K42;;7315;;QD on the referral.", "K42;;7315;;QD"),
        ("Legacy study code HT::5820-J4::MV was archived.", "HT::5820-J4::MV"),
        ("Filed as R07//4418-B2//XN in the old system.", "R07//4418-B2//XN"),
    ],
)
def test_a_delimited_unusual_identifier_is_a_unique_identifier(text, value):
    assert set(_found(text)) == {value}


@pytest.mark.parametrize(
    "text,value",
    [
        ("Study ID RX-418273 was assigned.", "RX-418273"),
        ("Participant id QL4827 enrolled today.", "QL4827"),
        ("Tracking number ZK-2291-77 attached.", "ZK-2291-77"),
    ],
)
def test_a_six_digit_mixed_identifier_after_an_announcing_word_is_found(text, value):
    assert value in _found(text)


@pytest.mark.parametrize(
    "text,value",
    [
        ("Code K42 ;; 7315 ;; QD on the scan.", "K42 ;; 7315 ;; QD"),
        ("Code K42||7315||QD on the scan.", "K42||7315||QD"),
        ("Code K42__7315__QD on the scan.", "K42__7315__QD"),
        ("Code HT :: 5820-J4 :: MV on the scan.", "HT :: 5820-J4 :: MV"),
    ],
)
def test_strange_delimiters_and_ocr_spacing_are_still_one_identifier(text, value):
    assert set(_found(text)) == {value}


@pytest.mark.parametrize(
    "text",
    [
        "BP 132/84 and HR 90 at rest.",
        "NEWS2 score 5 prompted escalation.",
        "dose 500-1000 mg daily",
        "Review on 12/03/2024 and 2026-08-03.",
        "ratio 3:1 recorded",
        "Timed cortisol 452 nmol/L was within range.",
        "MRI showed C3-C4/C5-C6 disc bulges.",
        "HLA-B*57:01 negative.",
        "| bay 7 | stable | 555-0142 |",
        "Sodium reference range 135-145 mmol/L.",
        "reference 3.5-5.0 applies",
        "Obs 12:30::13:45 handover times.",
    ],
)
def test_clinical_readings_dates_and_codes_are_not_unique_identifiers(text):
    assert _found(text) == []


# -- a model's phone claim on a code --------------------------------------------------


def test_a_model_phone_claim_on_a_six_digit_code_is_removed_as_a_unique_identifier():
    text = "Local reference K42;;7315;;QD on the referral."
    start = text.index("K42")

    assessment = assess_finding(
        text,
        start,
        start + 13,
        "PHONE_NUMBER",
        "gliner_multi_pii",
        contains_proper_noun=False,
    )

    assert (assessment.action, assessment.category) == ("redact", UNIQUE)


def test_the_seven_digit_phone_minimum_still_holds():
    text = "BP 132/84 recorded at triage."
    start = text.index("132/84")

    assessment = assess_finding(
        text,
        start,
        start + 6,
        "PHONE_NUMBER",
        "gliner_multi_pii",
        contains_proper_noun=False,
    )

    assert assessment.rationale != "exact_structured_match"
    assert assessment.action != "redact"


def test_digits_under_a_structured_claim_without_its_syntax_go_to_review():
    text = "Ext 4412 for the ward."

    assessment = assess_finding(
        text, 4, 8, "PHONE_NUMBER", "gliner_multi_pii", contains_proper_noun=False
    )

    assert assessment.action == "review"


def test_a_short_model_claim_holds_the_document_for_review(monkeypatch):
    # No announcing label ("Ext", "bleep", "phone"): only the model's claim.
    text = "See 4412 for the ward."
    model = _FixedSpanModel("gliner_multi_pii", "PHONE_NUMBER", "4412")

    outcome = _anonymize_with(monkeypatch, text, model)

    assert outcome["review_required_reasons"]
    assert not _release(outcome).releasable


@pytest.mark.parametrize(
    "text,claimed,category,token",
    [
        (
            "Driver licence <LICENSE_ID_1>; BP 128/82 mmHg.",
            "Driver licence <LICENSE_ID_1>",
            "DRIVER_LICENSE",
            "<LICENSE_ID_1>",
        ),
        (
            "County of residence <LOCATION_2>; stable.",
            "County of residence <LOCATION_2>",
            "ADDRESS",
            "<LOCATION_2>",
        ),
        (
            "Plate <VEHICLE_ID_2>; stable.",
            "Plate <VEHICLE_ID_2>",
            "ORGANIZATION",
            "<VEHICLE_ID_2>",
        ),
    ],
)
def test_a_model_span_of_label_words_and_a_generated_token_is_not_surviving_phi(
    monkeypatch, text, claimed, category, token
):
    findings = _residual(monkeypatch, text, claimed, category, {token})
    assert not [
        item
        for item in findings
        if item["blocking"] and item["detector"] == "gliner_multi_pii"
    ], findings


@pytest.mark.parametrize(
    "text,claimed,category,token",
    [
        (
            "Driver Jordan <LICENSE_ID_1>; stable.",
            "Driver Jordan <LICENSE_ID_1>",
            "DRIVER_LICENSE",
            "<LICENSE_ID_1>",
        ),
        (
            "Patient Noël Fictionel <PERSON_1>; stable.",
            "Patient Noël Fictionel <PERSON_1>",
            "PERSON",
            "<PERSON_1>",
        ),
        (
            "Driver licence D994437 <LICENSE_ID_1>; stable.",
            "Driver licence D994437 <LICENSE_ID_1>",
            "DRIVER_LICENSE",
            "<LICENSE_ID_1>",
        ),
    ],
)
def test_a_name_or_code_beside_a_generated_token_still_blocks(
    monkeypatch, text, claimed, category, token
):
    findings = _residual(monkeypatch, text, claimed, category, {token})
    assert [item for item in findings if item["blocking"]], findings


def test_trimming_a_label_word_never_leaves_the_span_inside_a_token(monkeypatch):
    """The model proposed "portal <URL_2>"; clinical-edge trimming cut it to "URL_2>"."""
    text = "SYNTHETIC record. portal <URL_2>; nasogastric tube in situ; SpO2 98%."
    model = _FixedSpanModel("gliner_multi_pii", "ORGANIZATION", "portal <URL_2>")

    rendered = _anonymize_with(monkeypatch, text, model)["anonymized_text"]

    assert "<<" not in rendered and ">>" not in rendered, rendered
    assert rendered.count("<") == rendered.count(">"), rendered


@pytest.mark.parametrize("claimed", ["URL_1>", "<URL_1", "RL_1", "REDACTED_URL>"])
def test_a_span_covering_part_of_an_existing_token_widens_to_the_whole_token(
    monkeypatch, claimed
):
    text = (
        "portal <URL_1>; stable."
        if "REDACTED" not in claimed
        else "portal <REDACTED_URL>; stable."
    )
    model = _FixedSpanModel("ner", "ORGANIZATION", claimed)

    outcome = _anonymize_with(monkeypatch, text, model)
    rendered = outcome["anonymized_text"]

    assert "<<" not in rendered and ">>" not in rendered, rendered
    assert not rendered.count("<") - rendered.count(">"), rendered


def test_a_token_already_in_the_text_is_replaced_whole_never_inside_its_brackets(
    monkeypatch,
):
    text = "Asian MRN <RECORD_ID_1>; extension 4417; laboratory result 4417."
    model = _FixedSpanModel("gliner_multi_pii", "MEDICAL_CONDITION", "Asian MRN")

    outcome = _anonymize_with(monkeypatch, text, model)

    assert (
        "<<" not in outcome["anonymized_text"]
        and ">>" not in outcome["anonymized_text"]
    )
    assert "laboratory result 4417" in outcome["anonymized_text"]


def test_a_number_inside_a_replaced_date_does_not_displace_the_whole_date(monkeypatch):
    """The year of "March 14, 2026" must not outrank the model's whole-date span."""
    text = "Discharged March 14, 2026; BP 128/82 mmHg."
    model = _FixedSpanModel("gliner_multi_pii", "DATE_TIME", "March 14, 2026")

    outcome = _anonymize_with(monkeypatch, text, model)

    assert "March" not in outcome["anonymized_text"]
    assert "2026" not in outcome["anonymized_text"]
    assert "BP 128/82 mmHg" in outcome["anonymized_text"]
    assert "ambiguous_short_number" not in outcome["review_required_reasons"]


def test_a_partial_date_span_still_holds_the_stranded_number(monkeypatch):
    """A span that stops short is not an enclosing replacement: the year still holds."""
    text = "Discharged March 14, 2026; BP 128/82 mmHg."
    model = _FixedSpanModel("gliner_multi_pii", "DATE_TIME", "March 14")

    outcome = _anonymize_with(monkeypatch, text, model)

    assert "2026" not in outcome["anonymized_text"]
    assert "ambiguous_short_number" in outcome["review_required_reasons"]


def test_a_labelled_extension_is_replaced_by_its_own_evidence(monkeypatch):
    """The same digits after an explicit extension label need no model claim."""
    text = "Ext 4412 for the ward."
    model = _FixedSpanModel("gliner_multi_pii", "PHONE_NUMBER", "4412")

    outcome = _anonymize_with(monkeypatch, text, model)

    assert "4412" not in outcome["anonymized_text"]
    assert "ambiguous_short_number" not in outcome["review_required_reasons"]


def test_a_model_phone_claim_on_a_code_is_replaced_and_labelled_a_unique_identifier(
    monkeypatch,
):
    text = "Local reference K42;;7315;;QD on the referral."
    model = _FixedSpanModel("gliner_multi_pii", "PHONE_NUMBER", "K42;;7315;;QD")

    outcome = _anonymize_with(monkeypatch, text, model)

    assert "7315" not in outcome["anonymized_text"]
    assert "<REDACTED_PHONE>" not in outcome["anonymized_text"]
    assert UNIQUE in outcome["detected_entities"]


# -- partial spans and repeats ---------------------------------------------------------


def test_a_partial_span_is_widened_only_to_the_validated_identifier():
    text = "Local reference K42;;7315;;QD on the referral."
    start = text.index("7315")
    partial = DetectedEntity(
        entity_type="IDENTIFIER",
        start=start,
        end=start + 4,
        source="stanford_deidentifier",
    )

    [widened] = expand_partial_identifiers([partial], text)

    assert text[widened.start : widened.end] in (
        "K42;;7315;;QD",
        "reference K42;;7315;;QD",
    )


def test_a_partial_span_over_a_reading_is_not_widened():
    text = "BP 132/84 and 4412 noted."
    start = text.index("132")
    partial = DetectedEntity(
        entity_type="IDENTIFIER",
        start=start,
        end=start + 3,
        source="stanford_deidentifier",
    )

    assert expand_partial_identifiers([partial], text) == [partial]


def test_a_confirmed_unique_identifier_is_removed_where_it_recurs():
    text = "Local reference K42;;7315;;QD. Letter quotes K42;;7315;;QD again."

    output = _anonymize(text)["anonymized_text"]

    assert "7315" not in output


def test_an_announced_unique_identifier_is_removed_where_it_recurs_without_its_head():
    text = "Study ID RX-418273 assigned. Consent form lists RX-418273 and RX-4182739."

    output = _anonymize(text)["anonymized_text"]

    assert "RX-418273 " not in output
    assert "RX-418273 and" not in output


def test_a_unique_identifier_repeat_respects_boundaries():
    text = "Local reference K42;;7315;;QD. Batch XK42;;7315;;QDX differs."
    found = [
        e for e in StructuredPatternDetector().detect(text) if e.entity_type == UNIQUE
    ]

    assert propagate_repeated_identifiers(found, text) == []


# -- the independent residual validator --------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Local reference K42;;7315;;QD on the referral.",
        "Code K42 ;; 7315 ;; QD on the scan.",
    ],
)
def test_the_residual_scan_finds_a_unique_identifier_on_its_own(text):
    assert residual_scan(text)


def test_the_residual_pass_catches_a_unique_identifier_the_first_pass_misses(
    monkeypatch,
):
    from services import phi_detection
    from services import text_anonymization as ta

    monkeypatch.setattr(
        phi_detection.StructuredPatternDetector, "detect", lambda self, text: []
    )
    ta._build_detectors.cache_clear()

    findings = ta.residual_findings("Code K42 ;; 7315 ;; QD on the scan.")

    assert any(
        f["blocking"] and f["detector"] == "residual_identifier_scan" for f in findings
    )


def test_a_released_document_never_carries_the_identifier():
    outcome = _anonymize("Local reference K42;;7315;;QD on the referral.")
    decision = _release(outcome)

    assert "K42;;7315;;QD" not in outcome["anonymized_text"]
    if decision.releasable:
        assert "7315" not in outcome["anonymized_text"]


def _residual(monkeypatch, text, claimed, category, emitted):
    from services import text_anonymization as ta
    from services.transformation_provenance import generated_token_provenance

    model = _FixedSpanModel("gliner_multi_pii", category, claimed)
    monkeypatch.setattr(
        ta, "_detectors", lambda *_args, **_kwargs: (StructuredPatternDetector(), model)
    )
    return ta.residual_findings(
        text, generated_token_provenance(text, frozenset(emitted))
    )
