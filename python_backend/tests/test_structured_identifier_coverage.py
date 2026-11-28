"""Structured-identifier coverage: local phones, accounts, devices and repeats.

Acceptance-v3 released a document carrying a seven-digit local phone number,
and missed device identifiers announced by "UDI" and an account announced by
"acct", because no rule knew those forms. The fixes are grammar written once
in services.structured_identifier_spec and read twice: precisely by the first
pass, loosely and by separate code in the residual validator.

Every positive here has a counterexample that must not match - a blood
pressure, a range, a dose, a date - and none of the values come from any
corpus. All values are invented.
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "structured-identifier-test-salt")

from services import structured_identifier_spec as spec  # noqa: E402
from services.detection_evidence import assess_finding  # noqa: E402
from services.phi_detection import (  # noqa: E402
    SOURCE_REPEAT_PROPAGATION,
    SOURCE_RESIDUAL_IDENTIFIER_SCAN,
    DetectedEntity,
    StructuredPatternDetector,
    expand_partial_identifiers,
    propagate_repeated_identifiers,
)
from services.residual_identifier_scan import scan as residual_scan  # noqa: E402


@pytest.fixture(autouse=True)
def _legacy_models(monkeypatch):
    monkeypatch.setenv("PHI_MODEL_MODE", "legacy_test")


def _first_pass(text, category):
    return [
        text[e.start : e.end]
        for e in StructuredPatternDetector().detect(text)
        if e.entity_type == category
    ]


def _accounts(text):
    return [
        e
        for e in StructuredPatternDetector().detect(text)
        if e.entity_type == "ACCOUNT_NUMBER"
    ]


def _residual(text):
    return {e.entity_type for e in residual_scan(text)}


def _anonymize(text):
    from services import text_anonymization as ta

    ta._build_detectors.cache_clear()
    return ta.anonymize_clinical_text(text, profile="strict")


# -- the specification ----------------------------------------------------------


def test_repetition_is_limited_to_high_confidence_structured_categories():
    assert spec.REPEATABLE_CATEGORIES == frozenset(
        {
            "PHONE_NUMBER",
            "ACCOUNT_NUMBER",
            "DEVICE_ID",
            "US_SSN",
            "SSN",
            "URL",
            "IP_ADDRESS",
            "MEDICAL_RECORD_NUMBER",
            "HEALTH_PLAN_ID",
            "INSURANCE_ID",
            "OTHER_UNIQUE_ID",
        }
    )
    for category in (
        "PERSON",
        "ORGANIZATION",
        "DATE",
        "DATE_TIME",
        "LOCATION",
        "ADDRESS",
    ):
        assert category not in spec.REPEATABLE_CATEGORIES
    assert set(spec.REPEAT_MIN_DIGITS) == set(spec.REPEATABLE_CATEGORIES)


def test_every_announced_identifier_needs_digits():
    assert all(identifier.min_digits >= 1 for identifier in spec.KEYED_IDENTIFIERS)


# -- local telephone numbers -----------------------------------------------------


@pytest.mark.parametrize(
    "text,value",
    [
        ("reviewed with 555.0142 today.", "555.0142"),
        ("Contacts: 555-0187; pharmacy; ward.", "555-0187"),
        ("NOTE: 555.0163 REVIEWED", "555.0163"),
        ("| 555-0129 | bay 7 | stable |", "555-0129"),
        ("From: 555.0171", "555.0171"),
        ("BP 132/84 555.0155 HR 90.", "555.0155"),
        ("(555) 0142 on file", "(555) 0142"),
        ("call 555 0142 later", "555 0142"),
        ("tel: 5550142", "5550142"),
        ("Mobile no. 555 0199 for the daughter.", "555 0199"),
        ("landline:(555)0133", "(555)0133"),
    ],
)
def test_a_local_phone_number_is_found(text, value):
    assert value in _first_pass(text, "PHONE_NUMBER")
    assert "PHONE_NUMBER" in _residual(text)


@pytest.mark.parametrize(
    "text",
    [
        "BP 132/84 and HR 90 at rest.",
        "Sodium 135-1450 was queried.",
        "dose 150-2000 mg daily",
        "infusion 125.5000 ml over the shift",
        "reference 123-4567 applies",
        "Seen 2026-08-03 at 14:30.",
        "Review on 12/03/2024.",
        "Fracture code S72.0012 recorded.",
        "HbA1c 7.2 percent, ratio 3:1.",
        "temperature 37.4 overnight",
    ],
)
def test_a_reading_date_or_code_is_not_a_phone_number(text):
    assert _first_pass(text, "PHONE_NUMBER") == []


# -- account numbers ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "billing: acct 101-202-303.",
        "Account No. 44172290 was closed.",
        "account number: 9912-3344 on file",
        "refund to a/c 55120099",
        "ACCT-12345678 noted",
        "acc't #7781234 queried",
    ],
)
def test_an_announced_account_number_is_found(text):
    assert _first_pass(text, "ACCOUNT_NUMBER")
    assert "ACCOUNT_NUMBER" in _residual(text)


@pytest.mark.parametrize(
    "text",
    [
        "Admitted on account of pain.",
        "a/c 10 units before meals",
        "took into account 3 readings",
        "Account reviewed with the family.",
    ],
)
def test_the_word_account_alone_is_not_an_account_number(text):
    assert _first_pass(text, "ACCOUNT_NUMBER") == []


# -- device identifiers ------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "fitted UDI DEV7000001 today.",
        "udi-di 08445880 recorded",
        "serial no. SN-4471-22 checked",
        "S/N 88AB77CD",
        "implant serial IMP-7000001 noted.",
        "unique device identifier (01)08445880(17)141120",
    ],
)
def test_an_announced_device_identifier_is_found(text):
    assert _first_pass(text, "DEVICE_ID")
    assert "DEVICE_ID" in _residual(text)


@pytest.mark.parametrize(
    "text",
    [
        "device reviewed and removed",
        "implant site healing well",
        "equipment checked at the bedside",
        "serial observations continued",
    ],
)
def test_device_words_without_a_value_are_not_identifiers(text):
    assert _first_pass(text, "DEVICE_ID") == []


@pytest.mark.parametrize(
    "text,category,value",
    [
        (
            "Follow-up via lomar.12@mail.ward-rota.invalid.Next review booked.",
            "EMAIL_ADDRESS",
            "lomar.12@mail.ward-rota.invalid",
        ),
        (
            "Follow-up via https://portal.ward-rota.invalid/p/448129.Next review booked.",
            "URL",
            "https://portal.ward-rota.invalid/p/448129",
        ),
        (
            "Follow-up via legacy code QZ::5820-B4::TR.Next review booked.",
            "OTHER_UNIQUE_ID",
            "QZ::5820-B4::TR",
        ),
    ],
)
def test_a_match_stops_before_a_sentence_glued_on_with_no_space(text, category, value):
    assert set(_first_pass(text, category)) == {value}


@pytest.mark.parametrize(
    "text,category,value",
    [
        (
            "Write to lomar.12@mail.ward-rota.invalid for results.",
            "EMAIL_ADDRESS",
            "lomar.12@mail.ward-rota.invalid",
        ),
        (
            "Portal www.ward-rota.invalid/Results page.",
            "URL",
            "www.ward-rota.invalid/Results",
        ),
    ],
)
def test_an_ordinary_address_is_not_shortened(text, category, value):
    assert _first_pass(text, category) == [value]


def test_a_closing_bracket_outside_the_value_is_not_swallowed():
    text = "(implant serial IMP-7000001)"

    # Phase 2: the announcing word is a label and stays; only the value goes.
    assert _first_pass(text, "DEVICE_ID") == ["IMP-7000001"]


# -- a model's span that stops inside an identifier ----------------------------------


def _model_span(category, start, end, source="stanford_deidentifier"):
    return DetectedEntity(entity_type=category, start=start, end=end, source=source)


def test_a_partial_identifier_span_is_widened_to_the_whole_token():
    text = "fitted DEV7000001 today"

    [widened] = expand_partial_identifiers([_model_span("IDENTIFIER", 7, 12)], text)

    assert text[widened.start : widened.end] == "DEV7000001"


def test_a_name_span_is_never_widened():
    text = "Seen by Rukminidevi today"
    partial = _model_span("PERSON", 8, 13, source="gliner_multi_pii")

    assert expand_partial_identifiers([partial], text) == [partial]


def test_a_measurement_next_to_an_identifier_span_is_not_absorbed():
    text = "creatinine 141.55 mmol/L"
    partial = _model_span("IDENTIFIER", 11, 14)

    assert expand_partial_identifiers([partial], text) == [partial]


# -- repeats ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,value",
    [
        (
            "Account 44172290 closed. Refund issued against 44172290 on request.",
            "44172290",
        ),
        ("Tel 555 0187 for the ward. Family rang 555 0187 twice.", "555 0187"),
        ("UDI DEV7000123 implanted; dev7000123 checked at clinic.", "7000123"),
        ("Member ID HP-55120099 verified; HP-55120099 billed.", "55120099"),
    ],
)
def test_a_confirmed_value_is_removed_wherever_it_recurs(text, value):
    output = _anonymize(text)["anonymized_text"]

    assert value.casefold() not in output.casefold()


def test_names_dates_and_short_values_are_never_propagated():
    text = "Ashvelin reviewed Ashvelin. Seen 2026-08-03 and 2026-08-03. acct 12345 and 12345."
    entities = [
        _model_span("PERSON", 0, 8, source="gliner_multi_pii"),
        _model_span("DATE_TIME", 33, 43, source="structured_pattern"),
        *_accounts(text),
    ]

    assert propagate_repeated_identifiers(entities, text) == []


def test_propagation_respects_identifier_boundaries():
    text = (
        "Account 441722 closed. Batch 441722-7, A441722, 9-441722 and 1441722 differ."
    )

    assert propagate_repeated_identifiers(_accounts(text), text) == []


def test_a_propagated_finding_carries_a_category_and_offsets_only():
    text = "Account 44172290 closed; 44172290 refunded."

    [repeat] = propagate_repeated_identifiers(_accounts(text), text)

    assert repeat.source == SOURCE_REPEAT_PROPAGATION
    assert repeat.entity_type == "ACCOUNT_NUMBER"
    assert repeat.original_label == "repeat_of_structured_identifier"
    assert repeat.score is None


# -- the residual validator is independent and more conservative ----------------------


def test_the_residual_scan_shares_no_code_with_the_first_pass_expressions():
    from services import residual_identifier_scan

    source = inspect.getsource(residual_identifier_scan)
    for name in (
        "STRUCTURED_PATTERNS",
        "StructuredPatternDetector",
        "_keyed_definition",
        "_LOCAL_PHONE",
        "_CONTEXT_PHONE",
        "structured_value",
        "_matches",
    ):
        assert name not in source


def test_a_broken_first_pass_pattern_cannot_hide_an_identifier_from_the_residual_pass(
    monkeypatch,
):
    from services import phi_detection
    from services import text_anonymization as ta

    monkeypatch.setattr(
        phi_detection.StructuredPatternDetector, "detect", lambda self, text: []
    )
    ta._build_detectors.cache_clear()

    findings = ta.residual_findings(
        "Refund to acct 4417-22901. Fitted UDI DEV7000123. Reviewed with 555.0187 today."
    )

    blocking = {f["category"] for f in findings if f["blocking"]}
    assert {"ACCOUNT_NUMBER", "DEVICE_ID", "PHONE_NUMBER"} <= blocking


@pytest.mark.parametrize(
    "text",
    [
        "Sodium 135-1450 was queried.",
        "reference 123-4567 applies",
        "account on file 4417",
    ],
)
def test_the_residual_scan_flags_what_the_first_pass_deliberately_leaves(text):
    assert residual_scan(text)


def test_the_residual_scan_does_not_block_the_sanitizers_own_tokens():
    result = _anonymize(
        "Account 44172290 closed. UDI DEV7000123 fitted. Reviewed with 555.0187 today."
    )

    assert not set(result["residual_phi_categories"]) & {
        "ACCOUNT_NUMBER",
        "DEVICE_ID",
        "PHONE_NUMBER",
        "IDENTIFIER",
    }
    for value in ("44172290", "DEV7000123", "555.0187"):
        assert value not in result["anonymized_text"]


def _scan_blocking(text, regions):
    from services import text_anonymization as ta
    from services.transformation_provenance import KIND_SURROGATE, ProvenanceBuilder

    builder = ProvenanceBuilder()
    for start, length in regions:
        builder.record(start, length, KIND_SURROGATE, "PERSON")
    ta._build_detectors.cache_clear()
    return [
        f
        for f in ta.residual_findings(text, builder.build())
        if f["detector"] == SOURCE_RESIDUAL_IDENTIFIER_SCAN and f["blocking"]
    ]


def test_adjacent_surrogates_are_not_read_as_a_surviving_identifier():
    assert _scan_blocking("PATIENT_001PATIENT_002 attended.", [(0, 11), (11, 11)]) == []


def test_original_digits_beside_a_surrogate_still_block():
    assert _scan_blocking("PATIENT_00144172290 attended.", [(0, 11)])


# -- a model's category claim ----------------------------------------------------------


def test_a_model_phone_claim_on_a_short_number_gets_no_layer_one_authority():
    text = "Ext 4412 for the ward."

    assessment = assess_finding(text, 4, 8, "PHONE_NUMBER", "gliner_multi_pii")

    assert assessment.rationale != "exact_structured_match"


def test_a_model_phone_claim_on_a_real_number_still_wins():
    text = "Reviewed with 555.0187 today."
    start = text.index("555.0187")

    assessment = assess_finding(
        text, start, start + 8, "PHONE_NUMBER", "gliner_multi_pii"
    )

    assert (assessment.action, assessment.rationale) == (
        "redact",
        "exact_structured_match",
    )


def test_safe_harbor_maps_an_account_number_to_its_category():
    from services.safe_harbor import TEXT_ENTITY_CATEGORIES

    assert TEXT_ENTITY_CATEGORIES["ACCOUNT_NUMBER"] == 11


# -- a diagnosis code is read the same way by both passes -------------------


def _both_keep(text, term):
    """(first pass keeps it, residual validator keeps it)."""
    start = text.index(term)
    assessment = assess_finding(
        text,
        start,
        start + len(term),
        "IDENTIFIER",
        "gliner_multi_pii",
    )
    flagged = {text[e.start : e.end].casefold() for e in residual_scan(text)}
    return assessment.action == "preserve", term.casefold() not in flagged


@pytest.mark.parametrize(
    "text,term",
    [
        ("Diagnosis: J45.909. Asthma remains controlled.", "J45.909"),
        ("Diagnosis: J45.909", "J45.909"),
        ("Dx: j45.909 reviewed.", "j45.909"),
        ("ICD-10: J45.909 asthma", "J45.909"),
        ("ICD-10-CM: J45.909 asthma", "J45.909"),
        ("Primary Diagnosis: G43.909 migraine", "G43.909"),
        ("Diagnosis: I48.91 atrial fibrillation", "I48.91"),
        ("Diagnosis: N18.3 stage 3", "N18.3"),
    ],
)
def test_both_passes_keep_a_governed_diagnosis_code(text, term):
    """The first pass kept these and the residual validator called them
    identifiers, so the document was anonymized correctly and then held. The
    grammar is shared precisely so the two cannot disagree again."""
    assert _both_keep(text, term) == (True, True)


@pytest.mark.parametrize(
    "text,term",
    [
        ("MRN: J45.909", "J45.909"),
        ("Account: J45.909", "J45.909"),
        ("Patient ID: J45.909", "J45.909"),
        ("The diagnosis was confirmed. J45.909", "J45.909"),
        ("Diagnosis: A12.34567", "A12.34567"),
        ("MRN: FZ814001", "FZ814001"),
    ],
)
def test_both_passes_still_remove_an_identifier(text, term):
    """An ICD-looking value under an identifier's own head, one in a sentence
    that merely mentions the word, and a value that is not a code at all."""
    assert _both_keep(text, term) == (False, False)


@pytest.mark.parametrize("term", ["J45.909X7", "J45.909-22"])
def test_the_residual_validator_flags_a_longer_token_whole(term):
    """The exemption covers a whole token. A longer string that opens with an
    ICD-looking head is flagged entire, not trimmed back to the code."""
    text = f"Diagnosis: {term} on file."

    assert [e.entity_type for e in residual_scan(text)] == ["IDENTIFIER"]
    assert [text[e.start : e.end] for e in residual_scan(text)] == [term]


def test_the_diagnosis_grammar_is_read_from_the_shared_spec():
    """Neither pass may keep its own copy: that drift is what this module
    exists to prevent, and it is how the two disagreed here."""
    from services import detection_evidence, residual_identifier_scan

    assert spec.SPEC_VERSION.startswith("structured-identifier-spec-v")
    assert "diagnosis" in spec.DIAGNOSIS_FIELD_HEADS
    for source in (
        inspect.getsource(detection_evidence),
        inspect.getsource(residual_identifier_scan),
    ):
        assert "DIAGNOSIS_CODE_PATTERN" in source
        assert "DIAGNOSIS_FIELD_HEADS" in source


# -- a model span may cover the field as well as the code -------------------


def _both(text, term):
    """(first pass keeps it, residual validator leaves the code token alone)."""
    start = text.index(term)
    assessment = assess_finding(
        text,
        start,
        start + len(term),
        "HEALTH_PLAN_ID",
        "gliner_multi_pii",
    )
    codes = [
        word
        for word in term.replace(":", " ").split()
        if any(c.isdigit() for c in word)
    ]
    flagged = {text[e.start : e.end].casefold() for e in residual_scan(text)}
    kept = not codes or codes[-1].casefold() not in flagged
    return assessment.action == "preserve", kept


@pytest.mark.parametrize(
    "text,term",
    [
        # The span GLiNER actually proposed when the Phase 18 holdout failed.
        ("Dx: J45.909. Impression consistent with asthma.", "Dx: J45.909"),
        (
            "Diagnosis: J45.909. Impression consistent with asthma.",
            "Diagnosis: J45.909",
        ),
        ("Primary diagnosis: E11.9 reviewed.", "Primary diagnosis: E11.9"),
        ("Discharge diagnosis: N18.3 stable.", "Discharge diagnosis: N18.3"),
        ("ICD-10: N18.3 stable.", "ICD-10: N18.3"),
        ("Diagnosis  J45.909 recorded.", "Diagnosis  J45.909"),
        ("Dx : J45.909 recorded.", "Dx : J45.909"),
        ("Dx： J45.909 recorded.", "Dx： J45.909"),
        ("Dx – J45.909 recorded.", "Dx – J45.909"),
        ("Dx: j45.909 recorded.", "Dx: j45.909"),
        ("Diagnosis: J45 recorded.", "Diagnosis: J45"),
    ],
)
def test_a_model_span_covering_the_field_and_code_keeps_both(text, term):
    """A detector draws its span where it likes. GLiNER called "Dx: J45.909"
    a health plan identifier, and because that string is not an ICD token the
    exact-identifier layer removed the label and the code together. The field
    is now read as a structure, so the same span is recognised."""
    assert _both(text, term) == (True, True)


@pytest.mark.parametrize(
    "term",
    ["J45.909", "909", ".909", "45.909"],
)
def test_the_same_field_is_recognised_however_narrow_the_span(term):
    text = "Dx: J45.909. Impression consistent with asthma."

    assert _both(text, term) == (True, True)


@pytest.mark.parametrize(
    "text,term",
    [
        ("MRN: J45.909", "MRN: J45.909"),
        ("Patient ID: J45.909", "Patient ID: J45.909"),
        ("Account: J45.909", "Account: J45.909"),
        ("Member ID: J45.909", "Member ID: J45.909"),
        ("Insurance ID: J45.909", "Insurance ID: J45.909"),
        ("Health plan: J45.909", "Health plan: J45.909"),
        ("Device ID: J45.909", "Device ID: J45.909"),
        ("Reference: J45.909", "Reference: J45.909"),
        ("Accession: J45.909", "Accession: J45.909"),
    ],
)
def test_an_identifier_head_governing_the_same_shape_is_still_removed(text, term):
    """The shapes collide exactly. Only the head separates them, and an
    identifier's own head is not in the diagnosis alternation."""
    assert _both(text, term) == (False, False)


@pytest.mark.parametrize(
    "text,term",
    [
        ("Diagnosis: J45.909X7 on file.", "Diagnosis: J45.909"),
        ("Diagnosis: J45.909-22 on file.", "Diagnosis: J45.909"),
        ("Diagnosis: A12.34567 recorded.", "Diagnosis: A12.34567"),
        ("Diagnosis: 1145.909 recorded.", "Diagnosis: 1145.909"),
        ("Diagnosis: JJ5.909 recorded.", "Diagnosis: JJ5.909"),
        ("Diagnosis: J45.909 E11.9 both.", "Diagnosis: J45.909 E11.9"),
        ("Diagnosis: J45.909 MRN DV8261547", "Diagnosis: J45.909 MRN DV8261547"),
        ("The diagnosis was confirmed. J45.909", "J45.909"),
        ("Diagnosed previously; see J45.909.", "J45.909"),
    ],
)
def test_the_exemption_refuses_anything_but_one_governed_code(text, term):
    """A code inside a longer identifier, a malformed code, two codes in one
    span, a second identifier in the same span, and prose that merely uses
    the word - none of these is a diagnosis field."""
    start = text.index(term)

    assessment = assess_finding(
        text,
        start,
        start + len(term),
        "HEALTH_PLAN_ID",
        "gliner_multi_pii",
    )

    assert assessment.action != "preserve"


def test_a_deterministic_identifier_still_wins_over_the_field_reading():
    """The exemption only ever overrules a model's opinion. A deterministic
    match returns before it is reached."""
    text = "Dx: J45.909"
    start = text.index("J45.909")

    assessment = assess_finding(
        text,
        start,
        start + len("J45.909"),
        "MEDICAL_RECORD_NUMBER",
        "structured_pattern",
    )

    assert assessment.action == "redact"


def test_the_field_grammar_is_shared_and_versioned():
    from services import detection_evidence, residual_identifier_scan

    assert spec.SPEC_VERSION == "structured-identifier-spec-v6"
    for source in (
        inspect.getsource(detection_evidence),
        inspect.getsource(residual_identifier_scan),
    ):
        assert "DIAGNOSIS_FIELD_SEPARATORS" in source
        assert "DIAGNOSIS_FIELD_HEADS" in source
