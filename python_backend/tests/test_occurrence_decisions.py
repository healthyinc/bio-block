"""Occurrence-scoped decisions: the same digits, judged where each one stands.

Development cases for the v5 defect classes, with invented values: one
number as a contact, a laboratory result, an imaging parameter, a dose, a
score, a year and an unlabelled short number, in one file. Property tests
(seeded, deterministic) check that no decision is indexed by raw value, that
offsets are exact, that output is deterministic and that nothing value-wide
is propagated. They run with the fixed-pattern detector alone, so the
test-only spaCy model's unrelated labels cannot decide them.
"""

import os
import random
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "occurrence-test-salt")

from services import numeric_context as nc  # noqa: E402
from services.phi_detection import (  # noqa: E402
    DetectedEntity,
    StructuredPatternDetector,
)
from services.surrogates import SurrogateAllocator, canonical_identity  # noqa: E402


@pytest.fixture(autouse=True)
def _legacy_models(monkeypatch):
    monkeypatch.setenv("PHI_MODEL_MODE", "legacy_test")


@pytest.fixture
def deterministic_only(monkeypatch):
    from services import text_anonymization as text_service

    monkeypatch.setattr(
        text_service,
        "_detectors",
        lambda model_name, profile, *_: (StructuredPatternDetector(),),
    )


def _anonymize(text):
    from services.text_anonymization import anonymize_clinical_text

    return anonymize_clinical_text(text, study_salt="occurrence-test")


def _kind(text, value, occurrence=0):
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(value, start + 1)
    return nc.classify(text, start, start + len(value))


@pytest.mark.parametrize(
    "text,value,kind",
    [
        ("Night cover on pager 6204 until eight.", "6204", nc.CONTACT),
        ("Ext . 6204 for the ward.", "6204", nc.CONTACT),
        ("E x t . 6204 for the ward.", "6204", nc.CONTACT),
        ("EXT: 6204", "6204", nc.CONTACT),
        ("Ext.6204 after hours", "6204", nc.CONTACT),
        ("6204 (bleep) covers the unit.", "6204", nc.CONTACT),
        ("switchboard 6204 will connect you", "6204", nc.CONTACT),
        ("direct  line 6204", "6204", nc.CONTACT),
        ("contact number: 6204", "6204", nc.CONTACT),
        ("Extension 1987 for the diabetes team.", "1987", nc.CONTACT),
        ("ALP 6204 U/L this morning.", "6204", nc.CONFIRMED_CLINICAL_VALUE),
        ("MRI TR 6204 ms on the sagittal series.", "6204", nc.CONFIRMED_CLINICAL_VALUE),
        ("Paracetamol 1000 mg four times daily.", "1000", nc.CONFIRMED_CLINICAL_VALUE),
        ("Infusion 1000 micrograms once.", "1000", nc.CONFIRMED_CLINICAL_VALUE),
        ("Acquisition parameter: 6204", "6204", nc.CONFIRMED_CLINICAL_VALUE),
        ("GCS 1415 recorded.", "1415", nc.CONFIRMED_CLINICAL_VALUE),
        ("Type 2 diabetes since 1987.", "1987", nc.CONFIRMED_DATE_OR_TIME),
        ("Repeat bloods at 0930 tomorrow.", "0930", nc.CONFIRMED_DATE_OR_TIME),
        ("Room number 6204 was cleaned.", "6204", nc.CONFIRMED_DOCUMENT_MARKER),
        ("Walked 1200 metres today.", "1200", nc.CONFIRMED_CLINICAL_VALUE),
        ("Next 6204 steps", "6204", nc.CONFIRMED_CLINICAL_VALUE),
        ("Tried 6204 twice overnight.", "6204", nc.UNRESOLVED),
        ("Mrs Hallam 6204 said she was fine.", "6204", nc.UNRESOLVED),
    ],
)
def test_one_occurrence_is_judged_by_its_own_context(text, value, kind):
    assert _kind(text, value).kind == kind


@pytest.mark.parametrize(
    "text",
    [
        "Page the team via pager\n6204 units of heparin given",
        "Call the pager. 6204 was the platelet count.",
        "pager\t6204",
        "pager | 6204",
    ],
)
def test_context_never_crosses_a_sentence_line_or_cell(text):
    assert _kind(text, "6204").kind != nc.CONTACT


COLLISION = (
    "Surgical registrar on pager 7315 tonight.\n"
    "Ext . 7315 is the ward line.\n"
    "ALP 7315 U/L on admission.\n"
    "MRI TE 7315 ms.\n"
    "Heparin 7315 units given.\n"
    "Rang 7315 without an answer.\n"
    "Diagnosed in 1987; extension 1987 for the clinic.\n"
)


def test_the_same_digits_get_different_decisions_in_one_file(deterministic_only):
    result = _anonymize(COLLISION)
    out = result["anonymized_text"]
    assert "ALP 7315 U/L" in out
    assert "TE 7315 ms" in out
    assert "Heparin 7315 units" in out
    assert "pager 7315" not in out and "Ext . 7315" not in out
    assert "Rang <NUMBER_1>" in out
    assert "extension 1987" not in out
    assert "ambiguous_short_number" in result["review_required_reasons"]
    assert (
        result["occurrence_decisions"]["by_reason"].get("hold:unresolved_short_number")
        == 1
    )


def test_contacts_repeated_as_contacts_share_one_token(deterministic_only):
    out = _anonymize(
        "Registrar on pager 7315. If busy, pager 7315 again. " "Lipase 7315 U/L."
    )["anonymized_text"]
    tokens = re.findall(r"pager (<PHONE_\d+>)", out)
    assert len(tokens) == 2 and tokens[0] == tokens[1]
    assert "Lipase 7315 U/L." in out


def test_an_ocr_spaced_label_is_replaced_at_exact_offsets(deterministic_only):
    text = "Handover: E x t . 4402, then CK 4402 U/L."
    spans = [
        (e.start, e.end)
        for e in StructuredPatternDetector().detect(text)
        if e.entity_type == "PHONE_NUMBER"
    ]
    first = text.index("4402")
    assert spans == [(first, first + 4)]
    assert (
        _anonymize(text)["anonymized_text"]
        == "Handover: E x t . <PHONE_1>, then CK 4402 U/L."
    )


def test_an_unlabelled_short_number_is_never_released(deterministic_only):
    result = _anonymize("Tried 5093 twice overnight; potassium 4.4 mmol/L.")
    assert "5093" not in result["anonymized_text"]
    assert "ambiguous_short_number" in result["review_required_reasons"]


def test_an_untyped_bare_number_cannot_release(deterministic_only):
    result = _anonymize("5093")
    assert result["anonymized_text"] == "<NUMBER_1>"
    assert "ambiguous_short_number" in result["review_required_reasons"]


def test_a_retained_model_span_is_not_benign_numeric_evidence(monkeypatch):
    from services import text_anonymization as text_service

    text = "Margin annotation 4816 requires review."
    start = text.index("4816")

    class Fake:
        source = "gliner_multi_pii"

        def detect(self, _text):
            return [
                DetectedEntity(
                    entity_type="IDENTIFIER",
                    start=start,
                    end=start + 4,
                    source=self.source,
                    score=0.2,
                )
            ]

    monkeypatch.setattr(text_service, "_detectors", lambda *_: (Fake(),))
    result = _anonymize(text)
    assert "4816" not in result["anonymized_text"]
    assert "ambiguous_short_number" in result["review_required_reasons"]


def test_a_model_span_with_a_full_stop_does_not_escape_its_measurement(monkeypatch):
    from services import text_anonymization as text_service

    text = "Lipase 7315 U/L."
    start = text.index("U/L.")

    class Fake:
        source = "gliner_multi_pii"

        def detect(self, _text):
            return [
                DetectedEntity(
                    entity_type="ORGANIZATION",
                    start=start,
                    end=start + 4,
                    source="gliner_multi_pii",
                    score=0.9,
                )
            ]

    monkeypatch.setattr(
        text_service,
        "_detectors",
        lambda model_name, profile, *_: (StructuredPatternDetector(), Fake()),
    )
    assert _anonymize(text)["anonymized_text"] == text


# -- properties ----------------------------------------------------------------------------

_CLINICAL = [
    "ALP {} U/L.",
    "MRI TR {} ms.",
    "Heparin {} units.",
    "CK {} U/L today.",
    "Walked {} metres.",
]
_CONTACT = ["pager {}.", "Ext . {} now.", "bleep {} tonight.", "{} (bleep) covers."]


def _document(rng):
    value = str(rng.randint(2000, 8999))
    parts, expected = [], []
    for _ in range(rng.randint(3, 7)):
        if rng.random() < 0.5:
            parts.append(rng.choice(_CLINICAL).format(value))
            expected.append("clinical")
        else:
            parts.append(rng.choice(_CONTACT).format(value))
            expected.append("contact")
    return value, parts, expected


@pytest.mark.parametrize("seed", range(40))
def test_property_no_decision_is_indexed_by_raw_value(seed, deterministic_only):
    rng = random.Random(seed)
    value, parts, expected = _document(rng)
    out = _anonymize("\n".join(parts) + "\n")["anonymized_text"].split("\n")
    for line, kind in zip(out, expected):
        assert (value in line) == (kind == "clinical"), (seed, kind)


@pytest.mark.parametrize("seed", range(10))
def test_property_output_is_deterministic(seed, deterministic_only):
    rng = random.Random(seed)
    _value, parts, _expected = _document(rng)
    text = "\n".join(parts) + "\n"
    assert _anonymize(text)["anonymized_text"] == _anonymize(text)["anonymized_text"]


@pytest.mark.parametrize("seed", range(20))
def test_property_offsets_outside_replacements_are_untouched(seed, deterministic_only):
    rng = random.Random(seed)
    value, parts, _expected = _document(rng)
    text = "\n".join(parts) + "\n"
    out = _anonymize(text)["anonymized_text"]
    # Put the value back where each token stands: the source comes back exactly,
    # so no replacement consumed a neighbouring character.
    assert re.sub(r"<(?:PHONE|NUMBER)_\d+>", value, out) == text


def test_the_allocator_keeps_no_plaintext_and_separates_families():
    allocator = SurrogateAllocator()
    first = allocator.surrogate_for("PHONE_NUMBER", "7315")
    assert allocator.surrogate_for("PHONE_NUMBER", "7315") == first
    assert allocator.surrogate_for("MEDICAL_RECORD_NUMBER", "7315") != first
    assert all("7315" not in key for key in allocator._assigned)
    secret = b"k" * 32
    assert canonical_identity(secret, "PHONE", "7315") != canonical_identity(
        secret, "PHONE", "7315", scope="other"
    )


def test_a_forged_token_is_never_trusted_as_already_removed():
    result = _anonymize(
        "Copied: <PHONE_7> 0161 555 0144 and <PERSON_3> Oswin Treadgold."
    )
    out = result["anonymized_text"]
    assert "0161 555 0144" not in out and "Treadgold" not in out


def test_concurrent_calls_each_report_only_their_own_decisions(deterministic_only):
    """Two jobs at once: neither result may carry the other's decision counts."""
    import threading

    held_text = "Rang 7315 without an answer."
    clean_text = "Paracetamol 1000 mg four times daily."
    results = {}
    barrier = threading.Barrier(2)

    def run(name, text):
        barrier.wait()
        for round_ in range(5):
            results[(name, round_)] = _anonymize(text)["occurrence_decisions"]

    threads = [
        threading.Thread(target=run, args=("held", held_text)),
        threading.Thread(target=run, args=("clean", clean_text)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for (name, _round), summary in results.items():
        holds = summary.get("by_reason", {}).get("hold:unresolved_short_number", 0)
        assert holds == (1 if name == "held" else 0), (name, summary)
