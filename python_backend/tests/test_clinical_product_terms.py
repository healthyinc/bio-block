"""Vaccine, vaccine-code and clinical-device terms kept only where nothing reads as a name.

Found on the locked acceptance corpus: "pneumococcal PCV20" read as an
organization, "Influenza" at the start of a vaccine description read as a
name by the proper-noun fallback, and "Inhaler" at a sentence start read as a
name - each by one detector with no clinical reading to weigh against it.
The closed list gives them one, and only when the neighbouring words are not
name-like. Every identifier-shaped neighbour still wins. Invented values only.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.clinical_vocabulary import is_clinical_product_term  # noqa: E402
from services.detection_evidence import assess_finding  # noqa: E402


def _action(text, span, category="PERSON", source="ner", agreeing=1):
    start = text.index(span)
    return assess_finding(
        text, start, start + len(span), category, source, agreeing_detectors=agreeing
    ).action


@pytest.mark.parametrize(
    "text,span,category,source",
    [
        (
            "Vaccinations: influenza 2025 and pneumococcal PCV20. CRP 86 mg/L.",
            "pneumococcal PCV20",
            "ORGANIZATION",
            "ner",
        ),
        (
            "Influenza seasonal injectable preservative free",
            "Influenza",
            "PERSON",
            "strict_proper_noun",
        ),
        (
            "Pneumococcal conjugate PCV 20 valent",
            "Pneumococcal",
            "PERSON",
            "strict_proper_noun",
        ),
        (
            "Alice received influenza vaccine today.",
            "influenza vaccine",
            "PERSON",
            "ner",
        ),
        (
            "Peak flow 410 L/min. Inhaler technique was checked.",
            "Inhaler",
            "PERSON",
            "ner",
        ),
        (
            "Seen with her sister <PERSON_2> today. Inhaler technique was checked.",
            "Inhaler",
            "PERSON",
            "ner",
        ),
        ("Tdap given in the left deltoid.", "Tdap", "ORGANIZATION", "ner"),
        # A clinical action verb is not a name beside the code.
        ("Given PCV15 in the right deltoid.", "PCV15", "ORGANIZATION", "ner"),
        ("<Seq>FLAIR</Seq>", "FLAIR", "ORGANIZATION", "ner"),
    ],
)
def test_verified_clinical_product_terms_are_kept(text, span, category, source):
    assert _action(text, span, category, source) == "preserve"


@pytest.mark.parametrize(
    "text,span,category",
    [
        ("Given PCV20-Alice today.", "PCV20-Alice", "PERSON"),
        ("Reference MRN20 recorded.", "MRN20", "ORGANIZATION"),
        ("Reference John20 recorded.", "John20", "PERSON"),
        ("Row key PCV20X7Q4 recorded.", "PCV20X7Q4", "ORGANIZATION"),
        ("Administered by Influenza Smith today.", "Influenza Smith", "PERSON"),
        ("Administered by Influenza Smith today.", "Influenza", "PERSON"),
        ("Met Alice Inhaler at clinic.", "Inhaler", "PERSON"),
        ("Inhaler Smith attended clinic.", "Inhaler", "PERSON"),
        ("Seen by Dr. Inhaler today.", "Inhaler", "PERSON"),
        # A token-shaped neighbour, forged or genuine, may stand for a replaced
        # given name: "Alice Inhaler" with Alice already replaced.
        ("Reviewed <PERSON_1> Inhaler today.", "Inhaler", "PERSON"),
        ("Reviewed Inhaler <PERSON_1> today.", "Inhaler", "PERSON"),
        ("Reviewed PATIENT_004 Influenza today.", "Influenza", "PERSON"),
        ("Met Alice PCV15 today.", "PCV15", "ORGANIZATION"),
        ("Given Alice Tdap today.", "Tdap", "ORGANIZATION"),
    ],
)
def test_identifiers_and_name_structures_stay_protected(text, span, category):
    assert _action(text, span, category) == "redact"


def test_two_agreeing_detectors_send_a_listed_term_to_review_not_release():
    assert (
        _action(
            "Peak flow 410 L/min. Inhaler technique was checked.", "Inhaler", agreeing=2
        )
        == "review"
    )


@pytest.mark.parametrize(
    "text,span,category,source",
    [
        (
            "Contact pcv20@example.org for the record.",
            "pcv20@example.org",
            "EMAIL_ADDRESS",
            "structured_pattern",
        ),
        (
            "Call 020 7946 0321 about the vaccine.",
            "020 7946 0321",
            "PHONE_NUMBER",
            "structured_pattern",
        ),
    ],
)
def test_deterministic_identifiers_are_decided_before_the_vocabulary(
    text, span, category, source
):
    assert _action(text, span, category, source) == "redact"


@pytest.mark.parametrize(
    "term,expected",
    [
        ("PCV20", True),
        ("pneumococcal PCV20", True),
        ("Influenza vaccine", True),
        ("inhaler", True),
        ("PCV20-Alice", False),
        ("MRN20", False),
        ("John20", False),
        ("PCV2O", False),
        ("Walker", False),
        ("", False),
        ("PCV 20 valent", True),
        ("Influenza 5528", False),
        ("20", False),
        ("Pneumococcal conjugate PCV 20 valent", True),
        ("FLAIR", True),
        ("MPRAGE", True),
        ("Pet", False),
        ("Bold", False),
    ],
)
def test_the_list_is_closed_and_matched_token_by_token(term, expected):
    assert is_clinical_product_term(term) is expected


@pytest.mark.parametrize(
    "text,span,category,source,expected",
    [
        # A label left behind when its value was cut out of a redacted page.
        (
            "Patient:\nNHS\nTelephone\nDiagnosis: atrial fibrillation.",
            "Telephone",
            "PHONE_NUMBER",
            "gliner_multi_pii",
            "preserve",
        ),
        ("Signed: Dr\n", "Dr", "PERSON", "stanford_deidentifier", "preserve"),
        (
            "Clinic letter\nPatient:\nNHS\nTelephone\n",
            "NHS",
            "FACILITY",
            "stanford_deidentifier",
            "preserve",
        ),
        # Naming context still wins on the same line or when a word is not a label.
        ("Patient: NHS today", "NHS", "PERSON", "stanford_deidentifier", "redact"),
        ("Patient:\nContact Smith\n", "Contact Smith", "PERSON", "ner", "redact"),
        ("Seen by Dr Contact today", "Contact", "PERSON", "ner", "redact"),
        # The value itself, and a label word used inside a name, stay identifiers.
        (
            "Telephone 0113 496 0719 today",
            "0113 496 0719",
            "PHONE_NUMBER",
            "gliner_multi_pii",
            "redact",
        ),
        (
            "Patient: Mobile Tarrant attended.",
            "Mobile Tarrant",
            "PERSON",
            "ner",
            "redact",
        ),
    ],
)
def test_orphaned_contact_labels_are_labels_and_values_are_not(
    text, span, category, source, expected
):
    assert _action(text, span, category, source) == expected
