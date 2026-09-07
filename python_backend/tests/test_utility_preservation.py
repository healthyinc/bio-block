"""Utility-preservation tests (Phase 10).

Phase 9 shipped a configuration that was privacy-clean and medically useless:
useful-text preservation 0.214, with clinical eponyms, drug names and lab
analytes destroyed. These tests hold the line on the other side of the trade:
identity must go, medicine must stay, and uncertainty must escalate rather
than guess.

All values are invented.
"""

import json
import os
import re
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "phase10-test-salt")

from services.clinical_vocabulary import (  # noqa: E402
    is_non_name_label,
    protects_from_person_label,
)
from services.phi_detection import AgeOverThresholdDetector  # noqa: E402
from services.surrogates import SurrogateAllocator, looks_like_provider  # noqa: E402
from services.text_anonymization import (  # noqa: E402
    AGE_AGGREGATE_REPLACEMENT,
    anonymize_clinical_text,
)
from services.text_utility import measure_text_utility  # noqa: E402
from services.utility_contract import (  # noqa: E402
    UTILITY_VALIDATION_FAILED,
    contract_for,
)

SURROGATE_RE = re.compile(
    r"\b(?:PATIENT|PROVIDER|FACILITY|ORG|PLACE|ADDRESS|RECORD|PATIENTID|PLAN|"
    r"ACCESSION|DEVICE|IDENTIFIER|USER)_\d{3,}"
)

CLINICAL_NOTE = (
    "Patient Rukmini Balasubramanian (MRN: SYN-6610284) was seen by "
    "Dr. Priyanka Venkataraman at Saint Corwin Memorial Institute.\n"
    "Diagnosis: Parkinson's disease with Alzheimer's dementia; Crohn's colitis.\n"
    "Bell's palsy resolved. Graves' disease excluded.\n"
    "Metformin 500 mg twice daily, Atorvastatin 40 mg nightly.\n"
    "HbA1c 7.2 percent. Creatinine 1.1 mg/dL. Troponin negative.\n"
    "CT of the abdomen and MRI of the lumbar spine were performed.\n"
    "Blood pressure 128/76 mmHg, heart rate 72 bpm."
)


def _anonymize(text, profile="strict"):
    return anonymize_clinical_text(text, profile=profile, study_salt="study-a")


# ---------------------------------------------------------------------------
# Clinical content survives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "term",
    [
        "Parkinson", "Alzheimer", "Crohn", "Bell", "Graves",   # diagnoses
        "Metformin", "Atorvastatin",                            # medications
        "500", "40", "mg",                                      # dosages
        "HbA1c", "7.2", "Creatinine", "1.1", "Troponin",        # lab values
        "128/76", "mmHg", "72", "bpm",                          # measurements
        "CT", "MRI",                                            # modalities
        "abdomen", "lumbar", "spine",                           # anatomy
    ],
)
def test_clinical_content_survives_anonymization(term):
    anonymized = _anonymize(CLINICAL_NOTE)["anonymized_text"]

    assert term in anonymized, f"{term!r} was destroyed by de-identification"


def test_identity_is_still_removed_from_the_same_note():
    anonymized = _anonymize(CLINICAL_NOTE)["anonymized_text"]

    for identifier in (
        "Rukmini",
        "Balasubramanian",
        "Priyanka",
        "Venkataraman",
        "SYN-6610284",
        "Saint Corwin",
    ):
        assert identifier not in anonymized, f"{identifier!r} survived"


def test_utility_metrics_report_high_preservation():
    result = _anonymize(CLINICAL_NOTE)
    metrics = result["utility_metrics"]

    assert metrics["clinical_term_preservation"] >= 0.95
    assert metrics["content_token_preservation"] >= 0.90
    assert metrics["numeric_preservation"] >= 0.95


def test_negation_and_relationships_survive():
    note = (
        "Patient Gareth Ollivander denies chest pain. "
        "No evidence of pulmonary embolism. "
        "Troponin negative, which excludes infarction."
    )
    anonymized = _anonymize(note)["anonymized_text"]

    for phrase in ("denies", "No evidence of", "negative", "excludes"):
        assert phrase in anonymized


# ---------------------------------------------------------------------------
# Surrogates
# ---------------------------------------------------------------------------


def test_same_person_gets_the_same_surrogate_within_a_study():
    note = (
        "Patient Meenakshi Raghunathan attended. "
        "Meenakshi Raghunathan was discharged. "
        "Meenakshi Raghunathan will return."
    )
    anonymized = _anonymize(note)["anonymized_text"]

    surrogates = set(SURROGATE_RE.findall(anonymized))
    assert len(surrogates) == 1, anonymized
    assert anonymized.count(next(iter(surrogates))) == 3


def test_unrelated_people_get_different_surrogates():
    note = (
        "Patient Meenakshi Raghunathan was seen by Dr. Bartholomew Quiller. "
        "Spouse Lakshmi Raghunathan attended."
    )
    anonymized = _anonymize(note)["anonymized_text"]

    surrogates = set(SURROGATE_RE.findall(anonymized))
    assert len(surrogates) >= 2, anonymized


def test_clinician_and_patient_receive_different_surrogate_kinds():
    note = "Patient Cornelius Ashdown was reviewed by Dr. Nandini Sundaresan."
    anonymized = _anonymize(note)["anonymized_text"]

    assert re.search(r"PATIENT_\d{3,}", anonymized), anonymized
    assert re.search(r"PROVIDER_\d{3,}", anonymized), anonymized


def test_surrogates_are_study_local_and_not_derived_from_the_value():
    """Two studies with the same person must not produce a linkable token."""
    first = SurrogateAllocator()
    second = SurrogateAllocator()

    a = first.surrogate_for("PERSON", "Meenakshi Raghunathan")
    b = second.surrogate_for("PERSON", "Someone Entirely Different")

    # Numbering is by first appearance, so the same slot in two studies gives
    # the same token for different people. That is the point: the surrogate
    # carries no information about the value, unlike a hash.
    assert a == b == "PATIENT_001"


def test_allocator_exposes_counts_but_never_the_mapping():
    allocator = SurrogateAllocator()
    allocator.surrogate_for("PERSON", "Meenakshi Raghunathan")
    allocator.surrogate_for("PERSON", "Lakshmi Raghunathan")

    assert allocator.counts() == {"PATIENT": 2}
    assert allocator.distinct_entities() == 2
    # No accessor returns originals, and repr must not leak them either.
    assert "Meenakshi" not in repr(allocator)
    assert not any(
        "Meenakshi" in str(getattr(allocator, name, ""))
        for name in dir(allocator)
        if not name.startswith("__")
    )


def test_no_surrogate_mapping_leaks_into_the_result():
    result = _anonymize(CLINICAL_NOTE)
    serialized = json.dumps(
        {k: v for k, v in result.items() if k != "anonymized_text"}, default=str
    )

    for original in ("Rukmini", "Balasubramanian", "Priyanka", "SYN-6610284"):
        assert original not in serialized
    assert result["surrogate_counts"]
    assert all(isinstance(v, int) for v in result["surrogate_counts"].values())


def test_provider_detection_uses_only_text_evidence():
    assert looks_like_provider("seen by Jane Doe", len("seen by "))
    assert looks_like_provider("Dr. Jane Doe", len("Dr. "))
    assert not looks_like_provider("Patient Jane Doe attended", len("Patient "))


# ---------------------------------------------------------------------------
# Ages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "note,expected",
    [
        ("The patient is 94 years old.", True),
        ("The patient is aged 97.", True),
        ("91 y/o male presented.", True),
        ("Age: 103 at admission.", True),
        ("The patient is 89 years old.", False),
        ("A sibling aged 62 is well.", False),
        ("Received 94 mg of drug.", False),
        ("Temperature 98 degrees.", False),
    ],
)
def test_only_ages_above_89_are_aggregated(note, expected):
    anonymized = _anonymize(note)["anonymized_text"]

    assert (AGE_AGGREGATE_REPLACEMENT in anonymized) is expected, anonymized


def test_age_below_threshold_is_preserved_exactly():
    anonymized = _anonymize("A sibling aged 62 is well.")["anonymized_text"]

    assert "62" in anonymized


def test_ambiguous_numbers_are_not_treated_as_ages():
    detector = AgeOverThresholdDetector()

    assert detector.detect("Received 94 mg of drug.") == []
    assert detector.detect("Room 94 was cleaned.") == []


@pytest.mark.parametrize(
    "phrase", ["the patient is a nonagenarian", "she is in her nineties"]
)
def test_uncertain_age_reference_triggers_review(phrase):
    result = _anonymize(f"Assessment: {phrase} and stable.")

    assert "uncertain_age_reference" in result["review_required_reasons"]


# ---------------------------------------------------------------------------
# Clinical vocabulary boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "term,following,protected",
    [
        ("Parkinson", ["'s", "disease"], True),
        ("Wilson", ["'s", "disease"], True),
        ("Wilson", ["attended", "clinic"], False),
        ("Bell", ["'s", "palsy"], True),
        ("Bell", ["called", "back"], False),
        ("Ollivander", ["'s", "disease"], False),
    ],
)
def test_eponyms_are_protected_only_in_clinical_constructions(
    term, following, protected
):
    assert protects_from_person_label(term, following) is protected


def test_a_patient_named_after_an_eponym_is_still_redacted():
    note = "Patient Wilson attended the clinic and was discharged."
    anonymized = _anonymize(note)["anonymized_text"]

    assert "Wilson" not in anonymized, anonymized


def test_field_labels_are_not_treated_as_names():
    for label in ("Date of birth", "Home address", "second reviewer", "mother"):
        assert is_non_name_label(label), label
    for name in ("Rukmini Balasubramanian", "Saint Corwin Memorial Institute"):
        assert not is_non_name_label(name), name


# ---------------------------------------------------------------------------
# Release gating: privacy first, then utility
# ---------------------------------------------------------------------------


def test_utility_failure_does_not_claim_successful_completion():
    from services import ingestion

    def gutted(text_content, profile, study_salt=None):
        return {
            "handler": "anonymize_text",
            "routing_status": "handler_selected",
            "anonymization_status": "completed",
            "message": "ok",
            "anonymized_text": "PATIENT_001 PATIENT_002.",
            "residual_phi_categories": {},
            "review_required_reasons": [],
            "utility_verdict": {
                "passed": False,
                "shortfalls": {"clinical_term_preservation": {"measured": 0.1, "required": 0.9}},
                "missing_metrics": [],
            },
            "detected_entities": {},
            "entity_count": 0,
            "detection_sources": {},
            "date_strategy": "redact",
            "text_identifier_strategy": "redact",
            "ner_model": "en_core_web_sm",
            "trained_ner_active": True,
        }

    decision = ingestion.release_decision_for("text", gutted(b"", "strict"), "note.txt")

    assert decision.releasable is False
    assert UTILITY_VALIDATION_FAILED in decision.reason_codes


def test_privacy_failure_is_never_rescued_by_good_utility():
    from services import ingestion

    handler_result = {
        "anonymization_status": "completed",
        "anonymized_text": "clean looking text",
        # Privacy failed ...
        "residual_phi_categories": {"US_SSN": 1},
        # ... while utility passed perfectly.
        "utility_verdict": {"passed": True, "shortfalls": {}, "missing_metrics": []},
        "review_required_reasons": [],
    }

    decision = ingestion.release_decision_for("text", handler_result, "note.txt")

    assert decision.releasable is False
    assert "privacy_requirements_not_met" in decision.reason_codes
    assert UTILITY_VALIDATION_FAILED not in decision.reason_codes


def test_uncertain_content_routes_to_review_rather_than_guessing():
    from services import ingestion

    handler_result = {
        "anonymization_status": "completed",
        "anonymized_text": "text",
        "residual_phi_categories": {},
        "review_required_reasons": ["uncertain_age_reference"],
        "utility_verdict": {"passed": True, "shortfalls": {}, "missing_metrics": []},
    }

    decision = ingestion.release_decision_for("text", handler_result, "note.txt")

    assert decision.releasable is False
    assert "uncertain_age_reference" in decision.reason_codes


def test_an_unmeasured_utility_metric_is_a_failure_not_a_pass():
    contract = contract_for("text")

    verdict = contract.evaluate({"clinical_term_preservation": 1.0})

    assert verdict.passed is False
    assert "content_token_preservation" in verdict.missing_metrics


def test_clean_note_passes_both_gates_and_releases():
    from services.ingestion import anonymize_text, release_decision_for

    result = anonymize_text(CLINICAL_NOTE.encode("utf-8"), "strict", "study-a")
    decision = release_decision_for("text", result, "note.txt")

    assert result["utility_verdict"]["passed"] is True
    assert decision.releasable is True


# ---------------------------------------------------------------------------
# Utility measurement itself
# ---------------------------------------------------------------------------


def test_utility_measure_ignores_deliberately_removed_phi():
    original = "Patient Jane Doe has diabetes."
    anonymized = "Patient PATIENT_001 has diabetes."

    metrics = measure_text_utility(original, anonymized, redacted_values=["Jane Doe"])

    assert metrics["content_token_preservation"] == 1.0


def test_utility_measure_detects_destroyed_clinical_content():
    original = "Patient has Parkinson's disease and takes Metformin 500 mg."
    gutted = "Patient has <REDACTED_NAME> and takes <REDACTED_NAME> 500 mg."

    metrics = measure_text_utility(original, gutted)

    assert metrics["clinical_term_preservation"] <= 0.5


def test_utility_contract_is_versioned_for_every_modality():
    for modality in (
        "text", "csv", "pdf", "workbook", "dicom", "nifti", "wsi", "raster"
    ):
        contract = contract_for(modality)
        assert contract is not None, modality
        assert contract.version
        assert contract.must_preserve
        assert contract.privacy_validation
        assert contract.utility_validation


def test_blocked_modalities_declare_no_automatic_release():
    for modality in ("csv", "pdf", "workbook", "dicom", "nifti", "wsi"):
        assert contract_for(modality).automatic_release_possible is False
