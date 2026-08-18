import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import text_anonymization  # noqa: E402
from services.text_anonymization import (  # noqa: E402
    TextAnonymizationError,
    anonymize_clinical_text,
)


def test_mrn_with_context_is_detected_and_replaced():
    result = anonymize_clinical_text(
        "Patient has MRN: 123456.",
        study_salt="study-a",
    )

    assert result["anonymization_status"] == "completed"
    assert "123456" not in result["anonymized_text"]
    assert "<REDACTED_MRN>" in result["anonymized_text"]
    assert result["detected_entities"]["MEDICAL_RECORD_NUMBER"] == 1


def test_random_number_without_context_is_not_mrn():
    text = "The room number 123456 was cleaned."
    result = anonymize_clinical_text(text, study_salt="study-a")

    assert result["anonymized_text"] == text
    assert "MEDICAL_RECORD_NUMBER" not in result["detected_entities"]


def test_mrn_context_is_case_insensitive_and_accepts_separators():
    result = anonymize_clinical_text(
        "mrn: 123456. Medical Record Number - 654321.",
        study_salt="study-a",
    )

    assert "123456" not in result["anonymized_text"]
    assert "654321" not in result["anonymized_text"]
    assert result["detected_entities"]["MEDICAL_RECORD_NUMBER"] == 2


def test_patient_id_is_detected_and_replaced():
    result = anonymize_clinical_text(
        "Patient ID PT-1001 was admitted.",
        study_salt="study-a",
    )

    assert "PT-1001" not in result["anonymized_text"]
    assert "<REDACTED_PATIENT_ID>" in result["anonymized_text"]
    assert result["detected_entities"]["PATIENT_ID"] == 1


def test_health_plan_id_is_detected_and_replaced():
    result = anonymize_clinical_text(
        "Insurance ID ABC123456789 was verified.",
        study_salt="study-a",
    )

    assert "ABC123456789" not in result["anonymized_text"]
    assert "<REDACTED_HEALTH_PLAN>" in result["anonymized_text"]
    assert result["detected_entities"]["HEALTH_PLAN_ID"] == 1


def test_deterministic_surrogate_consistency_with_same_salt():
    text = "Patient has MRN: 123456."

    first = anonymize_clinical_text(text, profile="research", study_salt="study-a")
    second = anonymize_clinical_text(text, profile="research", study_salt="study-a")

    assert first["anonymized_text"] == second["anonymized_text"]


def test_different_salt_changes_surrogate():
    text = "Patient has MRN: 123456."

    first = anonymize_clinical_text(text, profile="research", study_salt="study-a")
    second = anonymize_clinical_text(text, profile="research", study_salt="study-b")

    assert first["anonymized_text"] != second["anonymized_text"]
    assert "123456" not in first["anonymized_text"]
    assert "123456" not in second["anonymized_text"]


def test_email_is_redacted():
    result = anonymize_clinical_text(
        "Contact john.doe@example.com after review.",
        study_salt="study-a",
    )

    assert "john.doe@example.com" not in result["anonymized_text"]
    assert "<REDACTED_EMAIL>" in result["anonymized_text"]
    assert result["detected_entities"]["EMAIL_ADDRESS"] == 1


def test_phone_is_redacted():
    result = anonymize_clinical_text(
        "Call 555-123-4567 for scheduling.",
        study_salt="study-a",
    )

    assert "555-123-4567" not in result["anonymized_text"]
    assert "<REDACTED_PHONE>" in result["anonymized_text"]
    assert result["detected_entities"]["PHONE_NUMBER"] == 1


def test_medical_terms_are_preserved():
    text = "History includes myocardial infarction, diabetes, aspirin, and CT scan."
    result = anonymize_clinical_text(text, study_salt="study-a")

    assert "myocardial infarction" in result["anonymized_text"]
    assert "diabetes" in result["anonymized_text"]
    assert "aspirin" in result["anonymized_text"]
    assert "CT scan" in result["anonymized_text"]


def test_no_phi_text_succeeds_and_is_preserved():
    text = "Patient was diagnosed with diabetes and prescribed metformin."
    result = anonymize_clinical_text(text, study_salt="study-a")

    assert result["anonymization_status"] == "completed"
    assert result["anonymized_text"] == text
    assert result["detected_entities"] == {}


def test_empty_text_is_rejected():
    with pytest.raises(TextAnonymizationError) as exc:
        anonymize_clinical_text("   ", study_salt="study-a")

    assert exc.value.status_code == 400
    assert exc.value.detail == "Text input is empty"


def test_missing_salt_is_rejected(monkeypatch):
    monkeypatch.delenv(text_anonymization.STUDY_SALT_ENV_VAR, raising=False)
    monkeypatch.setattr(text_anonymization, "_read_local_env_salt", lambda: None)

    with pytest.raises(TextAnonymizationError) as exc:
        anonymize_clinical_text("Patient has MRN: 123456.")

    assert exc.value.status_code == 500
    assert text_anonymization.STUDY_SALT_ENV_VAR in exc.value.detail


def test_entity_summary_does_not_include_raw_values():
    result = anonymize_clinical_text(
        "Patient has MRN: 123456 and email john.doe@example.com.",
        study_salt="study-a",
    )

    assert "123456" not in result["detected_entities"]
    assert "john.doe@example.com" not in result["detected_entities"]
    assert all(isinstance(key, str) for key in result["detected_entities"])
    assert all(isinstance(value, int) for value in result["detected_entities"].values())


def test_overlapping_email_text_does_not_leave_email_exposed():
    result = anonymize_clinical_text(
        "John Doe <john.doe@example.com> was notified.",
        study_salt="study-a",
    )

    assert "john.doe@example.com" not in result["anonymized_text"]
    assert "<REDACTED_EMAIL>" in result["anonymized_text"]


def test_research_profile_shifts_dates_and_removes_patient_context_name():
    result = anonymize_clinical_text(
        "Patient John Doe, MRN 123456, visited on 2026-06-16.",
        profile="research",
        study_salt="study-a",
    )

    assert result["privacy_profile"] == "research"
    assert result["date_strategy"] == "shift"
    assert result["text_identifier_strategy"] == "pseudonymize"
    assert "John Doe" not in result["anonymized_text"]
    assert "123456" not in result["anonymized_text"]
    assert "2026-06-16" not in result["anonymized_text"]
    assert "<REDACTED_DATE>" not in result["anonymized_text"]
    assert result["detected_entities"]["PERSON"] == 1
    assert result["detected_entities"]["DATE_TIME"] == 1


def test_strict_profile_redacts_dates():
    result = anonymize_clinical_text(
        "Patient John Doe, MRN 123456, visited on 2026-06-16.",
        profile="strict",
        study_salt="study-a",
    )

    assert result["privacy_profile"] == "strict"
    assert result["date_strategy"] == "redact"
    assert result["text_identifier_strategy"] == "redact"
    assert "<REDACTED_NAME>" in result["anonymized_text"]
    assert "<REDACTED_MRN>" in result["anonymized_text"]
    assert "2026-06-16" not in result["anonymized_text"]
    assert "<REDACTED_DATE>" in result["anonymized_text"]

