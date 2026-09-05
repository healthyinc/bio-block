import os
import re
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Phase 10: identifiers are replaced with consistent study-local surrogates
# (PATIENT_001, PROVIDER_001, RECORD_001) rather than a single fixed
# placeholder, so coreference survives. The privacy assertions below are
# unchanged: the original value must still be absent.
SURROGATE_RE = re.compile(
    r"\b(?:PATIENT|PROVIDER|FACILITY|ORG|PLACE|ADDRESS|RECORD|PATIENTID|"
    r"PLAN|ACCESSION|DEVICE|IDENTIFIER|USER)_\d{3,}"
)


def surrogate_count(text):
    return len(SURROGATE_RE.findall(text))


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
    assert SURROGATE_RE.search(result["anonymized_text"])
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
    assert SURROGATE_RE.search(result["anonymized_text"])
    assert result["detected_entities"]["PATIENT_ID"] == 1


def test_health_plan_id_is_detected_and_replaced():
    result = anonymize_clinical_text(
        "Insurance ID ABC123456789 was verified.",
        study_salt="study-a",
    )

    assert "ABC123456789" not in result["anonymized_text"]
    assert SURROGATE_RE.search(result["anonymized_text"])
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
        anonymize_clinical_text(
            "Patient has MRN: 123456.",
            profile="research",
        )

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
    assert SURROGATE_RE.search(result["anonymized_text"])
    assert SURROGATE_RE.search(result["anonymized_text"])
    assert "2026-06-16" not in result["anonymized_text"]
    assert "<REDACTED_DATE>" in result["anonymized_text"]


class _NoopNerDetector:
    def detect(self, text):
        return []


@pytest.mark.parametrize(
    ("text", "raw_value", "entity_type", "replacement"),
    [
        ("MRN-458921 was verified.", "458921", "MEDICAL_RECORD_NUMBER", "RECORD_001"),
        ("Patient ID PT-1001 was verified.", "PT-1001", "PATIENT_ID", "PATIENTID_001"),
        (
            "Health Plan ID PLAN-998877 was verified.",
            "PLAN-998877",
            "HEALTH_PLAN_ID",
            "PLAN_001",
        ),
        (
            "Accession Number ACC-445566 was verified.",
            "ACC-445566",
            "ACCESSION_NUMBER",
            "ACCESSION_001",
        ),
        ("Device ID DEV-998877 was verified.", "DEV-998877", "DEVICE_ID", "DEVICE_001"),
        (
            "Email synthetic.person@example.com was verified.",
            "synthetic.person@example.com",
            "EMAIL_ADDRESS",
            "<REDACTED_EMAIL>",
        ),
        ("Phone 555-321-7654 was verified.", "555-321-7654", "PHONE_NUMBER", "<REDACTED_PHONE>"),
        ("SSN 123-45-6789 was verified.", "123-45-6789", "US_SSN", "<REDACTED_SSN>"),
        ("Visit date 03/08/2026 was verified.", "03/08/2026", "DATE_TIME", "<REDACTED_DATE>"),
    ],
)
def test_structured_identifier_redaction_does_not_depend_on_ner(
    monkeypatch,
    text,
    raw_value,
    entity_type,
    replacement,
):
    from services.phi_detection import StructuredPatternDetector

    monkeypatch.setattr(
        text_anonymization,
        "_detectors",
        lambda model_name, profile: (
            StructuredPatternDetector(),
            _NoopNerDetector(),
        ),
    )

    result = anonymize_clinical_text(text, study_salt="study-a")

    assert raw_value not in result["anonymized_text"]
    assert replacement in result["anonymized_text"]
    assert result["detected_entities"] == {entity_type: 1}
    assert result["detection_sources"] == {"structured_pattern": 1}



# ---------------------------------------------------------------------------
# Local model adapters are detectors only, and fail closed (Phase 3)
# ---------------------------------------------------------------------------


def _model_detector_names(model_mode):
    from services import local_model_detectors as models

    text_anonymization._build_detectors.cache_clear()
    try:
        detectors = text_anonymization._build_detectors(
            "en_core_web_sm", "strict", model_mode
        )
        return [detector.__class__.__name__ for detector in detectors]
    finally:
        text_anonymization._build_detectors.cache_clear()
        assert models.SUPPORTED_MODEL_MODES  # keep the import meaningful


def test_offline_mode_wires_both_local_model_detectors():
    from services import local_model_detectors as models

    names = _model_detector_names(models.MODE_OFFLINE)

    assert "StanfordClinicalDetector" in names
    assert "GlinerPiiDetector" in names
    assert names[0] == "StructuredPatternDetector"


def test_legacy_test_mode_excludes_the_local_model_detectors():
    from services import local_model_detectors as models

    names = _model_detector_names(models.MODE_LEGACY_TEST)

    assert "StanfordClinicalDetector" not in names
    assert "GlinerPiiDetector" not in names


def test_unknown_model_mode_blocks_anonymization(monkeypatch):
    from services import local_model_detectors as models

    monkeypatch.setenv(models.MODEL_MODE_ENV_VAR, "cloud")

    with pytest.raises(TextAnonymizationError) as exc:
        anonymize_clinical_text("Patient MRN: 123456.", study_salt="study-a")

    assert exc.value.detail == "invalid_model_configuration"
    assert exc.value.status_code == 500


@pytest.mark.parametrize(
    "error_code,status_code",
    [
        ("model_checksum_mismatch", 503),
        ("model_files_unavailable", 503),
        ("model_inference_timeout", 503),
        ("stanford_inference_failed", 500),
    ],
)
def test_model_failure_blocks_instead_of_returning_text(
    monkeypatch, error_code, status_code
):
    from services.local_model_detectors import LocalModelError

    class BlowingUpDetector:
        def detect(self, text):
            raise LocalModelError(error_code, status_code)

    monkeypatch.setattr(
        text_anonymization,
        "_detectors",
        lambda model_name, profile: (BlowingUpDetector(),),
    )

    with pytest.raises(TextAnonymizationError) as exc:
        anonymize_clinical_text("Patient MRN: 123456.", study_salt="study-a")

    assert exc.value.detail == error_code
    assert exc.value.status_code == status_code
    assert "123456" not in str(exc.value)


def test_model_failure_blocks_detection_only_endpoint(monkeypatch):
    from services.local_model_detectors import LocalModelError
    from services.text_anonymization import detect_clinical_phi

    class BlowingUpDetector:
        def detect(self, text):
            raise LocalModelError("model_checksum_mismatch")

    monkeypatch.setattr(
        text_anonymization,
        "_detectors",
        lambda model_name, profile: (BlowingUpDetector(),),
    )

    with pytest.raises(TextAnonymizationError) as exc:
        detect_clinical_phi("Patient MRN: 123456.")

    assert exc.value.detail == "model_checksum_mismatch"


def test_model_candidates_are_redacted_not_released(monkeypatch):
    from services.local_model_detectors import SOURCE_GLINER
    from services.privacy_contracts import PhiEntity

    text = "Contact Ravi Kumar for details."

    class FakeModelDetector:
        def detect(self, _text):
            return [
                PhiEntity(
                    entity_type="PERSON",
                    start=8,
                    end=18,
                    source=SOURCE_GLINER,
                    score=0.42,
                    original_label="person",
                )
            ]

    monkeypatch.setattr(
        text_anonymization,
        "_detectors",
        lambda model_name, profile: (FakeModelDetector(),),
    )

    result = anonymize_clinical_text(text, study_salt="study-a")

    assert "Ravi Kumar" not in result["anonymized_text"]
    assert SURROGATE_RE.search(result["anonymized_text"])
    assert result["detection_sources"] == {SOURCE_GLINER: 1}
