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


from services import ner_phi_detector  # noqa: E402
from services.ingestion import anonymize_text  # noqa: E402
from services.ner_phi_detector import (  # noqa: E402
    DEFAULT_NER_MODEL,
    SPACY_PHI_LABEL_MAP,
)
from services.phi_detection import (  # noqa: E402
    SOURCE_CONTEXT_RULE,
    SOURCE_NER,
    SOURCE_STRICT_PROPER_NOUN,
    SOURCE_STRUCTURED_PATTERN,
    DetectedEntity,
    resolve_overlaps,
)
from services.text_anonymization import (  # noqa: E402
    MAX_TEXT_BYTES,
    TextAnonymizationError,
    anonymize_clinical_text,
    stable_hash,
)


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Patient Rahul Sharma was admitted.", "Rahul Sharma"),
        ("Dr. Amit Verma reviewed the scan.", "Amit Verma"),
        ("Mr. John Smith arrived.", "John Smith"),
        ("Mrs. Priya Nair called.", "Priya Nair"),
        ("Ms. Aisha Khan consented.", "Aisha Khan"),
        ("Professor Sarah Johnson consulted.", "Sarah Johnson"),
        ("Name: Arjun Mehta", "Arjun Mehta"),
        ("Attending physician: Maria Garcia", "Maria Garcia"),
    ],
)
def test_contextual_person_names_are_replaced(text, name):
    result = anonymize_clinical_text(text, study_salt="study-a")

    assert name not in result["anonymized_text"]
    assert SURROGATE_RE.search(result["anonymized_text"])
    assert result["detected_entities"]["PERSON"] == 1
    assert result["detection_sources"][SOURCE_CONTEXT_RULE] == 1


def test_arbitrary_name_failure_case_is_now_safe():
    result = anonymize_clinical_text(
        "Patient Rahul Sharma was admitted for treatment.",
        study_salt="study-a",
    )

    assert result["anonymized_text"] == (
        "Patient PATIENT_001 was admitted for treatment."
    )
    assert result["trained_ner_active"] is True
    assert result["ner_model"] == DEFAULT_NER_MODEL


def test_strict_profile_redacts_each_repeated_and_distinct_person():
    text = (
        "Rahul Sharma was examined by Dr. Amit Verma. "
        "Rahul Sharma was discharged later."
    )
    result = anonymize_clinical_text(text, study_salt="study-a")
    assert surrogate_count(result["anonymized_text"]) == 3
    assert result["detected_entities"]["PERSON"] == 3
    assert "Rahul Sharma" not in result["anonymized_text"]
    assert "Amit Verma" not in result["anonymized_text"]


def test_strict_person_redaction_does_not_depend_on_study_salt():
    text = "Patient Rahul Sharma was admitted."
    first = anonymize_clinical_text(text, study_salt="study-a")
    second = anonymize_clinical_text(text, study_salt="study-b")

    assert first["anonymized_text"] == second["anonymized_text"]
    assert SURROGATE_RE.search(first["anonymized_text"])


def test_surrogate_digest_includes_entity_type():
    assert stable_hash(
        "ABC12345", "study-a", entity_type="PATIENT_ID"
    ) != stable_hash(
        "ABC12345", "study-a", entity_type="MEDICAL_RECORD_NUMBER"
    )


def test_spacy_label_mapping_is_centralized_and_privacy_first():
    assert SPACY_PHI_LABEL_MAP == {
        "PERSON": "PERSON",
        "ORG": "ORGANIZATION",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "FAC": "FACILITY",
        "DATE": "DATE_TIME",
        "TIME": "TIME",
    }


def test_structured_patterns_cover_identifiers_and_network_values():
    text = (
        "MRN-123456 Patient ID PT-1001 Insurance ID ABC123456789 "
        "Accession No ACC-445566 Device ID DEV-998877 "
        "email jane@example.com phone 555-123-4567 SSN 123-45-6789 "
        "URL https://example.org/record IP 192.168.1.20 date 2026-08-03."
    )
    result = anonymize_clinical_text(text, study_salt="study-a")

    for raw_value in (
        "123456",
        "PT-1001",
        "ABC123456789",
        "ACC-445566",
        "DEV-998877",
        "jane@example.com",
        "555-123-4567",
        "123-45-6789",
        "https://example.org/record",
        "192.168.1.20",
        "2026-08-03",
    ):
        assert raw_value not in result["anonymized_text"]

    assert result["detection_sources"][SOURCE_STRUCTURED_PATTERN] == 11


def test_overlap_resolution_prefers_context_then_ner_and_longer_span():
    entities = [
        DetectedEntity("PERSON", 4, 14, SOURCE_NER, original_label="PERSON"),
        DetectedEntity(
            "PERSON", 4, 14, SOURCE_CONTEXT_RULE, original_label="TITLE_PERSON"
        ),
        DetectedEntity("ORGANIZATION", 20, 28, SOURCE_NER, original_label="ORG"),
        DetectedEntity("ORGANIZATION", 20, 30, SOURCE_NER, original_label="ORG"),
    ]

    selected = resolve_overlaps(entities, text_length=40)

    assert selected == [entities[1], entities[3]]


def test_exact_identifier_wins_overlap_with_broad_date():
    structured = DetectedEntity(
        "MEDICAL_RECORD_NUMBER",
        0,
        10,
        SOURCE_STRUCTURED_PATTERN,
        original_label="medical_record_number",
    )
    broad_date = DetectedEntity("DATE", 4, 10, SOURCE_NER, original_label="DATE")

    assert resolve_overlaps([broad_date, structured], text_length=10) == [structured]


def test_missing_model_fails_closed_without_returning_original(monkeypatch):
    monkeypatch.setenv(ner_phi_detector.NER_MODEL_ENV_VAR, "missing_phi_model")

    with pytest.raises(TextAnonymizationError) as exc:
        anonymize_clinical_text(
            "Patient Rahul Sharma was admitted.",
            study_salt="study-a",
        )

    assert exc.value.status_code == 503
    assert exc.value.detail == "ner_model_unavailable"


def test_model_path_configuration_is_rejected_without_exposure(monkeypatch):
    local_path = r"C:\private\models\phi"
    monkeypatch.setenv(ner_phi_detector.NER_MODEL_ENV_VAR, local_path)

    with pytest.raises(TextAnonymizationError) as exc:
        anonymize_clinical_text("Patient Rahul Sharma", study_salt="study-a")

    assert exc.value.detail == "ner_model_unavailable"
    assert local_path not in exc.value.detail


def test_inference_failure_is_safe(monkeypatch):
    class BrokenPipeline:
        def __call__(self, text):
            raise RuntimeError("sensitive local inference detail")

    monkeypatch.setattr(
        ner_phi_detector,
        "load_spacy_pipeline",
        lambda model_name: BrokenPipeline(),
    )

    with pytest.raises(TextAnonymizationError) as exc:
        anonymize_clinical_text("Patient Rahul Sharma", study_salt="study-a")

    assert exc.value.status_code == 500
    assert exc.value.detail == "ner_inference_failed"


def test_direct_text_limit_uses_utf8_byte_length():
    oversized_unicode_text = "é" * ((MAX_TEXT_BYTES // 2) + 1)

    with pytest.raises(TextAnonymizationError) as exc:
        anonymize_clinical_text(oversized_unicode_text, study_salt="study-a")

    assert exc.value.status_code == 413


def test_no_entities_is_distinct_from_model_failure():
    text = "Symptoms improved after rest."
    result = anonymize_clinical_text(text, study_salt="study-a")

    assert result["anonymization_status"] == "completed"
    assert result["entity_count"] == 0
    assert result["detected_entities"] == {}
    assert result["detection_sources"] == {}
    assert result["trained_ner_active"] is True
    assert result["anonymized_text"] == text


def test_safe_result_contains_counts_but_not_raw_phi():
    raw_name = "Rahul Sharma"
    result = anonymize_clinical_text(
        f"Patient {raw_name} has MRN: 123456.",
        study_salt="study-a",
    )

    assert result["entity_count"] == 2
    assert result["detected_entities"] == {
        "PERSON": 1,
        "MEDICAL_RECORD_NUMBER": 1,
    }
    assert raw_name not in repr(result)
    assert "123456" not in repr(result)


def test_ingestion_handler_exposes_only_safe_ner_metadata():
    result = anonymize_text(
        b"Patient Rahul Sharma was admitted.",
        profile="strict",
        study_salt="study-a",
    )

    assert result["anonymization_status"] == "completed"
    assert result["entity_count"] == 1
    assert result["detected_entities"] == {"PERSON": 1}
    assert result["detection_sources"] == {SOURCE_CONTEXT_RULE: 1}
    assert result["ner_model"] == DEFAULT_NER_MODEL
    assert result["trained_ner_active"] is True
    assert "Rahul Sharma" not in repr(result)


def test_trained_ner_redacts_privacy_sensitive_context_but_not_clinical_terms():
    text = (
        "Patient Rahul Sharma was treated at Alaska Regional Hospital in London "
        "on March 3, 2026 at 10:30 AM for diabetes with aspirin and CT scan."
    )
    result = anonymize_clinical_text(text, study_salt="study-a")

    assert "Rahul Sharma" not in result["anonymized_text"]
    assert "Alaska Regional Hospital" not in result["anonymized_text"]
    assert "London" not in result["anonymized_text"]
    assert "March 3, 2026" not in result["anonymized_text"]
    assert "10:30 AM" not in result["anonymized_text"]
    assert "diabetes with aspirin and CT scan" in result["anonymized_text"]


def test_invalid_input_type_is_rejected():
    with pytest.raises(TextAnonymizationError) as exc:
        anonymize_clinical_text(b"Patient Rahul Sharma", study_salt="study-a")

    assert exc.value.status_code == 400


@pytest.mark.parametrize("profile", ["strict", "research"])
@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Kartik is suffering fever.", "Kartik"),
        ("Rose has pain.", "Rose"),
        ("Nikhil was diagnosed with influenza.", "Nikhil"),
    ],
)
def test_clinical_subject_names_are_redacted_in_both_profiles(
    profile,
    text,
    name,
):
    result = anonymize_clinical_text(
        text,
        profile=profile,
        study_salt="study-a",
    )

    assert name not in result["anonymized_text"]
    assert result["detected_entities"] == {"PERSON": 1}
    assert result["detection_sources"] == {SOURCE_CONTEXT_RULE: 1}


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Kartik went home after lunch.", "Kartik"),
        ("The clinician spoke with Kartik.", "Kartik"),
        ("After lunch, Kartik returned.", "Kartik"),
    ],
)
def test_strict_profile_removes_names_from_arbitrary_positions(text, name):
    result = anonymize_clinical_text(
        text,
        profile="strict",
        study_salt="study-a",
    )

    assert name not in result["anonymized_text"]
    assert result["entity_count"] == 1


def test_strict_profile_records_proper_noun_fallback_when_model_misses_name():
    result = anonymize_clinical_text(
        "Kartik went home after lunch.",
        profile="strict",
        study_salt="study-a",
    )

    assert SURROGATE_RE.match(result["anonymized_text"])
    assert result["detection_sources"] == {SOURCE_STRICT_PROPER_NOUN: 1}


def test_research_profile_does_not_apply_strict_proper_noun_fallback():
    text = "Kartik went home after lunch."
    result = anonymize_clinical_text(
        text,
        profile="research",
        study_salt="study-a",
    )

    assert result["anonymized_text"] == text
    assert result["detected_entities"] == {}


def test_strict_fallback_preserves_known_non_name_medical_terms():
    text = "Diabetes is causing fatigue. MRI was reviewed."
    result = anonymize_clinical_text(
        text,
        profile="strict",
        study_salt="study-a",
    )

    assert result["anonymized_text"] == text
    assert result["detected_entities"] == {}


def test_contextual_patient_name_overrides_eponym_safeguard():
    result = anonymize_clinical_text(
        "Patient Hodgkin was admitted for observation.",
        study_salt="study-a",
    )

    assert "Hodgkin" not in result["anonymized_text"]
    assert result["detected_entities"] == {"PERSON": 1}


def test_date_shaped_ner_mislabel_does_not_override_structured_date():
    result = anonymize_clinical_text(
        "Patient Rahul Sharma was admitted on 03/08/2026.",
        study_salt="study-a",
    )

    assert "Rahul Sharma" not in result["anonymized_text"]
    assert "03/08/2026" not in result["anonymized_text"]
    assert "<REDACTED_DATE>" in result["anonymized_text"]
    assert result["detected_entities"]["DATE_TIME"] == 1


class _FakeNerDetector:
    def __init__(self, entities):
        self.entities = list(entities)

    def detect(self, text):
        return list(self.entities)


def _entity_for(text, value, source=SOURCE_NER, occurrence=0):
    import re

    starts = [match.start() for match in re.finditer(re.escape(value), text)]
    start = starts[occurrence]
    return DetectedEntity(
        entity_type="PERSON",
        start=start,
        end=start + len(value),
        source=source,
        original_label="PERSON",
    )


def _anonymize_with_fake(monkeypatch, text, entities):
    from services import text_anonymization as text_service
    from services.phi_detection import StructuredPatternDetector

    monkeypatch.setattr(
        text_service,
        "_detectors",
        lambda model_name, profile: (
            StructuredPatternDetector(),
            _FakeNerDetector(entities),
        ),
    )
    return anonymize_clinical_text(text, study_salt="study-a")


def test_clinical_verb_rule_covers_model_missed_name():
    result = anonymize_clinical_text(
        "Dr. Amit Verma treated Priya Nair.",
        study_salt="study-a",
    )

    assert "Amit Verma" not in result["anonymized_text"]
    assert "Priya Nair" not in result["anonymized_text"]
    assert result["detected_entities"] == {"PERSON": 2}


def test_serialized_result_does_not_contain_detected_values(monkeypatch):
    import json

    text = "Synthetic Person has MRN-458921 and synthetic@example.com."
    result = _anonymize_with_fake(
        monkeypatch,
        text,
        [_entity_for(text, "Synthetic Person")],
    )
    serialized = json.dumps(result, sort_keys=True)

    assert "Synthetic Person" not in serialized
    assert "458921" not in serialized
    assert "synthetic@example.com" not in serialized
    assert '"start"' not in serialized
    assert '"end"' not in serialized


def test_duplicate_overlapping_detections_replace_once(monkeypatch):
    text = "Dr. Amit Verma was examined."
    entities = [
        _entity_for(text, "Amit"),
        _entity_for(text, "Amit Verma"),
        _entity_for(text, "Amit Verma", source=SOURCE_CONTEXT_RULE),
    ]

    result = _anonymize_with_fake(monkeypatch, text, entities)

    assert surrogate_count(result["anonymized_text"]) == 1
    assert "Amit" not in result["anonymized_text"]
    assert result["entity_count"] == 1


def test_adjacent_entities_do_not_corrupt_offsets(monkeypatch):
    text = "John SmithSarah Johnson"
    entities = [
        _entity_for(text, "John Smith"),
        _entity_for(text, "Sarah Johnson"),
    ]

    result = _anonymize_with_fake(monkeypatch, text, entities)
    anonymized = result["anonymized_text"]

    # Two adjacent names with no separator: each must get its own surrogate,
    # they must differ, and nothing of the originals may remain.
    assert anonymized == "PATIENT_001PATIENT_002"
    assert "John" not in anonymized and "Sarah" not in anonymized
    assert result["entity_count"] == 2


def test_no_fake_entities_preserves_non_phi_text(monkeypatch):
    text = "Symptoms improved after hydration and rest."

    result = _anonymize_with_fake(monkeypatch, text, [])

    assert result["anonymized_text"] == text
    assert result["entity_count"] == 0
