import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from services.ner_phi_detector import (
    NerPhiDetectionError,
    SpacyNerPhiDetector,
    configured_model_name,
)
from services.phi_detection import (
    DetectedEntity,
    PhiDetector,
    StructuredPatternDetector,
    resolve_overlaps,
)

SUPPORTED_PROFILES = {"strict", "research"}
STUDY_SALT_ENV_VAR = "BIOBLOCK_STUDY_SALT"
HASH_LENGTH = 8
MAX_TEXT_BYTES = 256 * 1024

_ID_ENTITY_TYPES = {
    "MEDICAL_RECORD_NUMBER",
    "PATIENT_ID",
    "HEALTH_PLAN_ID",
    "INSURANCE_ID",
    "ACCESSION_NUMBER",
    "DEVICE_ID",
}
_IDENTIFIER_AT_END = re.compile(r"([A-Z0-9][A-Z0-9-]{3,30})\b", re.IGNORECASE)


class TextAnonymizationError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def stable_hash(
    value: str,
    salt: str,
    length: int = HASH_LENGTH,
    entity_type: str = "VALUE",
) -> str:
    normalized = value.strip().lower()
    digest = hashlib.sha256(
        f"{salt}:{entity_type}:{normalized}".encode("utf-8")
    )
    return digest.hexdigest().upper()[:length]


def pseudonymize_person(value: str, salt: str) -> str:
    return f"PERSON_{stable_hash(value, salt, entity_type='PERSON')}"


def pseudonymize_mrn(value: str, salt: str) -> str:
    return f"MRN_{stable_hash(value, salt, entity_type='MEDICAL_RECORD_NUMBER')}"


def pseudonymize_patient_id(value: str, salt: str) -> str:
    return f"PATIENT_ID_{stable_hash(value, salt, entity_type='PATIENT_ID')}"


def pseudonymize_health_plan(value: str, salt: str) -> str:
    return f"HEALTH_PLAN_{stable_hash(value, salt, entity_type='HEALTH_PLAN_ID')}"


def pseudonymize_accession(value: str, salt: str) -> str:
    return f"ACCESSION_{stable_hash(value, salt, entity_type='ACCESSION_NUMBER')}"


def pseudonymize_device(value: str, salt: str) -> str:
    return f"DEVICE_{stable_hash(value, salt, entity_type='DEVICE_ID')}"


def _normalize_profile(profile: str) -> str:
    normalized = (profile or "").strip().lower()
    if normalized not in SUPPORTED_PROFILES:
        raise TextAnonymizationError(
            "Invalid privacy profile. Supported profiles: strict, research"
        )
    return normalized


@lru_cache(maxsize=4)
def _detectors(model_name: str) -> Tuple[PhiDetector, ...]:
    return (
        StructuredPatternDetector(),
        SpacyNerPhiDetector(model_name),
    )


def _detect_entities(text: str, model_name: str) -> List[DetectedEntity]:
    detected: List[DetectedEntity] = []
    try:
        for detector in _detectors(model_name):
            detected.extend(detector.detect(text))
    except NerPhiDetectionError:
        raise
    except Exception as exc:
        raise NerPhiDetectionError("phi_detection_failed", status_code=500) from exc
    return resolve_overlaps(detected, len(text))


def _hash_value(entity_type: str, value: str) -> str:
    if entity_type not in _ID_ENTITY_TYPES:
        return value

    match = _IDENTIFIER_AT_END.search(value.strip())
    if match:
        return match.group(1)
    return value


def _replacement_for(entity_type: str, value: str, salt: str) -> str:
    hash_value = _hash_value(entity_type, value)
    if entity_type == "PERSON":
        return pseudonymize_person(hash_value, salt)
    if entity_type == "MEDICAL_RECORD_NUMBER":
        return pseudonymize_mrn(hash_value, salt)
    if entity_type == "PATIENT_ID":
        return pseudonymize_patient_id(hash_value, salt)
    if entity_type in {"HEALTH_PLAN_ID", "INSURANCE_ID"}:
        return pseudonymize_health_plan(hash_value, salt)
    if entity_type == "ACCESSION_NUMBER":
        return pseudonymize_accession(hash_value, salt)
    if entity_type == "DEVICE_ID":
        return pseudonymize_device(hash_value, salt)
    if entity_type == "EMAIL_ADDRESS":
        return "<REDACTED_EMAIL>"
    if entity_type == "PHONE_NUMBER":
        return "<REDACTED_PHONE>"
    if entity_type in {"US_SSN", "SSN"}:
        return "<REDACTED_SSN>"
    if entity_type in {"DATE", "DATE_TIME"}:
        return "<REDACTED_DATE>"
    if entity_type == "TIME":
        return "<REDACTED_TIME>"
    if entity_type == "URL":
        return "<REDACTED_URL>"
    if entity_type == "IP_ADDRESS":
        return "<REDACTED_IP_ADDRESS>"
    return f"<REDACTED_{entity_type}>"


def _replace_entities(
    text: str,
    entities: Iterable[DetectedEntity],
    salt: str,
) -> str:
    anonymized_text = text
    for entity in sorted(entities, key=lambda item: item.start, reverse=True):
        original_value = text[entity.start : entity.end]
        replacement = _replacement_for(entity.entity_type, original_value, salt)
        anonymized_text = (
            anonymized_text[: entity.start]
            + replacement
            + anonymized_text[entity.end :]
        )
    return anonymized_text


def _entity_summary(entities: Iterable[DetectedEntity]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for entity in entities:
        summary[entity.entity_type] = summary.get(entity.entity_type, 0) + 1
    return summary


def _source_summary(entities: Iterable[DetectedEntity]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for entity in entities:
        summary[entity.source] = summary.get(entity.source, 0) + 1
    return summary


def _read_local_env_salt() -> str | None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#") or "=" not in cleaned:
            continue
        key, value = cleaned.split("=", 1)
        if key.strip() == STUDY_SALT_ENV_VAR:
            return value.strip().strip("\"'")

    return None


def _resolve_study_salt(study_salt: str | None) -> str:
    configured_salt = (
        study_salt
        or os.getenv(STUDY_SALT_ENV_VAR)
        or _read_local_env_salt()
    )
    if configured_salt and configured_salt.strip():
        return configured_salt.strip()

    raise TextAnonymizationError(
        f"Text anonymization salt is not configured. Set {STUDY_SALT_ENV_VAR}.",
        status_code=500,
    )


def anonymize_clinical_text(
    text: str,
    profile: str = "strict",
    study_salt: str | None = None,
) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise TextAnonymizationError("Text input must be a string")
    if not text.strip():
        raise TextAnonymizationError("Text input is empty")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise TextAnonymizationError(
            f"Text input exceeds the {MAX_TEXT_BYTES} byte limit",
            status_code=413,
        )

    _normalize_profile(profile)
    salt = _resolve_study_salt(study_salt)

    try:
        model_name = configured_model_name()
        entities = _detect_entities(text, model_name)
    except NerPhiDetectionError as exc:
        raise TextAnonymizationError(
            exc.error_code,
            status_code=exc.status_code,
        ) from exc

    return {
        "anonymization_status": "completed",
        "anonymized_text": _replace_entities(text, entities, salt),
        "entity_count": len(entities),
        "detected_entities": _entity_summary(entities),
        "detection_sources": _source_summary(entities),
        "ner_model": model_name,
        "trained_ner_active": True,
    }
