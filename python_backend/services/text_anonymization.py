import hashlib
import os
import re
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from services.ner_phi_detector import (
    NerPhiDetectionError,
    SpacyNerPhiDetector,
    configured_model_name,
)
from services.local_model_detectors import (
    MODE_OFFLINE,
    SOURCE_GLINER,
    SOURCE_STANFORD,
    GlinerPiiDetector,
    LocalModelError,
    StanfordClinicalDetector,
    calibrated_config_for,
    resolve_model_mode,
)
from services.phi_detection import (
    SOURCE_STRICT_PROPER_NOUN,
    AgeOverThresholdDetector,
    DetectedEntity,
    PhiDetector,
    StructuredPatternDetector,
    resolve_overlaps,
)
from services.clinical_vocabulary import protects_from_person_label
from services.surrogates import (
    SURROGATE_PATTERN,
    SurrogateAllocator,
    looks_like_provider,
)
from services.model_client import RemoteModelDetector, worker_enabled
from services.text_utility import measure_text_utility

from services.privacy_profiles import (
    PrivacyProfileError,
    get_privacy_profile,
    validate_privacy_profile,
)

SUPPORTED_PROFILES = {"strict", "research"}
STUDY_SALT_ENV_VAR = "BIOBLOCK_STUDY_SALT"
HASH_LENGTH = 8
MAX_TEXT_BYTES = 256 * 1024
#: Safe Harbor aggregation for an age above 89.
AGE_AGGREGATE_REPLACEMENT = "90+"

_ID_ENTITY_TYPES = {
    "MEDICAL_RECORD_NUMBER",
    "PATIENT_ID",
    "HEALTH_PLAN_ID",
    "INSURANCE_ID",
    "ACCESSION_NUMBER",
    "DEVICE_ID",
}
_IDENTIFIER_AT_END = re.compile(r"([A-Z0-9][A-Z0-9-]{3,30})\b", re.IGNORECASE)
DIRECT_IDENTIFIER_REDACTIONS = {
    "PERSON": "<REDACTED_NAME>",
    "MEDICAL_RECORD_NUMBER": "<REDACTED_MRN>",
    "PATIENT_ID": "<REDACTED_PATIENT_ID>",
    "HEALTH_PLAN_ID": "<REDACTED_HEALTH_PLAN>",
    "INSURANCE_ID": "<REDACTED_HEALTH_PLAN>",
    "ACCESSION_NUMBER": "<REDACTED_ACCESSION>",
    "DEVICE_ID": "<REDACTED_DEVICE_ID>",
}

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
    """Build a research-profile compatibility surrogate, not a Safe Harbor result."""
    normalized = value.strip().lower()
    digest = hashlib.sha256(
        f"{salt}:{entity_type}:{normalized}".encode("utf-8")
    )
    return digest.hexdigest().upper()[:length]


def pseudonymize_person(value: str, salt: str) -> str:
    """Research-only compatibility helper; strict processing never calls this."""
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


def _profile_settings(profile: str) -> tuple[str, Dict[str, Any]]:
    try:
        normalized = validate_privacy_profile(profile)
        return normalized, get_privacy_profile(normalized)
    except PrivacyProfileError as exc:
        raise TextAnonymizationError(exc.detail, status_code=exc.status_code) from exc


def _normalize_profile(profile: str) -> str:
    return _profile_settings(profile)[0]


def _detectors(model_name: str, profile: str) -> Tuple[PhiDetector, ...]:
    """Build the detector chain for the currently configured model mode.

    An unrecognized mode raises ``LocalModelError`` rather than silently
    dropping the model adapters.
    """
    return _build_detectors(model_name, profile, resolve_model_mode())


@lru_cache(maxsize=8)
def _build_detectors(
    model_name: str,
    profile: str,
    model_mode: str,
) -> Tuple[PhiDetector, ...]:
    """Cached detector chain. Model adapters only propose spans."""
    if model_mode == MODE_OFFLINE:
        # Each model carries its own calibrated threshold. They were selected
        # jointly against the combined chain's recall, not in isolation, so
        # they are not interchangeable.
        #
        # The main API environment cannot hold the model stack without
        # downgrading ChromaDB's pins, so when the worker is enabled inference
        # runs out of process. Both paths apply the same calibration; only the
        # execution boundary differs.
        if worker_enabled():
            stanford: PhiDetector = RemoteModelDetector("stanford", SOURCE_STANFORD)
            gliner: PhiDetector = RemoteModelDetector("gliner", SOURCE_GLINER)
        else:
            stanford = StanfordClinicalDetector(calibrated_config_for(SOURCE_STANFORD))
            gliner = GlinerPiiDetector(calibrated_config_for(SOURCE_GLINER))
        return (
            StructuredPatternDetector(),
            AgeOverThresholdDetector(),
            stanford,
            gliner,
            SpacyNerPhiDetector(
                model_name,
                high_recall_proper_nouns=profile == "strict",
            ),
        )
    return (
        StructuredPatternDetector(),
        AgeOverThresholdDetector(),
        SpacyNerPhiDetector(
            model_name,
            high_recall_proper_nouns=profile == "strict",
        ),
    )


def _detect_entities(
    text: str,
    model_name: str,
    profile: str,
) -> List[DetectedEntity]:
    detected: List[DetectedEntity] = []
    try:
        for detector in _detectors(model_name, profile):
            detected.extend(detector.detect(text))
    except LocalModelError as exc:
        # Model load, checksum, inference, and timeout failures block the
        # artifact; they never degrade to returning unredacted content.
        raise NerPhiDetectionError(exc.error_code, exc.status_code) from exc
    except NerPhiDetectionError:
        raise
    except Exception as exc:
        raise NerPhiDetectionError("phi_detection_failed", status_code=500) from exc

    detected = _drop_clinical_vocabulary(detected, text)
    return _merge_adjacent_same_type(resolve_overlaps(detected, len(text)), text)


#: Labels that a clinical term can be wrongly assigned. A detection of one of
#: these over recorded clinical vocabulary is a misclassification, whichever
#: detector produced it.
_NAME_SHAPED_TYPES = frozenset(
    {"PERSON", "ORGANIZATION", "FACILITY", "LOCATION", "ADDRESS", "MEDICAL_CONDITION"}
)


def _drop_clinical_vocabulary(
    entities: List[DetectedEntity],
    text: str,
) -> List[DetectedEntity]:
    """Remove name-shaped detections that are actually clinical vocabulary.

    Applied to every source, not just spaCy. The pinned models label
    "Parkinson" a PERSON as readily as spaCy does, and filtering only the
    detectors we happen to control leaves the same clinical term destroyed by
    a different route.
    """
    kept: List[DetectedEntity] = []
    for entity in entities:
        if entity.entity_type in _NAME_SHAPED_TYPES:
            span = text[entity.start : entity.end]
            following = re.findall(r"[^\s]+", text[entity.end : entity.end + 64])[:4]
            if protects_from_person_label(span, following):
                continue
        kept.append(entity)
    return kept


#: Only whitespace, a hyphen or a possessive may sit between two fragments of
#: one name. Anything else is two different entities.
_NAME_GAP = re.compile(r"^[\s\-']*$")


def _merge_adjacent_same_type(
    entities: List[DetectedEntity],
    text: str,
) -> List[DetectedEntity]:
    """Join same-type spans split across a name by different detectors.

    Two detectors proposing "Padmavathi" and "Venkataraghavan" separately
    resolve to two adjacent spans, which would then receive two different
    surrogates for one person. Merging them keeps one person as one entity.
    """
    if not entities:
        return entities
    ordered = sorted(entities, key=lambda e: (e.start, e.end))
    merged: List[DetectedEntity] = [ordered[0]]
    for entity in ordered[1:]:
        previous = merged[-1]
        gap = text[previous.end : entity.start]
        if (
            entity.entity_type == previous.entity_type
            and entity.entity_type in _NAME_SHAPED_TYPES
            # A real separator must be present. Two spans that touch with no
            # gap at all are two entities written without a space, not one
            # name that detectors split.
            and 1 <= len(gap) <= 2
            and _NAME_GAP.match(gap)
        ):
            merged[-1] = DetectedEntity(
                entity_type=previous.entity_type,
                start=previous.start,
                end=entity.end,
                source=previous.source,
                score=previous.score,
                original_label=previous.original_label,
            )
            continue
        merged.append(entity)
    return merged


def _hash_value(entity_type: str, value: str) -> str:
    if entity_type not in _ID_ENTITY_TYPES:
        return value

    match = _IDENTIFIER_AT_END.search(value.strip())
    if match:
        return match.group(1)
    return value


def _date_shift_days(salt: str) -> int:
    value = int(stable_hash("date-shift", salt, length=6), 16)
    days = value % 731 - 365
    return days or 17


def _shift_date_text(value: str, salt: str) -> str:
    cleaned = value.strip()
    formats = [
        ("%Y-%m-%d", "%Y-%m-%d"),
        ("%m/%d/%Y", "%m/%d/%Y"),
        ("%m/%d/%y", "%m/%d/%Y"),
    ]
    for input_format, output_format in formats:
        try:
            shifted = datetime.strptime(cleaned, input_format) + timedelta(
                days=_date_shift_days(salt)
            )
            return shifted.strftime(output_format)
        except ValueError:
            continue
    return "<REDACTED_DATE>"

def _replacement_for(
    entity_type: str,
    value: str,
    salt: str,
    settings: Dict[str, Any],
    allocator: "SurrogateAllocator | None" = None,
    is_provider: bool = False,
) -> str:
    # Safe Harbor requires ages over 89 to be aggregated rather than removed.
    # "94 years old" becomes "90+ years old", which keeps the clinical fact.
    if entity_type == "AGE_OVER_89":
        return AGE_AGGREGATE_REPLACEMENT
    if entity_type == "AGE_UNCERTAIN":
        # Cannot be resolved to a number, so it is removed and the document is
        # routed to review rather than guessed at in either direction.
        return "<REDACTED_AGE>"

    if settings["text_identifier_strategy"] == "redact":
        # A consistent study-local surrogate preserves coreference where a
        # fixed placeholder would collapse every mention into one token. Any
        # type with a surrogate form uses one, not only the direct-identifier
        # set, so a facility or organisation stays distinguishable too.
        if allocator is not None:
            surrogate_type = (
                "PERSON_PROVIDER"
                if entity_type == "PERSON" and is_provider
                else entity_type
            )
            surrogate = allocator.surrogate_for(surrogate_type, value)
            if surrogate is not None:
                return surrogate
        if entity_type in DIRECT_IDENTIFIER_REDACTIONS:
            return DIRECT_IDENTIFIER_REDACTIONS[entity_type]

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
        if settings["date_strategy"] == "shift":
            return _shift_date_text(value, salt)
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
    settings: Dict[str, Any],
    allocator: "SurrogateAllocator | None" = None,
) -> str:
    ordered = sorted(entities, key=lambda item: item.start, reverse=True)

    # Allocate surrogates in reading order so numbering follows first
    # appearance, then apply replacements from the end so earlier offsets stay
    # valid. Without this pass the numbering would run backwards through the
    # document, which is harmless but confusing to a reader.
    if allocator is not None:
        for entity in reversed(ordered):
            surrogate_type = (
                "PERSON_PROVIDER"
                if entity.entity_type == "PERSON"
                and looks_like_provider(text, entity.start)
                else entity.entity_type
            )
            allocator.surrogate_for(surrogate_type, text[entity.start : entity.end])

    anonymized_text = text
    for entity in ordered:
        original_value = text[entity.start : entity.end]
        replacement = _replacement_for(
            entity.entity_type,
            original_value,
            salt,
            settings,
            allocator=allocator,
            is_provider=(
                entity.entity_type == "PERSON"
                and looks_like_provider(text, entity.start)
            ),
        )
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


def detect_clinical_phi(text: str) -> Dict[str, int]:
    """Return safe entity counts without returning source text or offsets."""
    if not isinstance(text, str):
        raise TextAnonymizationError("Text input must be a string")
    if not text.strip():
        return {}
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise TextAnonymizationError(
            f"Text input exceeds the {MAX_TEXT_BYTES} byte limit",
            status_code=413,
        )

    try:
        entities = _detect_entities(text, configured_model_name(), "strict")
    except NerPhiDetectionError as exc:
        raise TextAnonymizationError(
            exc.error_code,
            status_code=exc.status_code,
        ) from exc
    return _entity_summary(entities)


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

    privacy_profile, settings = _profile_settings(profile)
    if settings["text_identifier_strategy"] == "redact":
        salt = ""
    else:
        salt = _resolve_study_salt(study_salt)

    try:
        model_name = configured_model_name()
        entities = _detect_entities(text, model_name, privacy_profile)
    except NerPhiDetectionError as exc:
        raise TextAnonymizationError(
            exc.error_code,
            status_code=exc.status_code,
        ) from exc

    # One allocator per call. It dies with the request, so two studies
    # containing the same person get unrelated surrogates and no cross-study
    # linkage is created. The mapping is never returned or persisted.
    allocator = (
        SurrogateAllocator()
        if settings["text_identifier_strategy"] == "redact"
        else None
    )
    anonymized_text = _replace_entities(text, entities, salt, settings, allocator)

    # An age reference that cannot be resolved to a number cannot be judged
    # against the Safe Harbor threshold, so the document needs a human.
    review_reasons = sorted(
        {
            "uncertain_age_reference"
            for entity in entities
            if entity.entity_type == "AGE_UNCERTAIN"
        }
    )

    # Measured here because this is the only place that holds both the
    # original spans and the output. The removed values are needed to score
    # utility honestly - deleting PHI is the goal, not a utility loss - and
    # they never leave this function: only ratios and counts are returned.
    utility_metrics = measure_text_utility(
        text,
        anonymized_text,
        redacted_values=[text[e.start : e.end] for e in entities],
    )

    return {
        "anonymization_status": "completed",
        "privacy_profile": privacy_profile,
        "utility_metrics": utility_metrics,
        "date_strategy": settings["date_strategy"],
        "text_identifier_strategy": settings["text_identifier_strategy"],
        "anonymized_text": anonymized_text,
        "entity_count": len(entities),
        "detected_entities": _entity_summary(entities),
        "detection_sources": _source_summary(entities),
        # Counts of distinct entities replaced, per surrogate kind. Never the
        # mapping and never an original value.
        "surrogate_counts": allocator.counts() if allocator else {},
        "review_required_reasons": review_reasons,
        "ner_model": model_name,
        "trained_ner_active": True,
    }




# ---------------------------------------------------------------------------
# Post-redaction validation (Phase 4)
# ---------------------------------------------------------------------------

# Placeholders and research-profile surrogates are our own output, not PHI.
# They are masked out before the residual scan so the validator measures what
# survived redaction rather than re-flagging the redaction itself.
_PLACEHOLDER_PATTERN = re.compile(r"<REDACTED_[A-Z0-9_]+>")
_SURROGATE_PATTERN = re.compile(
    r"\b(?:PERSON|MRN|PATIENT_ID|HEALTH_PLAN|ACCESSION|DEVICE)"
    rf"_[0-9A-F]{{{HASH_LENGTH}}}\b"
)


def mask_release_placeholders(text: str) -> str:
    """Blank out our own replacement tokens, preserving offsets.

    Study-local surrogates and the Safe Harbor age aggregate are our own
    output, so the residual scan must not re-detect them as PHI.
    """
    masked = _PLACEHOLDER_PATTERN.sub(lambda match: " " * len(match.group(0)), text)
    masked = _SURROGATE_PATTERN.sub(lambda match: " " * len(match.group(0)), masked)
    masked = SURROGATE_PATTERN.sub(lambda match: " " * len(match.group(0)), masked)
    return masked.replace(AGE_AGGREGATE_REPLACEMENT, " " * len(AGE_AGGREGATE_REPLACEMENT))


def residual_phi_categories(anonymized_text: str) -> Dict[str, int]:
    """Re-scan redacted output. A non-empty result must block the release.

    Returns categories and counts only, never the surviving values.

    The high-recall proper-noun heuristic is excluded from this second pass.
    In strict mode the first pass already ran it and redacted everything it
    flagged, so anything it finds here is an ordinary capitalized word left in
    the surviving prose - masking the placeholders changes the sentence shape
    and exposes words like "Portal" or "Contact" to it. Re-applying it would
    block releases on our own leftovers rather than on surviving PHI. Every
    evidence-based detector (structured patterns, context rules, and the
    trained NER models) still counts.
    """
    if not isinstance(anonymized_text, str):
        raise TextAnonymizationError("Text input must be a string")
    masked = mask_release_placeholders(anonymized_text)
    if not masked.strip():
        return {}
    if len(masked.encode("utf-8")) > MAX_TEXT_BYTES:
        raise TextAnonymizationError(
            f"Text input exceeds the {MAX_TEXT_BYTES} byte limit",
            status_code=413,
        )
    try:
        entities = _detect_entities(masked, configured_model_name(), "strict")
    except NerPhiDetectionError as exc:
        raise TextAnonymizationError(
            exc.error_code,
            status_code=exc.status_code,
        ) from exc
    return _entity_summary(
        entity
        for entity in entities
        if entity.source != SOURCE_STRICT_PROPER_NOUN
        and _is_substantive_residual(masked, entity)
    )


#: A residual span must be mostly real characters. Masking replaces each
#: placeholder with spaces of the same length, and the models reliably predict
#: that a name belongs in the resulting hole - "Dr.<blank>at" comes back as a
#: PERSON. Such a span contains no surviving text and is not evidence of a
#: leak.
_MIN_RESIDUAL_SUBSTANCE = 0.5


def _is_substantive_residual(masked: str, entity: DetectedEntity) -> bool:
    span = masked[entity.start : entity.end]
    if not span:
        return False
    non_space = sum(1 for character in span if not character.isspace())
    return (non_space / len(span)) >= _MIN_RESIDUAL_SUBSTANCE
