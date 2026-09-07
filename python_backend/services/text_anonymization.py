import hashlib
import os
import re
import string
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from services.ner_phi_detector import (
    proper_noun_offsets,
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
    AgeOverThresholdDetector,
    DetectedEntity,
    PhiDetector,
    StructuredPatternDetector,
    resolve_overlaps,
)
from services.surrogates import (
    SURROGATE_PATTERN,
    SurrogateAllocator,
    looks_like_provider,
)
from services.model_client import RemoteModelDetector, worker_enabled
from services.detection_evidence import (
    contains_proper_noun_token,
    has_clinical_support,
    is_confirmed_clinical_marker,
    EVIDENCE_DETERMINISTIC,
    assess_finding,
    count_agreeing,
    evidence_type,
)
from services.text_utility import measure_text_utility
from services.transformation_provenance import (
    KIND_PLACEHOLDER,
    KIND_SURROGATE,
    ProvenanceBuilder,
    TransformationProvenance,
    kind_for_replacement,
)

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

    kept, review = _apply_evidence_model(detected, text)
    resolved = _merge_adjacent_same_type(resolve_overlaps(kept, len(text)), text)
    return resolved, review


#: Labels that a clinical term can be wrongly assigned. A detection of one of
#: these over recorded clinical vocabulary is a misclassification, whichever
#: detector produced it.
_NAME_SHAPED_TYPES = frozenset(
    {"PERSON", "ORGANIZATION", "FACILITY", "LOCATION", "ADDRESS", "MEDICAL_CONDITION"}
)


def _apply_evidence_model(
    entities: List[DetectedEntity],
    text: str,
) -> Tuple[List[DetectedEntity], List[str]]:
    """Decide each candidate by evidence strength rather than word membership.

    Applied to every source. The pinned models label "Parkinson" a PERSON as
    readily as spaCy does, so filtering only the detectors we control leaves
    the same clinical term destroyed by a different route.

    Returns the spans to redact and the review reasons raised by candidates
    that could not be resolved either way. An ambiguous span is neither
    deleted on suspicion nor kept on hope: the document goes to a human.
    """
    kept: List[DetectedEntity] = []
    review: List[str] = []
    # Parsed once per document, not once per candidate. None means the parse
    # failed, which the evidence model treats as no information rather than
    # as "no proper nouns here".
    proper_nouns = proper_noun_offsets(text, configured_model_name())
    for entity in entities:
        agreeing = count_agreeing(
            entities,
            entity.start,
            entity.end,
            entity.entity_type,
            exclude_source=entity.source,
        )
        assessment = assess_finding(
            text,
            entity.start,
            entity.end,
            entity.entity_type,
            entity.source,
            agreeing_detectors=agreeing,
            clinical_support=has_clinical_support(
                entities, entity.start, entity.end
            ),
            contains_proper_noun=contains_proper_noun_token(
                proper_nouns, entity.start, entity.end
            ),
        )
        if assessment.action == "redact":
            kept.append(entity)
        elif assessment.action == "review":
            review.append(f"ambiguous_{entity.entity_type.lower()}_requires_review")
        # "preserve" keeps the clinical content and adds no review reason.
    return kept, sorted(set(review))


#: Only horizontal whitespace, a hyphen or a possessive may sit between two
#: fragments of one name. Anything else is two different entities - and a line
#: break in particular is not a gap inside a name. Allowing one merged
#: "Template FP-TEMPLATE-52123" and the "Fax" on the next line into a single
#: span, and the replacement swallowed a whole field.
_NAME_GAP = re.compile("^[" + chr(32) + chr(9) + chr(45) + chr(39) + "]*$")


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
):
    """Rewrite the document and record where each replacement landed.

    Built left to right so the provenance offsets are correct by
    construction. Replacing back to front keeps *input* offsets valid but
    makes every recorded output offset wrong the moment an earlier entity is
    replaced, because that replacement shifts everything after it. Surrogate
    numbering also follows reading order this way, which is what a reader
    expects.
    """
    ordered = sorted(entities, key=lambda item: (item.start, item.end))
    provenance = ProvenanceBuilder()
    pieces: List[str] = []
    output_length = 0
    cursor = 0

    for entity in ordered:
        if entity.start < cursor:
            # Overlap resolution should have removed these, but a stray
            # overlapping span must never corrupt the output.
            continue
        untouched = text[cursor : entity.start]
        pieces.append(untouched)
        output_length += len(untouched)

        original_value = text[entity.start : entity.end]
        is_provider = entity.entity_type == "PERSON" and looks_like_provider(
            text, entity.start
        )
        replacement = _replacement_for(
            entity.entity_type,
            original_value,
            salt,
            settings,
            allocator=allocator,
            is_provider=is_provider,
        )
        pieces.append(replacement)
        provenance.record(
            output_length,
            len(replacement),
            kind_for_replacement(entity.entity_type, replacement),
            entity.entity_type,
        )
        output_length += len(replacement)
        cursor = entity.end

    pieces.append(text[cursor:])
    return "".join(pieces), provenance.build()


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
        entities, _review = _detect_entities(text, configured_model_name(), "strict")
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
        entities, detection_review = _detect_entities(
            text, model_name, privacy_profile
        )
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
    anonymized_text, provenance = _replace_entities(
        text, entities, salt, settings, allocator
    )

    # An age reference that cannot be resolved to a number cannot be judged
    # against the Safe Harbor threshold, so the document needs a human.
    review_reasons = sorted(
        {
            "uncertain_age_reference"
            for entity in entities
            if entity.entity_type == "AGE_UNCERTAIN"
        }
        | set(detection_review)
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

    # Scanned once, here, because this is the only place holding the
    # provenance map for the text it just produced.
    second_pass = residual_findings(anonymized_text, provenance)

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
        # Computed here, not by the caller, because this is the only place
        # holding the provenance map. Handing the map out instead would put
        # generated-region offsets into every serialized response, and every
        # caller that forgot to pass it back would silently re-scan surrogates
        # as if they were surviving text - the exact defect Phase 11 removes.
        "residual_phi_categories": _residual_summary(second_pass),
        # The classified second-pass record for each finding: detector,
        # category, evidence type, location type, whether it overlaps a
        # generated region, and the classification. No value, no offsets, no
        # sentence - a diagnostic report on a privacy pipeline is exactly the
        # document that ends up pasted into a ticket.
        "residual_findings": second_pass,
        # How many regions the sanitizer wrote, by kind. Counts only: the
        # offsets stay inside this function, where the validator uses them.
        "provenance_counts": provenance.counts(),
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


# -- classification vocabulary for second-pass findings ---------------------
#
# Every residual finding lands in exactly one of these. Four are benign and
# three block release; nothing is dropped silently, because a finding with no
# recorded classification is a finding nobody reviewed.

CLASSIFICATION_GENUINE_PHI = "genuine_surviving_phi"
CLASSIFICATION_PLAUSIBLE_PHI = "additional_plausible_phi"
CLASSIFICATION_EXACT_SURROGATE = "exact_generated_surrogate"
CLASSIFICATION_PLACEHOLDER = "anonymizer_placeholder"
CLASSIFICATION_DETECTOR_ARTEFACT = "detector_artefact_modified_context"
CLASSIFICATION_USEFUL_CLINICAL = "useful_clinical_content"
CLASSIFICATION_MALFORMED = "malformed_output"

RESIDUAL_CLASSIFICATIONS = (
    CLASSIFICATION_GENUINE_PHI,
    CLASSIFICATION_PLAUSIBLE_PHI,
    CLASSIFICATION_EXACT_SURROGATE,
    CLASSIFICATION_PLACEHOLDER,
    CLASSIFICATION_DETECTOR_ARTEFACT,
    CLASSIFICATION_USEFUL_CLINICAL,
    CLASSIFICATION_MALFORMED,
)

#: A generated token that got truncated, nested or otherwise mangled leaves a
#: fragment like "<REDACTED_PERSON" or a bare "PATIENT_" behind. A fragment is
#: not PHI, but it is proof the sanitizer's own writing is unsound, so the
#: document must not be released on the strength of a scan of it.
_PLACEHOLDER_FRAGMENT = re.compile(r"<\s*REDACTED_?[A-Z0-9_]*(?![A-Z0-9_]*>)")
_SURROGATE_STEM = re.compile(
    r"\b(?:PATIENT|PROVIDER|FACILITY|ORG|PLACE|ADDRESS|RECORD|PATIENTID|PLAN"
    r"|ACCESSION|DEVICE|IDENTIFIER|USER)_(?![0-9A-Z])"
)


def _region_text_is_well_formed(kind: str, span: str) -> bool:
    """Whether a recorded region still holds the token we meant to write."""
    if not span:
        return False
    if kind == KIND_PLACEHOLDER:
        return bool(_PLACEHOLDER_PATTERN.fullmatch(span))
    if kind == KIND_SURROGATE:
        return bool(
            SURROGATE_PATTERN.fullmatch(span) or _SURROGATE_PATTERN.fullmatch(span)
        )
    # Generalized ages and shifted dates are ordinary text by design; they are
    # malformed only if a replacement marker leaked into them.
    return "<" not in span and ">" not in span


def _malformed_output_findings(
    anonymized_text: str,
    provenance: "TransformationProvenance",
) -> List[Dict[str, Any]]:
    """Structural check on the sanitizer's own output.

    This is not a PHI scan. It asks whether the text we are about to judge is
    the text we intended to write: a half-written placeholder means the
    replacement loop produced something we cannot reason about, and reasoning
    about it anyway is how a partially-redacted value gets released. Findings
    here block, and they carry a shape, never a value.
    """
    findings: List[Dict[str, Any]] = []

    for region in provenance.regions:
        span = anonymized_text[region.start : region.end]
        if not _region_text_is_well_formed(region.kind, span):
            findings.append(
                {
                    "detector": "output_structure_validator",
                    "category": "MALFORMED_REPLACEMENT",
                    "evidence_type": EVIDENCE_DETERMINISTIC,
                    "location_type": "inside_generated_region",
                    "overlaps_generated_region": True,
                    "classification": CLASSIFICATION_MALFORMED,
                    "blocking": True,
                }
            )

    for pattern in (_PLACEHOLDER_FRAGMENT, _SURROGATE_STEM):
        for match in pattern.finditer(anonymized_text):
            if provenance.covering(match.start(), match.end()):
                continue
            findings.append(
                {
                    "detector": "output_structure_validator",
                    "category": "MALFORMED_REPLACEMENT",
                    "evidence_type": EVIDENCE_DETERMINISTIC,
                    "location_type": provenance.location_type(
                        match.start(), match.end()
                    ),
                    "overlaps_generated_region": bool(
                        provenance.touching(match.start(), match.end())
                    ),
                    "classification": CLASSIFICATION_MALFORMED,
                    "blocking": True,
                }
            )

    return findings


#: How an extracted fragment reads. Used by the pixel pipelines, where an OCR
#: box carries a few words with no sentence around them.
EXTRACTED_PHI = "phi"
EXTRACTED_NON_PHI = "non_phi"
EXTRACTED_UNCERTAIN = "uncertain"


def classify_extracted_text(text: str) -> str:
    """Classify a short extracted string as PHI, useful non-PHI, or unknown.

    Burned-in image text is the case this exists for. Blacking out every box
    an OCR engine reports makes the residual-text count zero and destroys the
    laterality marker, the scale bar and the burned-in measurement along with
    the patient name - a clean privacy number bought with the diagnostic
    content the image was kept for.

    Three answers, and the middle one is the point: only text that is
    *recognisably* clinical is preserved. Everything the pipeline cannot place
    comes back uncertain, and an uncertain fragment is treated as PHI by the
    caller, not waved through.
    """
    if not text or not text.strip():
        return EXTRACTED_NON_PHI

    try:
        entities, review = _detect_entities(text, configured_model_name(), "strict")
    except NerPhiDetectionError as exc:
        raise TextAnonymizationError(
            exc.error_code, status_code=exc.status_code
        ) from exc

    if entities:
        return EXTRACTED_PHI
    if review:
        return EXTRACTED_UNCERTAIN
    if is_confirmed_clinical_marker(text):
        return EXTRACTED_NON_PHI
    return EXTRACTED_UNCERTAIN


#: Characters that carry no identity of their own. Stripped from the
#: uncovered remainder of a straddling prediction before it is assessed,
#: so a trailing comma is not mistaken for surviving content.
_BOUNDARY_CHARACTERS = string.whitespace + '.,;:()[]{}<>\'"-/\\'


def _remainder_is_accounted_for(
    text: str,
    start: int,
    end: int,
    provenance: "TransformationProvenance",
    category: str,
    source: str,
    agreeing: int,
) -> bool:
    """Whether a straddling prediction claims anything beyond our own token.

    A detector reading the redacted output routinely proposes a span one word
    wider than the surrogate it is really responding to - "Dr. PROVIDER_001",
    "PATIENT_001 was". The span is not wholly inside a generated region, so it
    cannot simply be discounted, and blocking on it would hold clean documents
    on the strength of a boundary.

    So the remainder is examined rather than ignored. Each part of the span
    that no generated region covers is put through the same evidence model as
    any other candidate, in full document context. Only when every remainder
    is punctuation, whitespace, or text the model reads as *not* an identifier
    is the finding an artefact of the boundary. A real name sitting beside a
    surrogate - "PATIENT_001 Balasubramanian" - assesses as an identifier and
    still blocks, which is the case this must never wave through.
    """
    for gap_start, gap_end in provenance.uncovered(start, end):
        fragment = text[gap_start:gap_end]
        stripped = fragment.strip(_BOUNDARY_CHARACTERS)
        if not stripped:
            continue
        offset = gap_start + fragment.index(stripped)
        assessment = assess_finding(
            text,
            offset,
            offset + len(stripped),
            category,
            source,
            agreeing_detectors=agreeing,
        )
        if assessment.action != "preserve":
            return False
    return True


def residual_findings(
    anonymized_text: str,
    provenance: "TransformationProvenance | None" = None,
) -> List[Dict[str, Any]]:
    """Re-scan the **exact** serialized output and classify what is found.

    The previous implementation blanked every generated token with spaces
    before re-scanning. That rewrote the sentence it was checking: "Dr.
    PROVIDER_001 at FACILITY_001" became "Dr.<spaces>at<spaces>", and both
    models predicted a name belonged in the hole. Sixty per cent of clean
    documents were blocked by artefacts of the masking itself.

    Nothing is masked now. The scan runs on the real output, and the
    provenance map says which character ranges the sanitizer wrote. A finding
    is discounted only when its span lies **wholly** inside a generated
    region - a prediction that merely touches a surrogate still covers text we
    did not generate, and that is exactly where a missed identifier would sit.

    Returns one record per finding carrying no value and no sentence.
    """
    if not isinstance(anonymized_text, str):
        raise TextAnonymizationError("Text input must be a string")
    if not anonymized_text.strip():
        return []
    if len(anonymized_text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise TextAnonymizationError(
            f"Text input exceeds the {MAX_TEXT_BYTES} byte limit",
            status_code=413,
        )

    provenance = provenance or TransformationProvenance()
    try:
        entities, _review = _detect_entities(
            anonymized_text, configured_model_name(), "strict"
        )
    except NerPhiDetectionError as exc:
        raise TextAnonymizationError(
            exc.error_code, status_code=exc.status_code
        ) from exc

    findings: List[Dict[str, Any]] = _malformed_output_findings(
        anonymized_text, provenance
    )
    proper_nouns = proper_noun_offsets(anonymized_text, configured_model_name())
    for entity in entities:
        location = provenance.location_type(entity.start, entity.end)
        covering = provenance.covering(entity.start, entity.end)
        evidence = evidence_type(entity.source)

        if covering is not None:
            # Wholly our own token: the detector is reading a surrogate.
            classification = (
                CLASSIFICATION_EXACT_SURROGATE
                if covering.kind == KIND_SURROGATE
                else CLASSIFICATION_PLACEHOLDER
            )
            blocking = False
        elif evidence == EVIDENCE_DETERMINISTIC:
            # Layer 1: an exact identifier match in the released text is a
            # genuine survivor, whatever else is true.
            classification, blocking = CLASSIFICATION_GENUINE_PHI, True
        else:
            agreeing = count_agreeing(
                entities,
                entity.start,
                entity.end,
                entity.entity_type,
                exclude_source=entity.source,
            )
            assessment = assess_finding(
                anonymized_text,
                entity.start,
                entity.end,
                entity.entity_type,
                entity.source,
                agreeing_detectors=agreeing,
                clinical_support=has_clinical_support(
                    entities, entity.start, entity.end
                ),
                contains_proper_noun=contains_proper_noun_token(
                    proper_nouns, entity.start, entity.end
                ),
            )
            if assessment.action == "preserve":
                classification, blocking = CLASSIFICATION_USEFUL_CLINICAL, False
            elif location == "spans_generated_and_original" and (
                _remainder_is_accounted_for(
                    anonymized_text,
                    entity.start,
                    entity.end,
                    provenance,
                    entity.entity_type,
                    entity.source,
                    agreeing,
                )
            ):
                # The prediction reaches past our token, but everything it
                # reaches is examined and reads as non-identifying.
                classification, blocking = CLASSIFICATION_DETECTOR_ARTEFACT, False
            else:
                classification, blocking = CLASSIFICATION_PLAUSIBLE_PHI, True

        findings.append(
            {
                "detector": entity.source,
                "category": entity.entity_type,
                "evidence_type": evidence,
                "location_type": location,
                "overlaps_generated_region": bool(
                    provenance.touching(entity.start, entity.end)
                ),
                "classification": classification,
                "blocking": blocking,
            }
        )
    return findings


def _residual_summary(findings: List[Dict[str, Any]]) -> Dict[str, int]:
    """Blocking findings only, counted by category."""
    summary: Dict[str, int] = {}
    for finding in findings:
        if finding["blocking"]:
            category = finding["category"]
            summary[category] = summary.get(category, 0) + 1
    return summary


def residual_phi_categories(
    anonymized_text: str,
    provenance: "TransformationProvenance | None" = None,
) -> Dict[str, int]:
    """Categories of genuinely surviving PHI. Non-empty must block release."""
    return _residual_summary(residual_findings(anonymized_text, provenance))
