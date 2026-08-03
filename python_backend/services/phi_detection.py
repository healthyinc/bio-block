import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Pattern, Protocol, Sequence

SOURCE_NER = "ner"
SOURCE_STRUCTURED_PATTERN = "structured_pattern"
SOURCE_CONTEXT_RULE = "context_rule"


@dataclass(frozen=True)
class DetectedEntity:
    """A PHI span whose offsets refer to the original full text."""

    entity_type: str
    start: int
    end: int
    source: str
    score: Optional[float] = None
    original_label: Optional[str] = None


class PhiDetector(Protocol):
    def detect(self, text: str) -> List[DetectedEntity]:
        """Return detected PHI without retaining or returning matched values."""


@dataclass(frozen=True)
class _PatternDefinition:
    name: str
    entity_type: str
    regex: Pattern[str]


def _compile(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# Context is part of identifier patterns so ordinary clinical numbers are not
# classified as identifiers. These expressions are compiled once at import.
STRUCTURED_PATTERNS: Sequence[_PatternDefinition] = (
    _PatternDefinition(
        "medical_record_number",
        "MEDICAL_RECORD_NUMBER",
        _compile(
            r"\b(?:MRN|medical\s+record(?:\s+(?:number|no))?|"
            r"hospital\s+number|chart\s+number)\s*[:#-]?\s*"
            r"[A-Z0-9][A-Z0-9-]{4,20}\b"
        ),
    ),
    _PatternDefinition(
        "patient_id",
        "PATIENT_ID",
        _compile(
            r"\b(?:patient\s+(?:id|identifier|number)|pt\s*id)"
            r"\s*[:#-]?\s*[A-Z0-9][A-Z0-9-]{3,24}\b"
        ),
    ),
    _PatternDefinition(
        "health_plan_id",
        "HEALTH_PLAN_ID",
        _compile(
            r"\b(?:health\s+plan(?:\s+(?:beneficiary\s+)?(?:id|number))?|"
            r"beneficiary\s+id|insurance\s+(?:id|number)|"
            r"policy\s+(?:id|number)|member\s+id|subscriber\s+id)"
            r"\s*[:#-]?\s*[A-Z0-9][A-Z0-9-]{5,30}\b"
        ),
    ),
    _PatternDefinition(
        "accession_number",
        "ACCESSION_NUMBER",
        _compile(
            r"\b(?:accession(?:\s+(?:number|no))?|acc\s*no)"
            r"\s*[:#-]?\s*[A-Z0-9][A-Z0-9-]{4,30}\b"
        ),
    ),
    _PatternDefinition(
        "device_id",
        "DEVICE_ID",
        _compile(
            r"\b(?:device(?:\s+id)?|serial(?:\s+number)?|implant|equipment)"
            r"\s*[:#-]?\s*[A-Z0-9][A-Z0-9-]{5,30}\b"
        ),
    ),
    _PatternDefinition(
        "email_address",
        "EMAIL_ADDRESS",
        _compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    ),
    _PatternDefinition(
        "phone_number",
        "PHONE_NUMBER",
        _compile(
            r"(?<!\w)(?:\+?1[\s.-]?)?"
            r"(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
        ),
    ),
    _PatternDefinition(
        "us_ssn",
        "US_SSN",
        _compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    _PatternDefinition(
        "url",
        "URL",
        _compile(
            r"\b(?:https?://|www\.)[A-Z0-9.-]+(?:/[A-Z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)?"
        ),
    ),
    _PatternDefinition(
        "ip_address",
        "IP_ADDRESS",
        _compile(
            r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)"
            r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])"
        ),
    ),
    _PatternDefinition(
        "numeric_date",
        "DATE",
        _compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b"),
    ),
    _PatternDefinition(
        "clock_time",
        "TIME",
        _compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?[AP]M)?\b"),
    ),
)


class StructuredPatternDetector:
    def detect(self, text: str) -> List[DetectedEntity]:
        entities: List[DetectedEntity] = []
        for definition in STRUCTURED_PATTERNS:
            for match in definition.regex.finditer(text):
                entities.append(
                    DetectedEntity(
                        entity_type=definition.entity_type,
                        start=match.start(),
                        end=match.end(),
                        source=SOURCE_STRUCTURED_PATTERN,
                        original_label=definition.name,
                    )
                )
        return entities


_EXACT_STRUCTURED_TYPES = {
    "MEDICAL_RECORD_NUMBER",
    "PATIENT_ID",
    "HEALTH_PLAN_ID",
    "INSURANCE_ID",
    "ACCESSION_NUMBER",
    "DEVICE_ID",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "SSN",
    "URL",
    "IP_ADDRESS",
}
_BROAD_TEMPORAL_TYPES = {"DATE", "DATE_TIME", "TIME"}
_SOURCE_ORDER = {
    SOURCE_STRUCTURED_PATTERN: 0,
    SOURCE_CONTEXT_RULE: 1,
    SOURCE_NER: 2,
}


def _priority(entity: DetectedEntity) -> int:
    if (
        entity.source == SOURCE_STRUCTURED_PATTERN
        and entity.entity_type in _EXACT_STRUCTURED_TYPES
    ):
        return 400
    if entity.source == SOURCE_CONTEXT_RULE:
        return 300
    if entity.entity_type not in _BROAD_TEMPORAL_TYPES:
        return 200
    return 100


def resolve_overlaps(
    entities: Iterable[DetectedEntity],
    text_length: int,
) -> List[DetectedEntity]:
    """Select valid, deterministic, non-overlapping spans."""

    valid = {
        entity
        for entity in entities
        if 0 <= entity.start < entity.end <= text_length
    }
    ordered = sorted(
        valid,
        key=lambda entity: (
            -_priority(entity),
            -(entity.end - entity.start),
            entity.start,
            _SOURCE_ORDER.get(entity.source, 99),
            entity.entity_type,
            entity.original_label or "",
        ),
    )

    selected: List[DetectedEntity] = []
    for entity in ordered:
        if any(
            entity.start < accepted.end and accepted.start < entity.end
            for accepted in selected
        ):
            continue
        selected.append(entity)

    return sorted(
        selected,
        key=lambda entity: (entity.start, entity.end, entity.entity_type),
    )
