import hashlib
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, Iterator, List, Tuple

SUPPORTED_PROFILES = {"strict", "research"}
DEFAULT_STUDY_SALT = "bio-block-week2-development-salt"
HASH_LENGTH = 8

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


def _blank_english_nlp_engine():
    """
    Minimal Presidio NLP engine backed by a blank spaCy tokenizer.

    Presidio's default engine may try to download a spaCy model. This keeps
    Week 2 text analysis deterministic and offline for pattern recognizers.
    """
    try:
        import spacy
        from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine
    except ImportError as exc:
        raise TextAnonymizationError(
            "Text anonymization NLP dependency is not available.",
            status_code=503,
        ) from exc

    class BlankEnglishNlpEngine(NlpEngine):
        def __init__(self) -> None:
            self._nlp = spacy.blank("en")

        def load(self) -> None:
            return None

        def is_loaded(self) -> bool:
            return True

        def get_supported_entities(self) -> List[str]:
            return []

        def get_supported_languages(self) -> List[str]:
            return ["en"]

        def process_text(self, text: str, language: str):
            doc = self._nlp(text)
            return NlpArtifacts(
                entities=[],
                tokens=doc,
                tokens_indices=[token.idx for token in doc],
                lemmas=[token.text.lower() for token in doc],
                nlp_engine=self,
                language=language,
            )

        def process_batch(
            self,
            texts: Iterable[str],
            language: str,
            batch_size: int = 1,
            n_process: int = 1,
            **kwargs: Any,
        ) -> Iterator[Tuple[str, Any]]:
            for text in texts:
                yield text, self.process_text(text, language)

        def is_stopword(self, word: str, language: str) -> bool:
            return False

        def is_punct(self, word: str, language: str) -> bool:
            return bool(word) and all(not char.isalnum() for char in word)

    return BlankEnglishNlpEngine()


def stable_hash(value: str, salt: str, length: int = HASH_LENGTH) -> str:
    digest = hashlib.sha256(f"{salt}:{value.strip().lower()}".encode("utf-8"))
    return digest.hexdigest().upper()[:length]


def pseudonymize_person(value: str, salt: str) -> str:
    return f"Patient_{stable_hash(value, salt)}"


def pseudonymize_mrn(value: str, salt: str) -> str:
    return f"MRN_{stable_hash(value, salt)}"


def pseudonymize_patient_id(value: str, salt: str) -> str:
    return f"PATIENT_ID_{stable_hash(value, salt)}"


def pseudonymize_health_plan(value: str, salt: str) -> str:
    return f"HEALTH_PLAN_{stable_hash(value, salt)}"


def pseudonymize_accession(value: str, salt: str) -> str:
    return f"ACCESSION_{stable_hash(value, salt)}"


def pseudonymize_device(value: str, salt: str) -> str:
    return f"DEVICE_{stable_hash(value, salt)}"


def _pattern(name: str, regex: str, score: float):
    from presidio_analyzer import Pattern

    return Pattern(name=name, regex=regex, score=score)


def _clinical_recognizers() -> List[Any]:
    from presidio_analyzer import PatternRecognizer

    return [
        PatternRecognizer(
            supported_entity="MEDICAL_RECORD_NUMBER",
            name="Clinical MRN Recognizer",
            patterns=[
                _pattern(
                    "mrn_with_context",
                    (
                        r"\b(?:MRN|medical\s+record(?:\s+number)?|"
                        r"hospital\s+number|chart\s+number)\s*[:#-]?\s*"
                        r"[A-Z0-9][A-Z0-9-]{4,20}\b"
                    ),
                    0.85,
                )
            ],
            context=[
                "mrn",
                "medical record",
                "medical record number",
                "hospital number",
                "chart number",
            ],
        ),
        PatternRecognizer(
            supported_entity="PATIENT_ID",
            name="Clinical Patient ID Recognizer",
            patterns=[
                _pattern(
                    "patient_id_with_context",
                    (
                        r"\b(?:patient\s+(?:id|identifier|number)|pt\s*id)"
                        r"\s*[:#-]?\s*[A-Z0-9][A-Z0-9-]{3,24}\b"
                    ),
                    0.82,
                )
            ],
            context=[
                "patient id",
                "patient number",
                "patient identifier",
                "pt id",
            ],
        ),
        PatternRecognizer(
            supported_entity="HEALTH_PLAN_ID",
            name="Clinical Health Plan ID Recognizer",
            patterns=[
                _pattern(
                    "health_plan_id_with_context",
                    (
                        r"\b(?:health\s+plan(?:\s+(?:beneficiary\s+)?"
                        r"(?:id|number))?|beneficiary\s+id|insurance\s+"
                        r"(?:id|number)|policy\s+(?:id|number)|"
                        r"member\s+id|subscriber\s+id)\s*[:#-]?\s*"
                        r"[A-Z0-9][A-Z0-9-]{5,30}\b"
                    ),
                    0.82,
                )
            ],
            context=[
                "health plan",
                "beneficiary",
                "insurance",
                "policy",
                "member id",
                "subscriber id",
            ],
        ),
        PatternRecognizer(
            supported_entity="ACCESSION_NUMBER",
            name="Clinical Accession Number Recognizer",
            patterns=[
                _pattern(
                    "accession_with_context",
                    (
                        r"\b(?:accession(?:\s+(?:number|no))?|acc\s*no)"
                        r"\s*[:#-]?\s*[A-Z0-9][A-Z0-9-]{4,30}\b"
                    ),
                    0.8,
                )
            ],
            context=["accession", "accession number", "acc no", "accession no"],
        ),
        PatternRecognizer(
            supported_entity="DEVICE_ID",
            name="Clinical Device ID Recognizer",
            patterns=[
                _pattern(
                    "device_id_with_context",
                    (
                        r"\b(?:device(?:\s+id)?|serial(?:\s+number)?|"
                        r"implant|equipment)\s*[:#-]?\s*"
                        r"[A-Z0-9][A-Z0-9-]{5,30}\b"
                    ),
                    0.78,
                )
            ],
            context=[
                "device",
                "serial",
                "serial number",
                "device id",
                "implant",
                "equipment",
            ],
        ),
    ]


def _common_phi_recognizers() -> List[Any]:
    from presidio_analyzer import PatternRecognizer

    return [
        PatternRecognizer(
            supported_entity="EMAIL_ADDRESS",
            name="Email Address Recognizer",
            patterns=[
                _pattern(
                    "email_address",
                    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                    0.95,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            name="Phone Number Recognizer",
            patterns=[
                _pattern(
                    "us_phone_number",
                    (
                        r"(?<!\w)(?:\+?1[\s.-]?)?"
                        r"(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}"
                        r"(?!\w)"
                    ),
                    0.78,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity="US_SSN",
            name="US SSN Recognizer",
            patterns=[_pattern("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b", 0.9)],
        ),
        PatternRecognizer(
            supported_entity="DATE_TIME",
            name="Common Date Recognizer",
            patterns=[
                _pattern(
                    "common_numeric_date",
                    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b",
                    0.6,
                )
            ],
        ),
    ]


@lru_cache(maxsize=1)
def _get_analyzer():
    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    except ImportError as exc:
        raise TextAnonymizationError(
            "Text anonymization engine is not available.",
            status_code=503,
        ) from exc

    try:
        nlp_engine = _blank_english_nlp_engine()
        registry = RecognizerRegistry(supported_languages=["en"])
        for recognizer in _common_phi_recognizers() + _clinical_recognizers():
            registry.add_recognizer(recognizer)

        return AnalyzerEngine(
            registry=registry,
            nlp_engine=nlp_engine,
            supported_languages=["en"],
        )
    except TextAnonymizationError:
        raise
    except Exception as exc:
        raise TextAnonymizationError(
            "Text anonymization engine could not be initialized.",
            status_code=503,
        ) from exc


def _normalize_profile(profile: str) -> str:
    normalized = (profile or "").strip().lower()
    if normalized not in SUPPORTED_PROFILES:
        raise TextAnonymizationError(
            "Invalid privacy profile. Supported profiles: strict, research"
        )
    return normalized


def _select_non_overlapping(results: Iterable[Any]) -> List[Any]:
    ordered = sorted(
        results,
        key=lambda result: (
            -float(result.score or 0),
            -(result.end - result.start),
            result.start,
        ),
    )

    selected = []
    occupied: List[Tuple[int, int]] = []
    for result in ordered:
        overlaps = any(
            result.start < end and start < result.end for start, end in occupied
        )
        if overlaps:
            continue
        selected.append(result)
        occupied.append((result.start, result.end))

    return sorted(selected, key=lambda result: result.start)


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
    if entity_type == "DATE_TIME":
        return "<REDACTED_DATE>"
    return f"<REDACTED_{entity_type}>"


def _replace_entities(text: str, results: List[Any], salt: str) -> str:
    anonymized_parts = []
    cursor = 0

    for result in results:
        anonymized_parts.append(text[cursor : result.start])
        original_value = text[result.start : result.end]
        anonymized_parts.append(
            _replacement_for(result.entity_type, original_value, salt)
        )
        cursor = result.end

    anonymized_parts.append(text[cursor:])
    return "".join(anonymized_parts)


def _entity_summary(results: Iterable[Any]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for result in results:
        summary[result.entity_type] = summary.get(result.entity_type, 0) + 1
    return summary


def anonymize_clinical_text(
    text: str,
    profile: str = "strict",
    study_salt: str | None = None,
) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise TextAnonymizationError("Text input must be a string")
    if not text.strip():
        raise TextAnonymizationError("Text input is empty")

    _normalize_profile(profile)
    salt = study_salt or DEFAULT_STUDY_SALT

    analyzer = _get_analyzer()
    try:
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=[
                "PERSON",
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
                "DATE_TIME",
            ],
        )
    except Exception as exc:
        raise TextAnonymizationError(
            "Text anonymization failed", status_code=500
        ) from exc

    selected_results = _select_non_overlapping(results)
    return {
        "anonymization_status": "completed",
        "anonymized_text": _replace_entities(text, selected_results, salt),
        "detected_entities": _entity_summary(selected_results),
    }
