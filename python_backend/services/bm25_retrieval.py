"""Privacy-gated in-memory BM25 retrieval over anonymized documents.

The checks in this module are an additional indexing safety gate. They do not
independently establish that upstream content has been de-identified.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_SNIPPET_LENGTH = 200

_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[.-][^\W_]+)*", re.UNICODE)
_DOCUMENT_FIELDS = {
    "document_id",
    "modality",
    "anonymized_text",
    "safe_metadata",
    "anonymization_status",
    "privacy_validation_status",
}
_REQUIRED_DOCUMENT_FIELDS = {
    "document_id",
    "modality",
    "anonymized_text",
    "anonymization_status",
}
_SAFE_METADATA_KEYS = {
    "modality": "modality",
    "documenttype": "document_type",
    "clinicalterms": "clinical_terms",
    "categorylabels": "category_labels",
    "clinicalcodes": "clinical_codes",
    "diseasetags": "disease_tags",
}
_UNSAFE_METADATA_KEY_PARTS = {
    "patientname",
    "patientid",
    "socialsecurity",
    "ssn",
    "medicalrecordnumber",
    "mrn",
    "passport",
    "driverlicense",
    "driverlicence",
    "email",
    "phone",
    "telephone",
    "mobile",
    "address",
    "coordinates",
    "latitude",
    "longitude",
    "originalfilename",
    "filename",
    "rawuploadpath",
    "uploadpath",
    "filepath",
    "filesystempath",
    "rawdicommetadata",
    "rawniftimetadata",
    "dicommetadata",
    "niftimetadata",
    "filebytes",
    "rawbytes",
    "uploadbytes",
    "temporarypath",
    "temppath",
    "rawcontent",
    "originalcontent",
    "extractedcontent",
}
_PASSING_PRIVACY_STATUSES = {
    "approved",
    "completed",
    "passed",
    "satisfied",
    "success",
    "valid",
}


class BM25RetrievalError(ValueError):
    """Base error for safe indexing and query validation failures."""


class UnsafeDocumentError(BM25RetrievalError):
    """Raised when a document is not eligible for the safe index."""


class BM25QueryError(BM25RetrievalError):
    """Raised when a BM25 query or filter is invalid."""


@dataclass(frozen=True)
class SearchableDocument:
    """Internal contract for content that has already been anonymized."""

    document_id: str
    modality: str
    anonymized_text: str
    anonymization_status: str
    safe_metadata: Mapping[str, Any] = field(default_factory=dict)
    privacy_validation_status: Optional[str] = None


@dataclass(frozen=True)
class _IndexedDocument:
    document_id: str
    modality: str
    anonymized_text: str
    safe_metadata: Mapping[str, Any]
    tokens: Sequence[str]
    term_frequencies: Mapping[str, int]


def tokenize(text: str) -> List[str]:
    """Tokenize locally while preserving clinical codes and numeric terms."""

    if not isinstance(text, str):
        raise BM25QueryError("Search text must be a string")
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def _normalized_key(key: str) -> str:
    return "".join(character for character in key.lower() if character.isalnum())


def _contains_binary(value: Any) -> bool:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if isinstance(value, Mapping):
        return any(
            _contains_binary(key) or _contains_binary(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_binary(item) for item in value)
    return False


def _is_unsafe_metadata_key(key: str) -> bool:
    normalized = _normalized_key(key)
    return any(part in normalized for part in _UNSAFE_METADATA_KEY_PARTS)


def _safe_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_metadata_value(item) for item in value]
    raise UnsafeDocumentError(
        "Safe metadata values must be scalar values or lists of scalar values"
    )


def _validate_safe_metadata(metadata: Any) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise UnsafeDocumentError("safe_metadata must be a mapping")
    if _contains_binary(metadata):
        raise UnsafeDocumentError("Raw file bytes cannot be indexed")

    validated: Dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise UnsafeDocumentError("Safe metadata keys must be non-empty strings")
        if _is_unsafe_metadata_key(key):
            raise UnsafeDocumentError("Unsafe metadata keys cannot be indexed")

        canonical_key = _SAFE_METADATA_KEYS.get(_normalized_key(key))
        if canonical_key is None:
            raise UnsafeDocumentError("Metadata key is not approved for BM25 indexing")
        if canonical_key in validated:
            raise UnsafeDocumentError("Duplicate normalized metadata keys are not allowed")
        validated[canonical_key] = _safe_metadata_value(value)
    return validated


def _metadata_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_metadata_text(item) for item in value)
    return str(value)


def _document_mapping(
    document: SearchableDocument | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(document, SearchableDocument):
        return {
            "document_id": document.document_id,
            "modality": document.modality,
            "anonymized_text": document.anonymized_text,
            "anonymization_status": document.anonymization_status,
            "safe_metadata": document.safe_metadata,
            "privacy_validation_status": document.privacy_validation_status,
        }
    if not isinstance(document, Mapping):
        raise UnsafeDocumentError("Documents must use the safe searchable contract")
    return document


def _prepare_document(
    document: SearchableDocument | Mapping[str, Any],
) -> _IndexedDocument:
    data = _document_mapping(document)
    if _contains_binary(data):
        raise UnsafeDocumentError("Raw file bytes cannot be indexed")

    unknown_fields = set(data) - _DOCUMENT_FIELDS
    if unknown_fields:
        raise UnsafeDocumentError("Raw or unsupported document fields cannot be indexed")
    missing_fields = _REQUIRED_DOCUMENT_FIELDS - set(data)
    if missing_fields:
        raise UnsafeDocumentError("Anonymization status and safe document fields are required")

    document_id = data["document_id"]
    modality = data["modality"]
    anonymized_text = data["anonymized_text"]
    anonymization_status = data["anonymization_status"]
    privacy_validation_status = data.get("privacy_validation_status")

    if not isinstance(document_id, str) or not document_id.strip():
        raise UnsafeDocumentError("document_id must be a non-empty string")
    if not isinstance(modality, str) or not modality.strip():
        raise UnsafeDocumentError("modality must be a non-empty string")
    if not isinstance(anonymized_text, str):
        raise UnsafeDocumentError("anonymized_text must be a string")
    if not isinstance(anonymization_status, str):
        raise UnsafeDocumentError("Anonymization status is required")
    if anonymization_status.strip().lower() != "completed":
        raise UnsafeDocumentError("Only completed anonymization outputs can be indexed")

    if privacy_validation_status is not None:
        if not isinstance(privacy_validation_status, str):
            raise UnsafeDocumentError("Privacy validation status must be a string")
        if privacy_validation_status.strip().lower() not in _PASSING_PRIVACY_STATUSES:
            raise UnsafeDocumentError("Privacy validation must pass before indexing")

    safe_metadata = _validate_safe_metadata(data.get("safe_metadata", {}))
    searchable_parts = [anonymized_text, modality]
    searchable_parts.extend(
        _metadata_text(safe_metadata[key])
        for key in sorted(safe_metadata)
    )
    tokens = tokenize(" ".join(part for part in searchable_parts if part))

    return _IndexedDocument(
        document_id=document_id.strip(),
        modality=modality.strip(),
        anonymized_text=anonymized_text,
        safe_metadata=safe_metadata,
        tokens=tokens,
        term_frequencies=Counter(tokens),
    )


def _snippet(text: str, query_tokens: Sequence[str], limit: int) -> str:
    normalized_text = " ".join(text.split())
    if not normalized_text:
        return ""
    if len(normalized_text) <= limit:
        return normalized_text

    lower_text = normalized_text.lower()
    positions = [lower_text.find(token) for token in query_tokens]
    positions = [position for position in positions if position >= 0]
    match_position = min(positions) if positions else 0
    start = max(0, match_position - (limit // 3))
    end = min(len(normalized_text), start + limit)
    start = max(0, end - limit)
    snippet = normalized_text[start:end]
    if start:
        snippet = f"…{snippet}"
    if end < len(normalized_text):
        snippet = f"{snippet}…"
    return snippet


class BM25RetrievalService:
    """Thread-safe, in-memory Okapi BM25 index for safe document records."""

    def __init__(
        self,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        snippet_length: int = DEFAULT_SNIPPET_LENGTH,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        if snippet_length < 1:
            raise ValueError("snippet_length must be greater than zero")

        self.k1 = float(k1)
        self.b = float(b)
        self.snippet_length = snippet_length
        self._documents: Dict[str, _IndexedDocument] = {}
        self._document_frequencies: Counter[str] = Counter()
        self._average_document_length = 0.0
        self._lock = RLock()

    @property
    def document_count(self) -> int:
        with self._lock:
            return len(self._documents)

    @property
    def average_document_length(self) -> float:
        with self._lock:
            return self._average_document_length

    def _rebuild_statistics(self) -> None:
        self._document_frequencies = Counter()
        total_length = 0
        for document in self._documents.values():
            total_length += len(document.tokens)
            self._document_frequencies.update(set(document.tokens))
        self._average_document_length = (
            total_length / len(self._documents) if self._documents else 0.0
        )

    def clear(self) -> None:
        with self._lock:
            self._documents = {}
            self._rebuild_statistics()

    def rebuild(
        self,
        documents: Iterable[SearchableDocument | Mapping[str, Any]],
    ) -> int:
        prepared: Dict[str, _IndexedDocument] = {}
        for document in documents:
            indexed = _prepare_document(document)
            prepared[indexed.document_id] = indexed

        with self._lock:
            self._documents = prepared
            self._rebuild_statistics()
            return len(self._documents)

    def index_documents(
        self,
        documents: Iterable[SearchableDocument | Mapping[str, Any]],
    ) -> int:
        prepared = [_prepare_document(document) for document in documents]
        with self._lock:
            for indexed in prepared:
                self._documents[indexed.document_id] = indexed
            self._rebuild_statistics()
            return len(prepared)

    def upsert_document(
        self,
        document: SearchableDocument | Mapping[str, Any],
    ) -> None:
        indexed = _prepare_document(document)
        with self._lock:
            self._documents[indexed.document_id] = indexed
            self._rebuild_statistics()

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            removed = self._documents.pop(document_id, None) is not None
            if removed:
                self._rebuild_statistics()
            return removed

    def _inverse_document_frequency(self, term: str) -> float:
        document_frequency = self._document_frequencies.get(term, 0)
        document_count = len(self._documents)
        if not document_frequency or not document_count:
            return 0.0
        return math.log(
            1.0
            + (
                document_count - document_frequency + 0.5
            )
            / (document_frequency + 0.5)
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            raise BM25QueryError("Search query must contain at least one token")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise BM25QueryError("top_k must be between 1 and 100")

        normalized_filters = dict(filters or {})
        unsupported_filters = set(normalized_filters) - {"modality"}
        if unsupported_filters:
            raise BM25QueryError("Unsupported BM25 search filter")
        modality_filter = normalized_filters.get("modality")
        if modality_filter is not None:
            if not isinstance(modality_filter, str) or not modality_filter.strip():
                raise BM25QueryError("Modality filter must be a non-empty string")
            modality_filter = modality_filter.strip().lower()

        query_frequencies = Counter(query_tokens)
        with self._lock:
            if not self._documents:
                return []

            scored: List[tuple[float, _IndexedDocument]] = []
            average_length = self._average_document_length
            for document in self._documents.values():
                if modality_filter and document.modality.lower() != modality_filter:
                    continue

                document_length = len(document.tokens)
                score = 0.0
                for term, query_frequency in query_frequencies.items():
                    term_frequency = document.term_frequencies.get(term, 0)
                    if not term_frequency:
                        continue
                    inverse_document_frequency = self._inverse_document_frequency(term)
                    if average_length:
                        length_ratio = document_length / average_length
                    else:
                        length_ratio = 0.0
                    denominator = term_frequency + self.k1 * (
                        1.0 - self.b + self.b * length_ratio
                    )
                    if denominator:
                        score += query_frequency * inverse_document_frequency * (
                            term_frequency * (self.k1 + 1.0) / denominator
                        )
                if score > 0:
                    scored.append((score, document))

            scored.sort(key=lambda item: (-item[0], item[1].document_id))
            results: List[Dict[str, Any]] = []
            for rank, (score, document) in enumerate(scored[:top_k], start=1):
                results.append(
                    {
                        "document_id": document.document_id,
                        "rank": rank,
                        "score": score,
                        "modality": document.modality,
                        "safe_metadata": dict(document.safe_metadata),
                        "snippet": _snippet(
                            document.anonymized_text,
                            query_tokens,
                            self.snippet_length,
                        ),
                    }
                )
            return results
