"""Offline-only local model adapters that propose PHI spans.

These adapters are detectors, never release authorities. Every failure path
(configuration, missing weights, checksum mismatch, inference error, timeout)
raises ``LocalModelError`` so the caller blocks the artifact instead of falling
back to unredacted content. No raw matched text is ever logged, returned in an
exception, or stored on an entity: only offsets, categories, scores, and the
originating detector name.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from services.privacy_contracts import PhiEntity


MANIFEST_PATH = Path(__file__).resolve().parent.parent / "config" / "model_manifest.json"

MODEL_MODE_ENV_VAR = "PHI_MODEL_MODE"
MODE_OFFLINE = "offline"
MODE_LEGACY_TEST = "legacy_test"
SUPPORTED_MODEL_MODES = (MODE_OFFLINE, MODE_LEGACY_TEST)
DEFAULT_MODEL_MODE = MODE_OFFLINE

DEFAULT_CANDIDATE_THRESHOLD = 0.0
DEFAULT_REDACTION_THRESHOLD = 0.0
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_BATCH_SIZE = 8
DEFAULT_INFERENCE_BUDGET_SECONDS = 120.0

# Weights are hundreds of megabytes; digest them incrementally.
_CHECKSUM_READ_SIZE = 1024 * 1024

SOURCE_STANFORD = "stanford_deidentifier"
SOURCE_GLINER = "gliner_multi_pii"

STANFORD_LABELS = {
    "VENDOR": "ORGANIZATION",
    "DATE": "DATE_TIME",
    "HCW": "PERSON",
    "HOSPITAL": "FACILITY",
    "ID": "IDENTIFIER",
    "PATIENT": "PERSON",
    "PHONE": "PHONE_NUMBER",
}
GLINER_LABELS = {
    "person": "PERSON",
    "organization": "ORGANIZATION",
    "phone number": "PHONE_NUMBER",
    "mobile phone number": "PHONE_NUMBER",
    "landline phone number": "PHONE_NUMBER",
    "fax number": "PHONE_NUMBER",
    "address": "ADDRESS",
    "passport number": "PASSPORT_NUMBER",
    "passport_number": "PASSPORT_NUMBER",
    "email": "EMAIL_ADDRESS",
    "email address": "EMAIL_ADDRESS",
    "social security number": "US_SSN",
    "social_security_number": "US_SSN",
    "health insurance id number": "HEALTH_PLAN_ID",
    "health insurance number": "HEALTH_PLAN_ID",
    "date of birth": "DATE_TIME",
    "driver's license number": "DRIVER_LICENSE",
    "medical condition": "MEDICAL_CONDITION",
    "identity card number": "IDENTIFIER",
    "national id number": "IDENTIFIER",
    "ip address": "IP_ADDRESS",
    "username": "USERNAME",
    "postal code": "POSTAL_CODE",
    "serial number": "DEVICE_ID",
}
GLINER_REQUESTED_LABELS = tuple(GLINER_LABELS)


class LocalModelError(RuntimeError):
    """Fail-closed model error. Carries an error code only, never PHI."""

    def __init__(self, error_code: str, status_code: int = 503):
        super().__init__(error_code)
        self.error_code = error_code
        self.status_code = status_code


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    revision: str
    license: str
    weight_file: str
    weight_sha256: str
    #: "detector" for a model that proposes spans, "tokenizer_backbone" for a
    #: transitive dependency a detector loads by name at construction time.
    role: str = "detector"
    #: Which detector requires this entry, for backbone specs.
    required_by: Optional[str] = None
    #: Restrict a download to the files actually needed. GLiNER's backbone is
    #: used for its tokenizer only, so its 1 GB of encoder weights are skipped.
    allow_patterns: Tuple[str, ...] = ()
    #: A backbone is resolved by repo name at load time, not by path, so the
    #: cache needs a branch ref pointing at the pinned commit for offline
    #: resolution to succeed. The content is still the pinned revision and is
    #: still checksum-verified.
    alias_ref: Optional[str] = None


@dataclass(frozen=True)
class DetectorConfig:
    candidate_threshold: float = DEFAULT_CANDIDATE_THRESHOLD
    redaction_threshold: float = DEFAULT_REDACTION_THRESHOLD
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    batch_size: int = DEFAULT_BATCH_SIZE
    inference_budget_seconds: float = DEFAULT_INFERENCE_BUDGET_SECONDS
    calibrated: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.candidate_threshold <= self.redaction_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= candidate <= redaction <= 1")
        if self.chunk_size <= 0 or not 1 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk overlap must be within 1..chunk_size-1")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.inference_budget_seconds <= 0:
            raise ValueError("inference budget must be positive")


def detector_config_from_env() -> DetectorConfig:
    try:
        return DetectorConfig(
            candidate_threshold=float(
                os.getenv("PHI_CANDIDATE_THRESHOLD", DEFAULT_CANDIDATE_THRESHOLD)
            ),
            redaction_threshold=float(
                os.getenv("PHI_REDACTION_THRESHOLD", DEFAULT_REDACTION_THRESHOLD)
            ),
            chunk_size=int(os.getenv("PHI_TEXT_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
            chunk_overlap=int(
                os.getenv("PHI_TEXT_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
            ),
            batch_size=int(os.getenv("PHI_MODEL_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
            inference_budget_seconds=float(
                os.getenv(
                    "PHI_MODEL_TIMEOUT_SECONDS", DEFAULT_INFERENCE_BUDGET_SECONDS
                )
            ),
            calibrated=os.getenv("PHI_THRESHOLDS_CALIBRATED", "0") == "1",
        )
    except (TypeError, ValueError) as exc:
        raise LocalModelError("invalid_model_configuration", 500) from exc


THRESHOLDS_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "detection_thresholds.json"
)


@lru_cache(maxsize=1)
def load_locked_thresholds() -> Dict[str, Dict[str, Any]]:
    """Load the calibrated, locked per-detector thresholds.

    Absent or unreadable, this returns an empty mapping and callers fall back
    to the uncalibrated zero defaults, which redact every candidate. Missing
    calibration therefore over-redacts rather than under-redacts.
    """
    try:
        if not THRESHOLDS_PATH.is_file():
            return {}
        data = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LocalModelError("invalid_threshold_configuration", 500) from exc
    return data.get("detectors", {})


def calibrated_config_for(detector_name: str) -> DetectorConfig:
    """Config for one detector, preferring an explicit environment override.

    Precedence: environment variable, then the locked calibration file, then
    the conservative zero default. An environment override always wins so an
    operator can tighten or loosen a single deployment without editing the
    committed calibration.
    """
    env_candidate = os.getenv("PHI_CANDIDATE_THRESHOLD")
    env_redaction = os.getenv("PHI_REDACTION_THRESHOLD")
    locked = load_locked_thresholds().get(detector_name, {})

    try:
        candidate = float(
            env_candidate
            if env_candidate is not None
            else locked.get("candidate_threshold", DEFAULT_CANDIDATE_THRESHOLD)
        )
        redaction = float(
            env_redaction
            if env_redaction is not None
            else locked.get("redaction_threshold", candidate)
        )
        base = detector_config_from_env()
        return DetectorConfig(
            candidate_threshold=candidate,
            redaction_threshold=max(candidate, redaction),
            chunk_size=base.chunk_size,
            chunk_overlap=base.chunk_overlap,
            batch_size=base.batch_size,
            inference_budget_seconds=base.inference_budget_seconds,
            calibrated=bool(locked) and env_candidate is None,
        )
    except (TypeError, ValueError) as exc:
        raise LocalModelError("invalid_model_configuration", 500) from exc


def resolve_model_mode() -> str:
    """Return the validated model mode; unknown values fail closed."""
    mode = os.getenv(MODEL_MODE_ENV_VAR, DEFAULT_MODEL_MODE).strip()
    if mode not in SUPPORTED_MODEL_MODES:
        raise LocalModelError("invalid_model_configuration", 500)
    return mode


def local_models_enabled() -> bool:
    return resolve_model_mode() == MODE_OFFLINE


@lru_cache(maxsize=1)
def load_model_manifest() -> Dict[str, ModelSpec]:
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        specs: Dict[str, ModelSpec] = {}
        for name, values in data.items():
            fields = dict(values)
            # Lists are not hashable and ModelSpec is frozen.
            if "allow_patterns" in fields:
                fields["allow_patterns"] = tuple(fields["allow_patterns"])
            specs[name] = ModelSpec(name=name, **fields)
        return specs
    except Exception as exc:
        raise LocalModelError("invalid_model_manifest", 500) from exc


def detector_specs() -> Dict[str, ModelSpec]:
    """Manifest entries that are detectors, excluding transitive backbones."""
    return {
        name: spec
        for name, spec in load_model_manifest().items()
        if spec.role == "detector"
    }


def backbone_specs_for(detector_name: str) -> List[ModelSpec]:
    """Pinned dependencies a detector loads by repository name at load time."""
    return [
        spec
        for spec in load_model_manifest().values()
        if spec.role != "detector" and spec.required_by == detector_name
    ]


def overlapping_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[Tuple[int, str]]:
    """Split text into windows that overlap, so boundary-straddling spans survive.

    An entity is guaranteed to fall wholly inside at least one window as long as
    it is no longer than ``overlap`` characters.
    """
    if not text:
        return []
    if chunk_size <= 0 or not 1 <= overlap < chunk_size:
        raise LocalModelError("invalid_model_configuration", 500)
    chunks: List[Tuple[int, str]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append((start, text[start:end]))
        if end == len(text):
            break
        start = end - overlap
    return chunks


def merge_chunk_entities(entities: Sequence[PhiEntity]) -> List[PhiEntity]:
    """Deterministically collapse duplicates produced by overlapping windows.

    Windows overlap, so the same entity is frequently proposed twice. Identical
    spans collapse to the highest-scoring copy; spans of the same category and
    detector that overlap collapse to the longest, then highest-scoring, copy.
    Cross-detector agreement is intentionally preserved for the pipeline-level
    resolver, so no evidence is discarded here.
    """
    best: Dict[Tuple[str, str, int, int], PhiEntity] = {}
    for entity in entities:
        key = (entity.source, entity.entity_type, entity.start, entity.end)
        current = best.get(key)
        if current is None or (entity.score or 0.0) > (current.score or 0.0):
            best[key] = entity

    grouped: Dict[Tuple[str, str], List[PhiEntity]] = {}
    for entity in best.values():
        grouped.setdefault((entity.source, entity.entity_type), []).append(entity)

    merged: List[PhiEntity] = []
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda item: (
                -(item.end - item.start),
                -(item.score or 0.0),
                item.start,
                item.original_label or "",
            ),
        )
        kept: List[PhiEntity] = []
        for entity in ordered:
            if any(
                entity.start < accepted.end and accepted.start < entity.end
                for accepted in kept
            ):
                continue
            kept.append(entity)
        merged.extend(kept)

    return sorted(
        merged,
        key=lambda item: (
            item.start,
            item.end,
            item.entity_type,
            item.source,
            item.original_label or "",
        ),
    )


def _verify_weight(snapshot_path: str, spec: ModelSpec) -> None:
    weight_path = Path(snapshot_path) / spec.weight_file
    digest = hashlib.sha256()
    try:
        with weight_path.open("rb") as handle:
            for block in iter(lambda: handle.read(_CHECKSUM_READ_SIZE), b""):
                digest.update(block)
    except OSError as exc:
        raise LocalModelError("model_files_unavailable") from exc
    if digest.hexdigest() != spec.weight_sha256:
        raise LocalModelError("model_checksum_mismatch")


def _offline_snapshot(spec: ModelSpec) -> str:
    """Resolve a locally cached snapshot. Never reaches the network."""
    if not local_models_enabled():
        raise LocalModelError("local_models_disabled", 500)
    # Defence in depth: a transitive helper that ignores local_files_only is
    # still forced offline by these flags.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            local_files_only=True,
            **({"allow_patterns": list(spec.allow_patterns)} if spec.allow_patterns else {}),
        )
    except Exception as exc:
        raise LocalModelError("model_files_unavailable") from exc
    _verify_weight(snapshot, spec)
    return snapshot


@lru_cache(maxsize=1)
def load_stanford_pipeline() -> Any:
    """Load the clinical de-identifier once per worker, offline only."""
    spec = load_model_manifest()["stanford_deidentifier"]
    snapshot = _offline_snapshot(spec)
    try:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(
            snapshot,
            local_files_only=True,
        )
        return pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=-1,
        )
    except Exception as exc:
        raise LocalModelError("stanford_model_unavailable") from exc


@lru_cache(maxsize=1)
def load_gliner_model() -> Any:
    """Load the open-ended PII model once per worker, offline only.

    GLiNER's snapshot ships only its own weights and config. At construction
    it resolves a tokenizer backbone *by repository name*, which is a second
    supply-chain input. That backbone is pinned and checksum-verified here
    before the model is built, so an unpinned or tampered tokenizer cannot be
    picked up silently.
    """
    spec = load_model_manifest()["gliner_multi_pii"]
    snapshot = _offline_snapshot(spec)
    for backbone in backbone_specs_for("gliner_multi_pii"):
        _offline_snapshot(backbone)
    try:
        from gliner import GLiNER

        return GLiNER.from_pretrained(snapshot, local_files_only=True)
    except Exception as exc:
        raise LocalModelError("gliner_model_unavailable") from exc


def _normalized_entity(
    raw: Mapping[str, Any],
    label_map: Mapping[str, str],
    offset: int,
    chunk_length: int,
    source: str,
    config: DetectorConfig,
) -> PhiEntity | None:
    """Map one native prediction onto the internal taxonomy.

    Unmapped labels and sub-candidate scores are skipped. Malformed offsets are
    an integrity failure rather than something to silently drop, so they fail
    closed. ``redaction_threshold`` never removes a candidate: while thresholds
    are uncalibrated, every surviving candidate is redacted.
    """
    label = str(raw.get("entity_group") or raw.get("label") or "").strip()
    if label not in label_map:
        return None
    try:
        score = float(raw.get("score", 0.0))
        start = int(raw["start"])
        end = int(raw["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LocalModelError("model_output_malformed", 500) from exc
    if score < config.candidate_threshold:
        return None
    if not 0 <= start < end <= chunk_length:
        raise LocalModelError("model_output_malformed", 500)
    return PhiEntity(
        entity_type=label_map[label],
        start=start + offset,
        end=end + offset,
        source=source,
        score=score,
        original_label=label,
    )


def _batches(items: Sequence[Any], batch_size: int) -> List[Sequence[Any]]:
    return [
        items[index : index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


def _run_batched(
    chunks: Sequence[Tuple[int, str]],
    config: DetectorConfig,
    runner: Callable[[List[str]], Sequence[Sequence[Mapping[str, Any]]]],
    failure_code: str,
) -> List[Sequence[Mapping[str, Any]]]:
    """Run inference batch by batch under a wall-clock budget, failing closed."""
    deadline = time.monotonic() + config.inference_budget_seconds
    predictions: List[Sequence[Mapping[str, Any]]] = []
    for batch in _batches(list(chunks), config.batch_size):
        if time.monotonic() > deadline:
            raise LocalModelError("model_inference_timeout")
        try:
            batch_predictions = runner([chunk for _, chunk in batch])
        except LocalModelError:
            raise
        except Exception as exc:
            raise LocalModelError(failure_code, 500) from exc
        if len(batch_predictions) != len(batch):
            raise LocalModelError("model_output_malformed", 500)
        predictions.extend(batch_predictions)
        if time.monotonic() > deadline:
            raise LocalModelError("model_inference_timeout")
    return predictions


class _ChunkedModelDetector:
    """Shared overlapping-chunk, batch-aware, fail-closed detection loop."""

    source = ""
    label_map: Mapping[str, str] = {}
    failure_code = "model_inference_failed"

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or detector_config_from_env()

    def _infer(self, texts: List[str]) -> Sequence[Sequence[Mapping[str, Any]]]:
        raise NotImplementedError

    def detect(self, text: str) -> List[PhiEntity]:
        chunks = overlapping_chunks(
            text, self.config.chunk_size, self.config.chunk_overlap
        )
        if not chunks:
            return []
        predictions = _run_batched(chunks, self.config, self._infer, self.failure_code)
        entities: List[PhiEntity] = []
        for (offset, chunk), chunk_predictions in zip(chunks, predictions):
            for raw in chunk_predictions:
                entity = _normalized_entity(
                    raw,
                    self.label_map,
                    offset,
                    len(chunk),
                    self.source,
                    self.config,
                )
                if entity is not None:
                    entities.append(entity)
        return merge_chunk_entities(entities)


class StanfordClinicalDetector(_ChunkedModelDetector):
    source = SOURCE_STANFORD
    label_map = STANFORD_LABELS
    failure_code = "stanford_inference_failed"

    def _infer(self, texts: List[str]) -> Sequence[Sequence[Mapping[str, Any]]]:
        pipeline = load_stanford_pipeline()
        results = pipeline(texts, batch_size=self.config.batch_size)
        if results and isinstance(results[0], Mapping):
            # A single-input call can return one flat list of predictions.
            return [results]
        return results


class GlinerPiiDetector(_ChunkedModelDetector):
    source = SOURCE_GLINER
    label_map = GLINER_LABELS
    failure_code = "gliner_inference_failed"

    def _infer(self, texts: List[str]) -> Sequence[Sequence[Mapping[str, Any]]]:
        model = load_gliner_model()
        if hasattr(model, "batch_predict_entities"):
            return model.batch_predict_entities(
                texts,
                list(GLINER_REQUESTED_LABELS),
                threshold=self.config.candidate_threshold,
                batch_size=self.config.batch_size,
            )
        return [
            model.predict_entities(
                chunk,
                list(GLINER_REQUESTED_LABELS),
                threshold=self.config.candidate_threshold,
            )
            for chunk in texts
        ]
