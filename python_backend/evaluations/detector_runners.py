"""Detector adapters for the real-model evaluation.

Each runner turns one detector configuration into a uniform list of
``Detection`` objects so the same scoring code can compare them. Five
configurations are measured separately, because "the pipeline finds it" and
"this particular model finds it" are different claims:

* ``rules``      - deterministic structured patterns only.
* ``spacy``      - the trained spaCy NER plus its context rules.
* ``stanford``   - the pinned clinical de-identifier alone.
* ``gliner``     - the pinned open-ended PII model alone.
* ``combined``   - the production chain, after overlap resolution.

Timing and memory are captured per document so latency percentiles and peak
memory are measured rather than estimated. No runner ever returns matched
text: a ``Detection`` carries offsets, a category, a source and a score.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from evaluations.metrics import Detection

# --------------------------------------------------------------------------
# Threshold plumbing
# --------------------------------------------------------------------------


def set_model_thresholds(candidate: float, redaction: Optional[float] = None) -> None:
    """Set detector thresholds for the current process.

    The detector config is read from the environment at construction time and
    the chain is ``lru_cache``d, so both must be reset together.
    """
    redaction = candidate if redaction is None else redaction
    os.environ["PHI_CANDIDATE_THRESHOLD"] = str(candidate)
    os.environ["PHI_REDACTION_THRESHOLD"] = str(redaction)
    _clear_detector_caches()


def clear_threshold_overrides() -> None:
    """Drop environment overrides so the locked calibration applies.

    ``calibrated_config_for`` gives the environment precedence over the
    committed calibration, so a leftover override would silently replace the
    configuration being measured.
    """
    for name in ("PHI_CANDIDATE_THRESHOLD", "PHI_REDACTION_THRESHOLD"):
        os.environ.pop(name, None)
    _clear_detector_caches()


def _clear_detector_caches() -> None:
    from services import local_model_detectors, text_anonymization

    text_anonymization._build_detectors.cache_clear()
    local_model_detectors.load_locked_thresholds.cache_clear()


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


def peak_rss_bytes() -> int:
    """Current process peak working set, in bytes. Windows-aware."""
    try:
        import ctypes
        from ctypes import wintypes

        class _COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _COUNTERS()
        counters.cb = ctypes.sizeof(_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        # Modern Windows forwards this from kernel32; psapi.dll is not always
        # loadable. argtypes/restype must be declared or the call marshals
        # wrongly and quietly reports zero.
        for getter in (
            getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None),
            getattr(getattr(ctypes.windll, "psapi", None), "GetProcessMemoryInfo", None),
        ):
            if getter is None:
                continue
            getter.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_COUNTERS),
                wintypes.DWORD,
            ]
            getter.restype = wintypes.BOOL
            if getter(handle, ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
    except Exception:
        pass
    try:  # pragma: no cover - non-Windows fallback
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return 0


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------


@dataclass
class RunnerResult:
    detections_by_doc: Dict[str, List[Detection]] = field(default_factory=dict)
    latencies_ms: List[float] = field(default_factory=list)
    load_seconds: float = 0.0
    peak_rss_bytes: int = 0
    failures: List[Dict[str, str]] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)


def _to_detections(entities: Sequence[Any]) -> List[Detection]:
    return [
        Detection(
            start=int(entity.start),
            end=int(entity.end),
            category=str(entity.entity_type),
            source=str(entity.source),
            score=None if entity.score is None else float(entity.score),
        )
        for entity in entities
    ]


def _run_over(
    documents: Sequence[Any],
    detect: Callable[[str], Sequence[Any]],
    load_seconds: float = 0.0,
) -> RunnerResult:
    result = RunnerResult(load_seconds=load_seconds)
    for document in documents:
        started = time.perf_counter()
        try:
            entities = detect(document.text)
        except Exception as exc:
            # A detector failure is recorded, never silently treated as "no
            # PHI here" - that would score a crash as a clean document.
            result.failures.append(
                {"doc_id": document.doc_id, "error": type(exc).__name__,
                 "code": getattr(exc, "error_code", getattr(exc, "detail", ""))}
            )
            result.detections_by_doc[document.doc_id] = []
            result.unsupported.append(document.doc_id)
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000
        result.latencies_ms.append(round(elapsed_ms, 2))
        result.detections_by_doc[document.doc_id] = _to_detections(entities)
    result.peak_rss_bytes = peak_rss_bytes()
    return result


def run_rules(documents: Sequence[Any]) -> RunnerResult:
    from services.phi_detection import StructuredPatternDetector

    detector = StructuredPatternDetector()
    return _run_over(documents, detector.detect)


def run_spacy(documents: Sequence[Any]) -> RunnerResult:
    from services.ner_phi_detector import SpacyNerPhiDetector, configured_model_name

    started = time.perf_counter()
    detector = SpacyNerPhiDetector(configured_model_name(), high_recall_proper_nouns=True)
    # Force the pipeline to load now so load time is not billed to document 1.
    detector.detect("warm up")
    load_seconds = time.perf_counter() - started
    return _run_over(documents, detector.detect, load_seconds)


def run_stanford(documents: Sequence[Any], config: Any = None) -> RunnerResult:
    from services.local_model_detectors import StanfordClinicalDetector, load_stanford_pipeline

    started = time.perf_counter()
    load_stanford_pipeline()
    load_seconds = time.perf_counter() - started
    detector = StanfordClinicalDetector(config)
    return _run_over(documents, detector.detect, load_seconds)


def run_gliner(documents: Sequence[Any], config: Any = None) -> RunnerResult:
    from services.local_model_detectors import GlinerPiiDetector, load_gliner_model

    started = time.perf_counter()
    load_gliner_model()
    load_seconds = time.perf_counter() - started
    detector = GlinerPiiDetector(config)
    return _run_over(documents, detector.detect, load_seconds)


def run_combined(documents: Sequence[Any]) -> RunnerResult:
    """The production chain, including pipeline-level overlap resolution."""
    from services.ner_phi_detector import configured_model_name
    from services.text_anonymization import _detect_entities

    model_name = configured_model_name()
    started = time.perf_counter()
    _detect_entities("warm up", model_name, "strict")
    load_seconds = time.perf_counter() - started

    return _run_over(
        documents,
        lambda text: _detect_entities(text, model_name, "strict"),
        load_seconds,
    )


def run_residual_validator(documents: Sequence[Any]) -> Dict[str, Any]:
    """Redact each document, then re-scan the output for surviving PHI.

    This is the end-to-end privacy question: after the whole pipeline runs,
    does anything the detectors can still identify remain?
    """
    from services.text_anonymization import (
        anonymize_clinical_text,
        residual_phi_categories,
    )

    residual_docs: List[str] = []
    residual_categories: Dict[str, int] = {}
    surviving_gold: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    latencies: List[float] = []

    for document in documents:
        started = time.perf_counter()
        try:
            outcome = anonymize_clinical_text(
                document.text, profile="strict", study_salt="evaluation-salt"
            )
            redacted = outcome["anonymized_text"]
            residual = residual_phi_categories(redacted)
        except Exception as exc:
            failures.append(
                {"doc_id": document.doc_id, "error": type(exc).__name__,
                 "code": str(getattr(exc, "detail", ""))}
            )
            continue
        latencies.append(round((time.perf_counter() - started) * 1000, 2))

        if residual:
            residual_docs.append(document.doc_id)
            for name, count in residual.items():
                residual_categories[name] = residual_categories.get(name, 0) + count

        # The decisive check: did any literal gold value survive redaction?
        for span in document.spans:
            value = document.value(span)
            if value and value in redacted:
                surviving_gold.append(
                    {
                        "doc_id": document.doc_id,
                        "category": span.category,
                        "tags": list(span.tags),
                    }
                )

    return {
        "documents": len(documents),
        "documents_with_residual_findings": sorted(residual_docs),
        "residual_categories": dict(sorted(residual_categories.items())),
        # Counts and categories only. The value itself is never reported.
        "surviving_gold_values": surviving_gold,
        "surviving_gold_count": len(surviving_gold),
        "failures": failures,
        "latencies_ms": latencies,
    }


def release_memory() -> None:
    """Drop cached models between configurations on a memory-tight machine."""
    from services import local_model_detectors as models

    for loader in (models.load_stanford_pipeline, models.load_gliner_model):
        try:
            loader.cache_clear()
        except Exception:
            pass
    _clear_detector_caches()
    gc.collect()
