"""Prepare the model worker before the first user request.

Phase 12A measured a cold first request at 35.3 s against a warm 1 KB request
at 4.0 s. Almost all of that gap was the worker starting, verifying checksums
and loading two models, paid by whichever user happened to arrive first.

This moves that work to service startup. It does not make the work cheaper —
the cost is the same, it simply lands before the service reports itself ready
instead of inside somebody's upload. Readiness stays false until verification,
loading and a synthetic warm-up have all succeeded, so nothing is accepted
early.

Liveness and readiness are deliberately separate. The process being alive says
nothing about whether the models are usable, and a health check that conflated
the two would route traffic at a backend that can only fail closed.

Failure never degrades to unprotected processing. If loading or warm-up fails,
readiness stays false and the recorded reason is an error code; requests
continue to fail closed exactly as they already did.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from services.local_model_detectors import (
    MODE_OFFLINE,
    LocalModelError,
    resolve_model_mode,
)
from services.model_client import (
    current_client,
    get_client,
    shutdown_client,
    worker_enabled,
)
from services.request_deadline import (
    DeadlineConfigurationError,
)
from services.request_deadline import (
    validate_configuration as _validate_request_deadline,
)
from services.residual_profile import (
    ResidualProfileError,
)
from services.residual_profile import (
    validate_configuration as _validate_validator_profile,
)

#: Prewarming follows the worker: with no worker there is nothing to warm, and
#: the mocked detectors used by the ordinary suite must not start subprocesses.
PREWARM_ENV_VAR = "PHI_MODEL_PREWARM"

#: Short, invented, and free of anything resembling an identifier. It exists to
#: force the first forward pass, not to test detection.
WARMUP_TEXT = "The clinic reviewed the routine observation record."

STATE_UNINITIALIZED = "uninitialized"
STATE_PREPARING = "preparing"
STATE_READY = "ready"
STATE_FAILED = "failed"
ERROR_PRODUCTION_WORKER_REQUIRED = "production_model_worker_required"
ERROR_PRODUCTION_PREWARM_REQUIRED = "production_model_prewarm_required"


@dataclass
class LifecycleState:
    """What the service knows about model readiness.

    Durations are recorded per stage so startup, verification and loading, and
    warm-up can be reported separately rather than as one opaque number.
    """

    state: str = STATE_UNINITIALIZED
    validator_profile: str = ""
    #: An error *code*, never a raw exception, path or hash.
    error_code: Optional[str] = None
    worker_start_ms: float = 0.0
    spacy_load_ms: float = 0.0
    warmup_ms: float = 0.0
    total_ms: float = 0.0
    models: tuple = ()

    def is_ready(self) -> bool:
        if self.state != STATE_READY:
            return False
        # A worker being replaced, or one that died, is not ready to serve
        # even though startup once succeeded.
        client = current_client()
        return client is None or client.is_ready()

    def to_public_dict(self) -> Dict[str, object]:
        """Safe for a readiness endpoint: no paths, hashes or raw errors."""
        from services.detector_capability import current_capability

        return {
            "ready": self.is_ready(),
            "state": self.state,
            "error_code": self.error_code,
            # Whether results can ever be released; a capability, never a name.
            "detector_capability": current_capability(),
            "models": list(self.models),
            "validator_profile": self.validator_profile,
            "worker_start_ms": round(self.worker_start_ms, 2),
            "spacy_load_ms": round(self.spacy_load_ms, 2),
            "warmup_ms": round(self.warmup_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "inference": inference_status(),
        }


#: What an ordinary user is told, and nothing more.
STATUS_ACCELERATED = "Hardware acceleration active"
STATUS_STANDARD = "Running in standard mode"
STATUS_DEGRADED = "Temporarily using standard mode"


def inference_status() -> Dict[str, object]:
    """Where the models actually run, from the worker's own report.

    Hardware detection, a CUDA build and active acceleration are separate
    facts: only models verified on the GPU count as acceleration. No path, no
    process detail beyond counts, and never anything from a request.
    """
    client = current_client()
    runtime = dict(client.runtime) if client is not None else {}
    device = dict(runtime.get("device") or {})
    counters = client.status() if client is not None else {}
    active = bool(device.get("acceleration_active"))
    degraded = bool(device.get("degraded"))
    models = device.get("model_devices") or {}
    return {
        "requested_device": device.get("requested"),
        "selected_device": device.get("selected") or ("cpu" if runtime else None),
        "torch_build": device.get("torch_build"),
        "cuda_build": bool(device.get("cuda_build")),
        "cuda_runtime": device.get("cuda_runtime"),
        "gpu_visible": bool(device.get("gpu_visible")),
        "gpu_name": device.get("gpu_name"),
        "total_vram_bytes": device.get("total_vram_bytes"),
        "free_vram_bytes": device.get("free_vram_bytes"),
        "allocated_vram_bytes": device.get("allocated_vram_bytes"),
        "reserved_vram_bytes": device.get("reserved_vram_bytes"),
        "peak_allocated_vram_bytes": device.get("peak_allocated_vram_bytes"),
        "stanford_device": models.get("stanford"),
        "gliner_device": models.get("gliner"),
        "precision": device.get("precision"),
        "acceleration_active": active,
        "degraded": degraded,
        "reason": device.get("reason"),
        "model_batch_size": runtime.get("model_batch_size"),
        "inference_concurrency": 1,
        "worker_generation": counters.get("generation"),
        "model_load_count": runtime.get("load_count"),
        "worker_restarts": counters.get("restarts", 0),
        "model_requests": counters.get("detect_requests", 0),
        "oom_events": device.get("oom_events", 0),
        "cpu_fallbacks": device.get("cpu_fallbacks", 0),
        "status": (
            STATUS_DEGRADED
            if degraded
            else STATUS_ACCELERATED if active else STATUS_STANDARD
        ),
    }


_STATE = LifecycleState()
_LOCK = threading.Lock()


def _prewarm_spacy() -> float:
    """Load the spaCy pipeline and its matcher in this process.

    The worker holds the two pinned transformers, but the API process runs
    spaCy itself and was loading it lazily inside the first request. Warming
    the pinned models alone left roughly four seconds of first-request cost
    that nothing had accounted for.

    Failure leaves readiness false; a missing required detector must not be
    presented as successful warm-up.
    """
    started = time.perf_counter()
    try:
        from services.ner_phi_detector import (
            _context_matcher,
            configured_model_name,
            load_spacy_pipeline,
        )

        model_name = configured_model_name()
        load_spacy_pipeline(model_name)
        _context_matcher(model_name)
    except Exception:
        raise LocalModelError("spacy_prewarm_failed", 503) from None
    return (time.perf_counter() - started) * 1000


def prewarm_enabled() -> bool:
    configured = os.getenv(PREWARM_ENV_VAR)
    if configured is not None:
        return configured.strip() == "1"
    return worker_enabled()


def _require_production_worker(mode: str) -> None:
    if mode != MODE_OFFLINE:
        return
    if not worker_enabled():
        raise LocalModelError(ERROR_PRODUCTION_WORKER_REQUIRED, 503)
    if not prewarm_enabled():
        raise LocalModelError(ERROR_PRODUCTION_PREWARM_REQUIRED, 503)


def state() -> LifecycleState:
    return _STATE


def reset_state() -> None:
    """Return to uninitialized. For tests that drive the lifecycle directly."""
    global _STATE
    with _LOCK:
        _STATE = LifecycleState()


def prewarm() -> LifecycleState:
    """Start and warm the worker. Safe to call more than once.

    Returns the resulting state rather than raising, because a failure here
    must leave the service running and not-ready, not crash it into a restart
    loop. The blocking behaviour that protects artifacts lives in the request
    path, which already fails closed when the worker is unavailable.
    """
    global _STATE
    with _LOCK:
        if _STATE.state in {STATE_READY, STATE_PREPARING}:
            return _STATE
        if not prewarm_enabled():
            # No worker to prepare, but spaCy still runs in this process and
            # is worth loading before the first request either way.
            try:
                validator_profile = _validate_validator_profile()
                # A deadline that cannot be honoured is a configuration
                # fault, and readiness is where it should show up - not on
                # the first request that needs it.
                _validate_request_deadline()
                # So is an unacknowledged switch to the test detectors.
                mode = resolve_model_mode()
                # Production readiness is evidence that the pinned worker
                # loaded both snapshots. Configuration validation alone
                # cannot prove that the interpreter or cache is usable.
                _require_production_worker(mode)
            except (
                ResidualProfileError,
                DeadlineConfigurationError,
                LocalModelError,
            ) as exc:
                _STATE = LifecycleState(state=STATE_FAILED, error_code=exc.error_code)
                return _STATE
            try:
                spacy_ms = _prewarm_spacy()
            except LocalModelError as exc:
                _STATE = LifecycleState(state=STATE_FAILED, error_code=exc.error_code)
                return _STATE
            _STATE = LifecycleState(
                state=STATE_READY,
                models=(),
                spacy_load_ms=spacy_ms,
                total_ms=spacy_ms,
                validator_profile=validator_profile,
            )
            return _STATE
        _STATE = LifecycleState(state=STATE_PREPARING)

    started = time.perf_counter()
    try:
        # Checked before anything is loaded. A profile or deadline that cannot
        # be honoured leaves readiness false rather than quietly selecting a
        # default.
        validator_profile = _validate_validator_profile()
        _validate_request_deadline()
        mode = resolve_model_mode()
        _require_production_worker(mode)
    except (ResidualProfileError, DeadlineConfigurationError, LocalModelError) as exc:
        with _LOCK:
            _STATE = LifecycleState(state=STATE_FAILED, error_code=exc.error_code)
        return _STATE

    try:
        client = get_client()
        # start() blocks until the worker's ready banner arrives, which the
        # worker sends only after checksum verification and both model loads.
        client.start()
        worker_started = time.perf_counter()
        spacy_ms = _prewarm_spacy()

        # One synthetic inference per detector, so the first real request does
        # not pay lazy initialisation inside the model libraries.
        for detector in ("stanford", "gliner"):
            client.detect(detector, WARMUP_TEXT)
        # The worker's own report after warm-up: device, memory, load count.
        client.ping()
        finished = time.perf_counter()

        with _LOCK:
            _STATE = LifecycleState(
                state=STATE_READY,
                worker_start_ms=(worker_started - started) * 1000,
                spacy_load_ms=spacy_ms,
                warmup_ms=(finished - worker_started) * 1000 - spacy_ms,
                total_ms=(finished - started) * 1000,
                models=("stanford", "gliner"),
                validator_profile=validator_profile,
            )
    except LocalModelError as exc:
        with _LOCK:
            _STATE = LifecycleState(
                state=STATE_FAILED,
                error_code=str(exc.error_code),
                total_ms=(time.perf_counter() - started) * 1000,
            )
    except Exception:
        # Deliberately no exception text: it can carry a filesystem path.
        with _LOCK:
            _STATE = LifecycleState(
                state=STATE_FAILED,
                error_code="model_prewarm_failed",
                total_ms=(time.perf_counter() - started) * 1000,
            )
    return _STATE


def shutdown() -> None:
    """Stop the worker and forget readiness, leaving no stray process."""
    global _STATE
    try:
        shutdown_client()
    finally:
        with _LOCK:
            _STATE = LifecycleState()
