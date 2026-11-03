"""Stage timing for the production request path.

The evaluation harness already reports a per-document mean, but that number
answers a different question: it times three detector passes plus utility
scoring in a loop, with no HTTP layer, no routing and no release decision. It
cannot tell you what a user waits for.

This records where wall-clock time actually goes inside one API request, and
nothing else. It is off unless ``PHI_LATENCY_METRICS`` is set, and when it is
off ``stage()`` returns a shared do-nothing object, so the instrumented path
costs one ``ContextVar.get()`` per stage.

What it may hold is deliberately narrow: a stage name, a duration, a nesting
depth, and a few coarse labels chosen so that no record can identify a
document. There is no field for text, filenames, entity values, surrogate
mappings or model input, because a timing record that could carry PHI would
be a new way to leak it.

Nesting is recorded rather than implied. ``depth`` says how deep a stage sat,
so a reader can see that ``detect.stanford`` is inside ``detect.pass1`` and
must not be added to it.
"""

from __future__ import annotations

import contextvars
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ENABLE_ENV_VAR = "PHI_LATENCY_METRICS"

#: Size buckets. A bucket, never a byte count, because an exact length is a
#: weak identifier for a known document.
_SIZE_BUCKETS: Tuple[Tuple[int, str], ...] = (
    (2_048, "tiny"),
    (32_768, "small"),
    (131_072, "medium"),
    (262_144, "large"),
)


def size_bucket(byte_length: int) -> str:
    for limit, name in _SIZE_BUCKETS:
        if byte_length <= limit:
            return name
    return "oversize"


def metrics_enabled() -> bool:
    return os.getenv(ENABLE_ENV_VAR, "0").strip() == "1"


@dataclass(frozen=True)
class StageTiming:
    """One measured stage. Carries no document-derived content."""

    name: str
    depth: int
    duration_ns: int

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000

    def to_public_dict(self) -> Dict[str, object]:
        return {
            "stage": self.name,
            "depth": self.depth,
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass
class RequestTimings:
    """Stage timings for one request, plus safe labels."""

    request_id: str = ""
    modality: str = "unknown"
    size_bucket: str = "unknown"
    concurrency: int = 1
    model_ready: bool = False
    outcome: str = "unknown"
    release_category: str = "unknown"
    stages: List[StageTiming] = field(default_factory=list)
    _depth: int = 0

    def stage(self, name: str) -> "_Stage":
        return _Stage(self, name)

    def record(self, name: str, depth: int, duration_ns: int) -> None:
        self.stages.append(StageTiming(name, depth, duration_ns))

    def total_ms(self, name: str) -> float:
        return sum(s.duration_ms for s in self.stages if s.name == name)

    def call_count(self, name: str) -> int:
        return sum(1 for s in self.stages if s.name == name)

    def to_public_dict(self) -> Dict[str, object]:
        return {
            "request_id": self.request_id,
            "modality": self.modality,
            "size_bucket": self.size_bucket,
            "concurrency": self.concurrency,
            "model_ready": self.model_ready,
            "outcome": self.outcome,
            "release_category": self.release_category,
            "stages": [s.to_public_dict() for s in self.stages],
        }


class _Stage:
    """Times one stage and restores the parent depth on the way out."""

    __slots__ = ("_owner", "_name", "_started", "_depth")

    def __init__(self, owner: RequestTimings, name: str):
        self._owner = owner
        self._name = name
        self._started = 0
        self._depth = 0

    def __enter__(self) -> "_Stage":
        self._depth = self._owner._depth
        self._owner._depth += 1
        self._started = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.perf_counter_ns() - self._started
        self._owner._depth = self._depth
        # Recorded even when the body raised: a stage that failed still
        # consumed time, and a benchmark that silently dropped it would
        # understate the cost of the failure path.
        self._owner.record(self._name, self._depth, elapsed)
        return False


class _NullStage:
    """What ``stage()`` returns when nothing is collecting."""

    __slots__ = ()

    def __enter__(self) -> "_NullStage":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


_NULL_STAGE = _NullStage()

#: Request-scoped, so concurrent requests never share a record. A ContextVar
#: is copied into the worker thread FastAPI runs sync endpoints on, which a
#: module-level global would not be.
_ACTIVE: contextvars.ContextVar[Optional[RequestTimings]] = contextvars.ContextVar(
    "phi_latency_request", default=None
)


def start_request(
    request_id: str = "",
    modality: str = "unknown",
    size_bucket_name: str = "unknown",
    concurrency: int = 1,
) -> Optional[RequestTimings]:
    """Begin collecting, if collection is switched on."""
    if not metrics_enabled():
        return None
    timings = RequestTimings(
        request_id=request_id,
        modality=modality,
        size_bucket=size_bucket_name,
        concurrency=concurrency,
    )
    _ACTIVE.set(timings)
    return timings


def finish_request() -> Optional[RequestTimings]:
    timings = _ACTIVE.get()
    _ACTIVE.set(None)
    return timings


def current_request() -> Optional[RequestTimings]:
    return _ACTIVE.get()


def stage(name: str):
    """Time a stage, or do nothing at all when collection is off."""
    timings = _ACTIVE.get()
    if timings is None:
        return _NULL_STAGE
    return timings.stage(name)


def annotate(**labels: object) -> None:
    """Attach a safe label to the active record. Unknown names are ignored."""
    timings = _ACTIVE.get()
    if timings is None:
        return
    for key, value in labels.items():
        if key in {"modality", "size_bucket", "outcome", "release_category"}:
            setattr(timings, key, str(value))
        elif key == "model_ready":
            timings.model_ready = bool(value)
        elif key == "concurrency":
            timings.concurrency = int(value)
