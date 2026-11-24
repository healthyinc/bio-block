"""One absolute deadline for a whole anonymization request.

Every budget in the pipeline used to be a *per-call* budget. ``_run_batched``
started a fresh 120-second window each time a detector's ``detect()`` was
invoked, and a request invokes that four times - two passes across two
models - so the budget reset four times. The worker client's read timeout
reset per call as well, and nothing above the detectors measured the request
as a whole. Four calls that each finish inside their own window can still add
up to an hour, which is how one request reached 2,846 seconds without a
single timeout firing.

This adds the thing that was missing: a single deadline, created once at the
trusted backend boundary, that every later stage reads and none may extend.
The per-call budgets stay, but only as stricter sub-limits - whichever is
nearer wins, and the request deadline can never be pushed further out.

It is server configuration. The deadline is created from the environment at
the API boundary and stored in a ContextVar, so an HTTP field cannot set,
extend or disable it, and concurrent requests cannot see each other's.

Expiry always blocks. There is no path here that returns a partial result:
callers convert the raised error into the same fail-closed handling that a
model failure already gets.
"""

from __future__ import annotations

import contextvars
import os
import time
from dataclasses import dataclass
from typing import Optional

#: Server-side only. Never populated from request data.
DEADLINE_ENV_VAR = "PHI_REQUEST_DEADLINE_SECONDS"

#: Chosen against measured latency and the 256 KB text limit rather than to
#: make any particular run pass:
#:
#:   warm 1 KB p95      2.7 s (optimized) / 4.3 s (full)
#:   warm 10 KB p95    34.3 s (optimized) / 57.8 s (full)
#:   100 KB           ~181 s, 200 KB ~212 s  (both already over budget in 12A)
#:
#: 120 s would reject 100 KB uploads the service currently completes, and 180 s
#: leaves no headroom above a 200 KB document. 300 s covers the largest upload
#: the route accepts with roughly 40% margin, while still bounding a request to
#: something a human would wait through. Documents near the top of that range
#: remain impractical for other reasons, which Phase 12A already recorded.
DEFAULT_DEADLINE_SECONDS = 300.0

#: A deadline is only meaningful inside these bounds. Outside them the
#: configuration is wrong, and startup says so rather than guessing.
MIN_DEADLINE_SECONDS = 5.0
MAX_DEADLINE_SECONDS = 1800.0

#: One stable machine-readable category for the whole-request case. The
#: narrower per-stage codes below let metrics separate where time went.
ERROR_REQUEST_TIMEOUT = "request_deadline_exceeded"
ERROR_QUEUE_TIMEOUT = "request_queue_timeout"
ERROR_WORKER_TIMEOUT = "model_worker_timeout"


class DeadlineConfigurationError(ValueError):
    """A configured deadline that cannot be honoured."""

    def __init__(self, error_code: str = "invalid_request_deadline"):
        super().__init__(error_code)
        self.error_code = error_code


class RequestDeadlineExceeded(RuntimeError):
    """The request ran out of time. Carries a code and a stage, never PHI."""

    def __init__(self, stage: str = "", error_code: str = ERROR_REQUEST_TIMEOUT):
        super().__init__(error_code)
        self.error_code = error_code
        #: Coarse stage name for metrics. Not returned to clients.
        self.stage = stage
        self.status_code = 503


def configured_seconds() -> float:
    """The configured deadline. Raises rather than falling back on nonsense."""
    raw = os.getenv(DEADLINE_ENV_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_DEADLINE_SECONDS
    try:
        value = float(raw.strip())
    except (TypeError, ValueError) as exc:
        raise DeadlineConfigurationError() from exc
    if not MIN_DEADLINE_SECONDS <= value <= MAX_DEADLINE_SECONDS:
        raise DeadlineConfigurationError()
    return value


def validate_configuration() -> float:
    """Checked at startup so a bad value fails readiness, not a request."""
    return configured_seconds()


@dataclass(frozen=True)
class Deadline:
    """An absolute point on the monotonic clock."""

    expires_at: float
    total_seconds: float

    def remaining(self) -> float:
        return self.expires_at - time.monotonic()

    def expired(self) -> bool:
        return self.remaining() <= 0


#: Request-scoped. A ContextVar is copied into the worker thread FastAPI runs
#: sync endpoints on, and is not shared between concurrent requests.
_DEADLINE: contextvars.ContextVar[Optional[Deadline]] = contextvars.ContextVar(
    "phi_request_deadline", default=None
)


def start(seconds: Optional[float] = None) -> Deadline:
    """Begin the one deadline for this request.

    Called at the trusted boundary only. ``seconds`` exists for tests and for
    a caller that already holds validated configuration; it is never derived
    from request data.
    """
    total = configured_seconds() if seconds is None else float(seconds)
    deadline = Deadline(expires_at=time.monotonic() + total, total_seconds=total)
    _DEADLINE.set(deadline)
    return deadline


def clear() -> None:
    _DEADLINE.set(None)


def current() -> Optional[Deadline]:
    return _DEADLINE.get()


def remaining() -> Optional[float]:
    """Seconds left, or None when no deadline is active."""
    deadline = _DEADLINE.get()
    return None if deadline is None else deadline.remaining()


def check(stage: str = "") -> None:
    """Raise if the request has run out of time.

    Called between stages. It cannot interrupt a call already blocked inside
    a model, which is why the worker client also enforces the remaining time
    on its own read.
    """
    deadline = _DEADLINE.get()
    if deadline is not None and deadline.expired():
        raise RequestDeadlineExceeded(stage=stage)


def budget_for(stage: str, sub_limit: Optional[float] = None) -> Optional[float]:
    """The time a stage may take: the nearer of its sub-limit and what is left.

    A stage may be stricter than the request deadline but never more generous,
    which is what stops a per-call budget from resetting the whole request's
    clock.
    """
    check(stage)
    left = remaining()
    if left is None:
        return sub_limit
    if left <= 0:
        raise RequestDeadlineExceeded(stage=stage)
    return left if sub_limit is None else min(sub_limit, left)
