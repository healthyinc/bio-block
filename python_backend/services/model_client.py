"""Client for the out-of-process model worker.

Runs in the main FastAPI environment and talks to a child process started
under the model virtualenv. The main environment therefore never imports
torch, transformers or gliner, and ChromaDB's pins are left alone.

Every failure path blocks. A worker that will not start, will not become
ready, times out, dies mid-request or answers with an error yields
``LocalModelError``, which the text pipeline already turns into a blocked
artifact. There is no path here that returns "no PHI found" because the
worker was unavailable.

The protocol (version 2) correlates by request id, not by position. One reader
thread per worker generation reads every line the worker writes and hands it
to the caller waiting on that id. A caller whose time runs out *abandons* its
request rather than killing the worker: the answer arrives later, is
recognised by its id and discarded, and the worker - which was only busy -
serves the next request. Only a request still unanswered past its limit plus a
grace period, a crash, or a protocol violation replaces the worker, through
one supervisor path that starts exactly one successor.

Inference stays serial: one request is in the worker at a time, so the models
are never driven concurrently.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.local_model_detectors import (
    DEFAULT_INFERENCE_BUDGET_SECONDS,
    LocalModelError,
)
from services.privacy_contracts import PhiEntity
from services.request_deadline import ERROR_WORKER_TIMEOUT, budget_for
from services.request_deadline import remaining as request_remaining

BACKEND_ROOT = Path(__file__).resolve().parent.parent

#: Interpreter that owns the model stack. Overridable for deployment.
WORKER_PYTHON_ENV_VAR = "PHI_MODEL_WORKER_PYTHON"
DEFAULT_WORKER_PYTHON = BACKEND_ROOT / ".venv-models" / "Scripts" / "python.exe"
DEFAULT_WORKER_PYTHON_POSIX = BACKEND_ROOT / ".venv-models" / "bin" / "python"

#: Enables the out-of-process path. Absent, the in-process detectors are used,
#: which is what the mocked test suite relies on.
WORKER_ENABLED_ENV_VAR = "PHI_MODEL_WORKER"

PROTOCOL_VERSION = 2
GENERATION_ENV_VAR = "BIOBLOCK_WORKER_GENERATION"
GRACE_ENV_VAR = "BIOBLOCK_MODEL_WORKER_GRACE_SECONDS"
DEFAULT_GRACE_SECONDS = 30.0

STARTUP_TIMEOUT_SECONDS = 300.0
REQUEST_TIMEOUT_SECONDS = DEFAULT_INFERENCE_BUDGET_SECONDS

#: Callers allowed to wait for the one in-flight slot. More is a queue with no
#: end in sight; they are refused as retryable instead.
MAX_PENDING_REQUESTS = 8
#: Abandoned ids remembered so their late answers are recognised and dropped.
MAX_ABANDONED_REMEMBERED = 64

ERROR_WORKER_CRASHED = "model_worker_crashed"
ERROR_WORKER_RESTARTED = "model_worker_restarted"
ERROR_WORKER_UNRESPONSIVE = "model_worker_unresponsive"
ERROR_WORKER_BUSY = "model_worker_busy"
ERROR_PROTOCOL = "model_worker_protocol_error"
#: A call refused or abandoned because the file's own deadline ran out.
ERROR_REQUEST_DEADLINE = "request_deadline_exceeded"
#: Below this, a model call cannot usefully start inside the file's deadline.
MIN_CALL_SECONDS = 2.0


def request_timeout_seconds() -> float:
    """The worker RPC limit, shared with the detector's inference budget.

    The request deadline is still authoritative and ``budget_for`` takes the
    nearer limit.  Reading the environment for every call keeps tests and a
    deliberately stricter deployment deterministic without restarting a
    second, differently configured worker.
    """
    raw = os.getenv("PHI_MODEL_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return REQUEST_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise LocalModelError("invalid_model_configuration", 500) from exc
    if not 5.0 <= value <= 1800.0:
        raise LocalModelError("invalid_model_configuration", 500)
    return value


def grace_seconds() -> float:
    """How long past its limit an unanswered request is given before replacement."""
    raw = os.getenv(GRACE_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_GRACE_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise LocalModelError("invalid_model_configuration", 500) from exc
    if not 0.0 <= value <= 600.0:
        raise LocalModelError("invalid_model_configuration", 500)
    return value


#: How long a polite shutdown is given before signals are used.
GRACEFUL_SHUTDOWN_SECONDS = 10.0
#: How long terminate is given before kill, and kill before giving up.
TERMINATE_GRACE_SECONDS = 5.0
#: One in-flight request at a time. The worker is a single process holding two
#: models on a memory-tight machine; queueing is safer than contention.
MAX_CONCURRENT_REQUESTS = 1
#: The watchdog's polling interval.
WATCHDOG_INTERVAL_SECONDS = 0.25


def worker_python() -> Optional[Path]:
    configured = os.getenv(WORKER_PYTHON_ENV_VAR)
    if configured:
        path = Path(configured)
        return path if path.is_file() else None
    for candidate in (DEFAULT_WORKER_PYTHON, DEFAULT_WORKER_PYTHON_POSIX):
        if candidate.is_file():
            return candidate
    return None


def worker_enabled() -> bool:
    return os.getenv(WORKER_ENABLED_ENV_VAR, "0").strip() == "1"


@dataclass
class _Pending:
    """One request the worker has been sent and not yet answered."""

    request_id: Any
    generation: int
    limit: float
    sent_at: float = field(default_factory=time.monotonic)
    #: How long the worker itself may take before it is judged unresponsive.
    #: A caller whose own deadline is nearly spent waits less; that says
    #: nothing about the worker, so the watchdog uses this instead.
    worker_limit: Optional[float] = None
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    abandoned: bool = False

    def settle(
        self, response: Optional[Dict[str, Any]] = None, error: Optional[str] = None
    ) -> None:
        if self.event.is_set():
            return
        self.response, self.error = response, error
        self.event.set()


class ModelWorkerClient:
    """Owns one worker subprocess at a time and correlates its answers by id."""

    def __init__(self, python_executable: Optional[Path] = None):
        self._python = python_executable or worker_python()
        self._process: Optional[subprocess.Popen] = None
        #: Guards the process, the pending map and the in-flight slot.
        self._lock = threading.RLock()
        self._slot_free = threading.Condition(self._lock)
        #: One supervisor: only the holder may start or replace the worker.
        self._restart_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._request_id = 0
        #: Bumped whenever a worker is started. A response carrying another
        #: generation belongs to a retired process and is never matched.
        self._generation = 0
        self._ready = False
        self._models: List[str] = []
        self._pending: Dict[Any, _Pending] = {}
        self._inflight: Optional[_Pending] = None
        self._waiting = 0
        self._abandoned: List[Any] = []
        self._reader: Optional[threading.Thread] = None
        self._watchdog: Optional[threading.Thread] = None
        self._closed = threading.Event()
        #: Counts, for diagnostics. Never content.
        self.stats = {
            "restarts": 0,
            "late_discarded": 0,
            "abandoned": 0,
            "protocol_errors": 0,
            "crashes": 0,
            "starts": 0,
            "detect_requests": 0,
        }
        #: What the worker last said about itself: device, load count, pid.
        self.runtime: Dict[str, Any] = {}

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        """Start a worker if none is running. Concurrent callers get one."""
        with self._restart_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self.is_ready():
            return
        if self._python is None:
            raise LocalModelError("model_worker_python_unavailable", 503)
        self._retire(graceful=False, error=ERROR_WORKER_RESTARTED)
        with self._lock:
            self._generation += 1
            generation = self._generation
            banner = _Pending(0, generation, STARTUP_TIMEOUT_SECONDS)
            self._pending[0] = banner
        environment = dict(os.environ)
        environment[GENERATION_ENV_VAR] = str(generation)
        try:
            process = subprocess.Popen(
                [str(self._python), "-m", "services.model_worker"],
                cwd=str(BACKEND_ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=environment,
            )
        except Exception as exc:
            with self._lock:
                self._pending.pop(0, None)
            raise LocalModelError("model_worker_start_failed", 503) from exc
        with self._lock:
            self._process = process
            self.stats["starts"] += 1
        self._reader = threading.Thread(
            target=self._read_loop,
            args=(process, generation),
            daemon=True,
            name=f"model-worker-reader-{generation}",
        )
        self._reader.start()
        threading.Thread(
            target=self._diagnostics_loop,
            args=(process,),
            daemon=True,
            name=f"model-worker-diagnostics-{generation}",
        ).start()
        self._ensure_watchdog()

        answered = banner.event.wait(STARTUP_TIMEOUT_SECONDS)
        response = banner.response if answered else None
        with self._lock:
            self._pending.pop(0, None)
        if response is None or not response.get("ready"):
            code = (
                (response or {}).get("error_code")
                or banner.error
                or "model_worker_not_ready"
            )
            self._retire(graceful=True, error=ERROR_WORKER_RESTARTED)
            raise LocalModelError(_safe_error_code(code), 503)
        with self._lock:
            self._ready = True
            self._models = list(response.get("models") or [])
            self.runtime = _runtime_from(response)

    def stop(self, graceful: bool = True) -> None:
        """Retire the current worker and stop watching.

        ``graceful`` asks it to shut itself down and is right at process exit,
        unless a request is still inside it: a worker busy with inference is
        not reading its stdin, so the polite request would sit unanswered for
        the whole wait. That case goes straight to signals.
        """
        self._closed.set()
        with self._restart_lock:
            self._retire(graceful=graceful, error="model_worker_stopped")
        watchdog = self._watchdog
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout=2.0)
        self._watchdog = None
        self._closed.clear()

    def _retire(self, graceful: bool, error: str) -> None:
        """Take the current process out of service and fail what it held."""
        with self._lock:
            process = self._process
            generation = self._generation
            self._process = None
            self._ready = False
            busy = self._inflight is not None
            for pending in list(self._pending.values()):
                pending.settle(error=error)
            self._pending.clear()
            self._inflight = None
            self._slot_free.notify_all()
            # Nothing the retired generation says can be matched any more.
            self._generation += 1
        if process is None:
            return
        try:
            if process.poll() is None:
                if graceful and not busy and process.stdin:
                    try:
                        with self._write_lock:
                            process.stdin.write(
                                json.dumps(
                                    {
                                        "v": PROTOCOL_VERSION,
                                        "gen": generation,
                                        "id": -1,
                                        "op": "shutdown",
                                    }
                                )
                                + "\n"
                            )
                            process.stdin.flush()
                        process.wait(timeout=GRACEFUL_SHUTDOWN_SECONDS)
                    except Exception:
                        pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=TERMINATE_GRACE_SECONDS)
                    except subprocess.TimeoutExpired:
                        # A worker that ignores terminate must not survive to
                        # answer a later request.
                        process.kill()
                        try:
                            process.wait(timeout=TERMINATE_GRACE_SECONDS)
                        except subprocess.TimeoutExpired:
                            pass
        finally:
            for stream in (process.stdin, process.stdout):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass

    def _replace(self, generation: int, reason: str) -> None:
        """The supervisor path: replace the worker of ``generation`` once."""
        with self._restart_lock:
            with self._lock:
                if generation != self._generation or self._process is None:
                    return  # already replaced, or stopped
                self.stats["restarts"] += 1
            self._retire(graceful=False, error=reason)
            if self._closed.is_set():
                return
            try:
                self._start_locked()
            except LocalModelError:
                # Readiness stays false; the next request tries again and
                # fails closed if the worker still cannot start.
                pass

    def is_ready(self) -> bool:
        with self._lock:
            return (
                self._ready
                and self._process is not None
                and self._process.poll() is None
            )

    # -- reading ----------------------------------------------------------

    def _read_loop(self, process, generation: int) -> None:
        """The only reader of this process's stdout."""
        stream = process.stdout
        try:
            for line in iter(stream.readline, ""):
                if not self._dispatch(line, generation):
                    return
        except Exception:
            pass
        # End of stream: the process exited, or was retired.
        with self._lock:
            current = generation == self._generation and self._process is process
            if current:
                self.stats["crashes"] += 1
                self._ready = False
                for pending in list(self._pending.values()):
                    pending.settle(error=ERROR_WORKER_CRASHED)
                self._pending.clear()
                self._inflight = None
                self._slot_free.notify_all()
        if current and not self._closed.is_set():
            # Restore readiness now rather than on a request that readiness
            # itself would turn away.
            threading.Thread(
                target=self._replace,
                args=(generation, ERROR_WORKER_CRASHED),
                daemon=True,
            ).start()

    def _dispatch(self, line: str, generation: int) -> bool:
        """Route one line to its caller. False ends this reader."""
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError
        except ValueError:
            return self._protocol_failure(generation, "model_response_malformed")
        with self._lock:
            if generation != self._generation:
                return False  # a retired worker's words; its reader stops
            if message.get("v") != PROTOCOL_VERSION:
                return self._protocol_failure(
                    generation, "model_protocol_version_mismatch"
                )
            if message.get("gen") != generation:
                return self._protocol_failure(
                    generation, "model_worker_generation_mismatch"
                )
            if "id" not in message or message.get("id") is None:
                return self._protocol_failure(generation, "model_response_missing_id")
            request_id = message["id"]
            if type(request_id) not in (int, str):
                return self._protocol_failure(generation, "model_response_invalid_id")
            pending = self._pending.pop(request_id, None)
            if pending is None:
                if request_id in self._abandoned:
                    # The late answer to a request whose caller gave up.
                    self._abandoned.remove(request_id)
                    self.stats["late_discarded"] += 1
                    self._free_slot(request_id)
                    return True
                return self._protocol_failure(generation, "model_response_id_mismatch")
            pending.settle(response=message)
            self._free_slot(request_id)
            return True

    def _free_slot(self, request_id: Any) -> None:
        if self._inflight is not None and self._inflight.request_id == request_id:
            self._inflight = None
            self._slot_free.notify_all()

    def _protocol_failure(self, generation: int, code: str) -> bool:
        """The stream can no longer be trusted: fail what it held, replace it."""
        with self._lock:
            if generation != self._generation:
                return False
            # Refuse new work immediately, even before the supervisor thread
            # gets scheduled to retire this generation.
            self._ready = False
            self.stats["protocol_errors"] += 1
            for pending in list(self._pending.values()):
                pending.settle(error=code)
        threading.Thread(
            target=self._replace, args=(generation, code), daemon=True
        ).start()
        return False

    # -- watching -----------------------------------------------------------

    def _ensure_watchdog(self) -> None:
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        self._watchdog = threading.Thread(
            target=self._watch, daemon=True, name="model-worker-watchdog"
        )
        self._watchdog.start()

    def _watch(self) -> None:
        """Replace a worker whose in-flight request is past its limit and grace."""
        while not self._closed.wait(WATCHDOG_INTERVAL_SECONDS):
            with self._lock:
                inflight = self._inflight
            if inflight is None:
                continue
            try:
                grace = grace_seconds()
            except LocalModelError:
                grace = DEFAULT_GRACE_SECONDS
            limit = (
                inflight.worker_limit
                if inflight.worker_limit is not None
                else inflight.limit
            )
            if time.monotonic() - inflight.sent_at > limit + grace:
                self._replace(inflight.generation, ERROR_WORKER_UNRESPONSIVE)

    # -- requests -------------------------------------------------------

    def ping(self) -> Dict[str, Any]:
        response = self._request({"op": "ping"}, timeout=30.0)
        if response.get("ok"):
            with self._lock:
                self.runtime = _runtime_from(response)
        return response

    def detect(self, detector: str, text: str) -> List[PhiEntity]:
        with self._lock:
            # A count, so an evaluator can prove a repeated upload ran nothing.
            self.stats["detect_requests"] += 1
        # Parent-side enforcement. The worker checks its own budget only
        # between batches, so it cannot interrupt one blocked inference; this
        # wait, then the watchdog's grace, are the authorities that can.
        rpc_limit = request_timeout_seconds()
        budget = budget_for("model_worker", rpc_limit)
        left = request_remaining()
        if left is not None and left < MIN_CALL_SECONDS:
            # The file's deadline is all but spent: this is a deadline, not a
            # worker fault. Nothing is sent, so no healthy worker is replaced
            # for a request its caller could never have waited for.
            with self._lock:
                self.stats["deadline_refusals"] = (
                    self.stats.get("deadline_refusals", 0) + 1
                )
            raise LocalModelError(ERROR_REQUEST_DEADLINE, 503)
        try:
            response = self._request(
                {"op": "detect", "detector": detector, "text": text},
                timeout=rpc_limit if budget is None else budget,
                worker_limit=rpc_limit,
            )
        except LocalModelError as exc:
            left = request_remaining()
            if (
                exc.error_code == ERROR_WORKER_TIMEOUT
                and left is not None
                and left < MIN_CALL_SECONDS
            ):
                # The file's deadline ran out, not the worker's limit.
                raise LocalModelError(ERROR_REQUEST_DEADLINE, 503) from exc
            raise
        if response.get("ok") is not True:
            raise LocalModelError(_safe_error_code(response.get("error_code")), 503)
        items = response.get("entities")
        try:
            if not isinstance(items, list):
                raise ValueError
            entities = []
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError
                start, end = item["start"], item["end"]
                if (
                    type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end <= len(text)
                ):
                    raise ValueError
                if not isinstance(item["entity_type"], str) or not item["entity_type"]:
                    raise ValueError
                if not isinstance(item["source"], str) or not item["source"]:
                    raise ValueError
                score = item.get("score")
                if score is not None and (
                    type(score) not in (int, float)
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                ):
                    raise ValueError
                entities.append(
                    PhiEntity(
                        entity_type=item["entity_type"],
                        start=start,
                        end=end,
                        source=item["source"],
                        score=score,
                        original_label=item.get("original_label"),
                    )
                )
        except (KeyError, TypeError, ValueError):
            with self._lock:
                generation = self._generation
            self._protocol_failure(generation, "model_output_malformed")
            raise LocalModelError("model_output_malformed", 503) from None
        memory = response.get("memory")
        if isinstance(memory, dict) and memory:
            with self._lock:
                device = dict(self.runtime.get("device") or {})
                device.update(
                    {
                        key: value
                        for key, value in memory.items()
                        if key.endswith("_vram_bytes")
                    }
                )
                self.runtime["device"] = device
        if response.get("degraded"):
            with self._lock:
                device = dict(self.runtime.get("device") or {})
                device.update(
                    {"selected": "cpu", "acceleration_active": False, "degraded": True}
                )
                self.runtime["device"] = device
        return entities

    def _request(
        self,
        payload: Dict[str, Any],
        timeout: float,
        worker_limit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send one request and wait for its own answer, never another's."""
        deadline = time.monotonic() + max(0.0, timeout)
        if not self.is_ready():
            self.start()
        with self._lock:
            if self._waiting >= MAX_PENDING_REQUESTS:
                raise LocalModelError(ERROR_WORKER_BUSY, 503)
            self._waiting += 1
            try:
                # MAX_CONCURRENT_REQUESTS == 1: wait for the worker to be free.
                while self._inflight is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self._slot_free.wait(remaining):
                        raise LocalModelError(ERROR_WORKER_TIMEOUT, 503)
                process = self._process
                if process is None or process.stdin is None or not self._ready:
                    raise LocalModelError("model_worker_unavailable", 503)
                self._request_id += 1
                request_id = self._request_id
                generation = self._generation
                pending = _Pending(
                    request_id, generation, max(0.0, timeout), worker_limit=worker_limit
                )
                self._pending[request_id] = pending
                self._inflight = pending
            finally:
                self._waiting -= 1
        message = {
            "v": PROTOCOL_VERSION,
            "gen": generation,
            "id": request_id,
            **payload,
        }
        try:
            with self._write_lock:
                process.stdin.write(json.dumps(message) + "\n")
                process.stdin.flush()
        except Exception as exc:
            with self._lock:
                self._pending.pop(request_id, None)
                self._free_slot(request_id)
            self._replace(generation, "model_worker_write_failed")
            raise LocalModelError("model_worker_write_failed", 503) from exc

        answered = pending.event.wait(max(0.0, deadline - time.monotonic()))
        if not answered:
            with self._lock:
                if not pending.event.is_set():
                    # Abandoned, not killed: the worker is busy, not broken.
                    # Its answer is dropped by id when it comes; the slot
                    # stays taken until then, and the watchdog replaces the
                    # worker if it never comes.
                    pending.abandoned = True
                    self._pending.pop(request_id, None)
                    self._abandoned.append(request_id)
                    del self._abandoned[:-MAX_ABANDONED_REMEMBERED]
                    self.stats["abandoned"] += 1
                    raise LocalModelError(ERROR_WORKER_TIMEOUT, 503)
        if pending.error is not None:
            raise LocalModelError(pending.error, 503)
        response = pending.response or {}
        if response.get("gen") != generation or generation != pending.generation:
            raise LocalModelError("model_worker_generation_mismatch", 503)
        return response

    def _diagnostics_loop(self, process) -> None:
        """Count the worker's error classes from stderr; keep nothing else.

        Only an exception or warning class name at the start of a line is
        recorded, as a count: never a message, path, value or key.
        """
        stream = process.stderr
        if stream is None:
            return
        try:
            for line in stream:
                match = _DIAGNOSTIC_CLASS.match(line)
                if not match:
                    continue
                name = match.group(1)[:64]
                with self._lock:
                    events = self.stats.setdefault("worker_diagnostics", {})
                    if name in events or len(events) < MAX_DIAGNOSTIC_CLASSES:
                        events[name] = events.get(name, 0) + 1
        except Exception:
            return

    def status(self) -> Dict[str, Any]:
        """Counts and what the worker said about itself. Never content."""
        with self._lock:
            return {
                "generation": self._generation,
                "ready": self.is_ready(),
                "busy": self._inflight is not None,
                **self.stats,
                "runtime": dict(self.runtime),
            }


def _safe_error_code(value: Any) -> str:
    if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
        return value
    return "model_worker_error"


def _runtime_from(message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: message.get(key)
        for key in ("pid", "load_count", "warmup_ms", "model_batch_size", "device")
    }


#: An exception or warning class at the start of a stderr line - the name only.
_DIAGNOSTIC_CLASS = re.compile(
    r"^\s*(?:[A-Za-z_][\w.]*\.)?([A-Z][A-Za-z0-9_]*(?:Error|Exception|"
    r"Warning|Exit|Interrupt))\b"
)
MAX_DIAGNOSTIC_CLASSES = 32


_CLIENT: Optional[ModelWorkerClient] = None
_CLIENT_LOCK = threading.Lock()


def get_client() -> ModelWorkerClient:
    """One worker per API process, started lazily on first use."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = ModelWorkerClient()
        return _CLIENT


def current_client() -> Optional[ModelWorkerClient]:
    """The client if one exists, without creating one."""
    return _CLIENT


def shutdown_client() -> None:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            _CLIENT.stop()
            _CLIENT = None


class RemoteModelDetector:
    """Detector facade that runs inference in the worker process."""

    def __init__(self, detector_name: str, source: str):
        self._detector_name = detector_name
        self.source = source

    def detect(self, text: str) -> List[PhiEntity]:
        if not text or not text.strip():
            return []
        return get_client().detect(self._detector_name, text)
