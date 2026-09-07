"""Client for the out-of-process model worker.

Runs in the main FastAPI environment and talks to a child process started
under the model virtualenv. The main environment therefore never imports
torch, transformers or gliner, and ChromaDB's pins are left alone.

Every failure path blocks. A worker that will not start, will not become
ready, times out, dies mid-request or answers with an error yields
``LocalModelError``, which the text pipeline already turns into a blocked
artifact. There is no path here that returns "no PHI found" because the
worker was unavailable.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.local_model_detectors import LocalModelError
from services.privacy_contracts import PhiEntity

BACKEND_ROOT = Path(__file__).resolve().parent.parent

#: Interpreter that owns the model stack. Overridable for deployment.
WORKER_PYTHON_ENV_VAR = "PHI_MODEL_WORKER_PYTHON"
DEFAULT_WORKER_PYTHON = BACKEND_ROOT / ".venv-models" / "Scripts" / "python.exe"
DEFAULT_WORKER_PYTHON_POSIX = BACKEND_ROOT / ".venv-models" / "bin" / "python"

#: Enables the out-of-process path. Absent, the in-process detectors are used,
#: which is what the mocked test suite relies on.
WORKER_ENABLED_ENV_VAR = "PHI_MODEL_WORKER"

STARTUP_TIMEOUT_SECONDS = 300.0
REQUEST_TIMEOUT_SECONDS = 120.0
#: One in-flight request at a time. The worker is a single process holding two
#: models on a memory-tight machine; queueing is safer than contention.
MAX_CONCURRENT_REQUESTS = 1


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


class ModelWorkerClient:
    """Owns one worker subprocess and serialises requests to it."""

    def __init__(self, python_executable: Optional[Path] = None):
        self._python = python_executable or worker_python()
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._request_id = 0
        self._ready = False
        self._models: List[str] = []

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if self._python is None:
            raise LocalModelError("model_worker_python_unavailable", 503)

        try:
            self._process = subprocess.Popen(
                [str(self._python), "-m", "services.model_worker"],
                cwd=str(BACKEND_ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except Exception as exc:
            raise LocalModelError("model_worker_start_failed", 503) from exc

        banner = self._read_line(STARTUP_TIMEOUT_SECONDS)
        if banner is None or not banner.get("ready"):
            code = (banner or {}).get("error_code") or "model_worker_not_ready"
            self.stop()
            raise LocalModelError(str(code), 503)
        self._ready = True
        self._models = list(banner.get("models") or [])

    def stop(self) -> None:
        process = self._process
        self._process = None
        self._ready = False
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin:
                try:
                    process.stdin.write(json.dumps({"id": -1, "op": "shutdown"}) + "\n")
                    process.stdin.flush()
                except Exception:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        finally:
            for stream in (process.stdin, process.stdout):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass

    def is_ready(self) -> bool:
        return (
            self._ready
            and self._process is not None
            and self._process.poll() is None
        )

    # -- protocol -------------------------------------------------------

    def _read_line(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Read one JSON line, enforcing a wall-clock deadline."""
        process = self._process
        if process is None or process.stdout is None:
            return None

        result: Dict[str, Any] = {}
        done = threading.Event()

        def reader() -> None:
            try:
                line = process.stdout.readline()
                if line:
                    result["value"] = json.loads(line)
            except Exception:
                pass
            finally:
                done.set()

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        if not done.wait(timeout):
            # The worker is wedged. Kill it rather than leaving a half-read
            # stream that would desynchronise every later request.
            self.stop()
            return None
        return result.get("value")

    def ping(self) -> Dict[str, Any]:
        return self._request({"op": "ping"}, timeout=30.0)

    def detect(self, detector: str, text: str) -> List[PhiEntity]:
        response = self._request(
            {"op": "detect", "detector": detector, "text": text},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if not response.get("ok"):
            raise LocalModelError(
                str(response.get("error_code") or "model_worker_error"), 503
            )
        return [
            PhiEntity(
                entity_type=item["entity_type"],
                start=int(item["start"]),
                end=int(item["end"]),
                source=item["source"],
                score=item.get("score"),
                original_label=item.get("original_label"),
            )
            for item in response.get("entities", [])
        ]

    def _request(self, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        with self._lock:  # MAX_CONCURRENT_REQUESTS == 1
            if not self.is_ready():
                self.start()
            process = self._process
            if process is None or process.stdin is None:
                raise LocalModelError("model_worker_unavailable", 503)

            self._request_id += 1
            message = dict(payload)
            message["id"] = self._request_id
            try:
                process.stdin.write(json.dumps(message) + "\n")
                process.stdin.flush()
            except Exception as exc:
                self.stop()
                raise LocalModelError("model_worker_write_failed", 503) from exc

            response = self._read_line(timeout)
            if response is None:
                raise LocalModelError("model_worker_timeout", 503)
            return response


_CLIENT: Optional[ModelWorkerClient] = None
_CLIENT_LOCK = threading.Lock()


def get_client() -> ModelWorkerClient:
    """One worker per API process, started lazily on first use."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = ModelWorkerClient()
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
