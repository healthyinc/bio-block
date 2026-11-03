"""Persistent local model worker.

FastAPI runs in the main environment, which holds ChromaDB and cannot take the
`transformers`/`tokenizers`/`huggingface_hub` pins the models require without
downgrading them underneath it. Before Phase 10 the consequence was concrete
and unnoticed: with ``PHI_MODEL_MODE=offline`` the API blocked every request
with ``model_files_unavailable``. The calibration was real, but nothing in
production could reach the models it calibrated.

This process is the bridge. It runs under the model virtualenv, loads each
pinned model once, and answers detection requests over its own stdin/stdout.

    <model-venv-python> -m services.model_worker

The protocol is newline-delimited JSON, one object per line. Every message
carries the protocol version ``v``, the worker ``gen``eration the parent gave
this process, and the request ``id``; the parent matches responses to callers
by id and discards anything for another generation:

    {"v": 2, "gen": 3, "id": "a1", "op": "ping"}
    {"v": 2, "gen": 3, "id": "a1", "ok": true, "ready": true, ...}

    {"v": 2, "gen": 3, "id": "a2", "op": "detect", "detector": "stanford", "text": "..."}
    {"v": 2, "gen": 3, "id": "a2", "ok": true, "entities": [...]}

Requests are executed one at a time, in arrival order: the models are never
used concurrently.

stdin/stdout rather than a socket on purpose: there is no port to bind, no
listener reachable from off the machine, and no authentication problem to get
wrong. The child is only reachable by its parent.

Text arrives here and never leaves. Nothing in this module writes request text
to stdout, stderr or any log: responses carry offsets, categories and scores,
and failures carry an error code.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

PROTOCOL_VERSION = 2
GENERATION_ENV_VAR = "BIOBLOCK_WORKER_GENERATION"
_GENERATION = int(os.getenv(GENERATION_ENV_VAR, "0") or 0)
#: Invented, identifier-free text that forces the first forward pass of each
#: model on the selected device before the worker says it is ready.
WARMUP_TEXT = "The clinic reviewed the routine observation record and plan."
#: Refuse anything larger rather than letting one request exhaust memory.
MAX_REQUEST_BYTES = 512 * 1024


def _configure_offline() -> None:
    cache = BACKEND_ROOT / ".model-cache"
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "hub"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["PHI_MODEL_MODE"] = "offline"


def _respond(payload: Dict[str, Any]) -> None:
    message = {"v": PROTOCOL_VERSION, "gen": _GENERATION, **payload}
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _entities_to_json(entities: List[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "start": int(e.start),
            "end": int(e.end),
            "entity_type": str(e.entity_type),
            "source": str(e.source),
            "score": None if e.score is None else float(e.score),
            "original_label": e.original_label,
        }
        for e in entities
    ]


class _Worker:
    def __init__(self) -> None:
        self._detectors: Dict[str, Any] = {}
        self._ready = False
        self._load_error: str | None = None
        self.generation = _GENERATION
        self.load_count = 0
        self.warmup_ms = 0.0
        self.device: Any = None

    def warm(self) -> None:
        """Select the device, load and verify both models once, warm them up."""
        from services import model_device

        try:
            requested = model_device.requested_device()
            self.device = model_device.probe(requested)
        except Exception as exc:
            self._fail(exc)
            return
        if self.device.selected == model_device.DEVICE_CUDA and not os.getenv(
            "PHI_MODEL_BATCH_SIZE"
        ):
            # Only this process's view; an explicit setting always wins.
            os.environ["PHI_MODEL_BATCH_SIZE"] = str(
                model_device.CUDA_DEFAULT_BATCH_SIZE
            )
        try:
            self._load(self.device.selected)
        except Exception as exc:
            if (
                self.device.selected == model_device.DEVICE_CUDA
                and requested == model_device.DEVICE_AUTO
            ):
                # auto: CUDA did not hold up before any job was accepted.
                # Drop it entirely and load on the CPU, saying why.
                reason = getattr(exc, "error_code", model_device.ERROR_CUDA_PLACEMENT)
                self._reset_models()
                self.device.selected, self.device.reason = (
                    model_device.DEVICE_CPU,
                    reason,
                )
                try:
                    self._load(model_device.DEVICE_CPU)
                except Exception as retry:
                    self._fail(retry)
                    return
            else:
                self._fail(exc)
                return
        self._ready = True

    def _fail(self, exc: BaseException) -> None:
        # Carries a code, never the text or a stack containing it.
        self._load_error = getattr(exc, "error_code", type(exc).__name__)
        self._ready = False

    def _reset_models(self) -> None:
        from services import local_model_detectors as models
        from services import model_device

        self._detectors = {}
        models.load_stanford_pipeline.cache_clear()
        models.load_gliner_model.cache_clear()
        model_device.release_cuda_cache()

    def _load(self, device: str) -> None:
        from services import local_model_detectors as models
        from services import model_device

        models.set_model_device(device)
        # Loading verifies the pinned revision and weight checksum, and
        # verifies GLiNER's tokenizer backbone too.
        stanford = models.load_stanford_pipeline()
        gliner = models.load_gliner_model()
        self.load_count += 1
        self.device.model_devices = model_device.verify_placement(
            {"stanford": stanford.model, "gliner": gliner}, device
        )
        self._detectors = {
            "stanford": models.StanfordClinicalDetector(
                models.calibrated_config_for(models.SOURCE_STANFORD)
            ),
            "gliner": models.GlinerPiiDetector(
                models.calibrated_config_for(models.SOURCE_GLINER)
            ),
        }
        started = time.perf_counter()
        for detector in self._detectors.values():
            detector.detect(WARMUP_TEXT)
        if device == model_device.DEVICE_CUDA:
            import torch

            torch.cuda.synchronize()
        self.warmup_ms = round((time.perf_counter() - started) * 1000, 2)
        self.device.acceleration_active = device == model_device.DEVICE_CUDA
        model_device.refresh_memory(self.device)

    def _fall_back_to_cpu(self) -> bool:
        """After a CUDA failure mid-service, reload on the CPU if configured."""
        from services import model_device

        if (
            not model_device.cpu_fallback_allowed()
            or self.device is None
            or self.device.selected != model_device.DEVICE_CUDA
        ):
            return False
        self._reset_models()
        self.device.selected = model_device.DEVICE_CPU
        self.device.acceleration_active = False
        self.device.degraded = True
        self.device.reason = model_device.ERROR_CUDA_OOM
        self.device.cpu_fallbacks += 1
        try:
            self._load(model_device.DEVICE_CPU)
        except Exception as exc:
            self._fail(exc)
            return False
        return True

    def status(self) -> Dict[str, Any]:
        from services import model_device

        if self.device is not None:
            model_device.refresh_memory(self.device)
        return {
            "ready": self._ready,
            "protocol": PROTOCOL_VERSION,
            "models": sorted(self._detectors),
            "error_code": self._load_error,
            "pid": os.getpid(),
            "load_count": self.load_count,
            "warmup_ms": self.warmup_ms,
            "model_batch_size": _batch_size(),
            "device": None if self.device is None else self.device.to_dict(),
        }

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request_id = request.get("id")
        op = request.get("op")
        if request.get("v") != PROTOCOL_VERSION:
            return {
                "id": request_id,
                "ok": False,
                "error_code": "protocol_version_mismatch",
            }
        if request.get("gen") != self.generation:
            return {
                "id": request_id,
                "ok": False,
                "error_code": "worker_generation_mismatch",
            }

        if op == "ping":
            return {"id": request_id, "ok": True, **self.status()}
        if op == "shutdown":
            return {"id": request_id, "ok": True, "shutdown": True}
        if op != "detect":
            return {"id": request_id, "ok": False, "error_code": "unknown_operation"}

        if not self._ready:
            return {
                "id": request_id,
                "ok": False,
                "error_code": self._load_error or "model_not_ready",
            }

        detector = self._detectors.get(str(request.get("detector")))
        if detector is None:
            return {"id": request_id, "ok": False, "error_code": "unknown_detector"}

        text = request.get("text")
        if not isinstance(text, str):
            return {"id": request_id, "ok": False, "error_code": "invalid_request"}
        if len(text.encode("utf-8")) > MAX_REQUEST_BYTES:
            return {"id": request_id, "ok": False, "error_code": "request_too_large"}

        from services import model_device

        started = time.perf_counter()
        degraded = False
        try:
            entities = detector.detect(text)
        except Exception as exc:
            code = getattr(exc, "error_code", "inference_failed")
            if code == model_device.ERROR_CUDA_OOM and self.device is not None:
                self.device.oom_events += 1
            if code != model_device.ERROR_CUDA_OOM or not self._fall_back_to_cpu():
                return {"id": request_id, "ok": False, "error_code": code}
            # The whole request again, on the CPU: nothing from the failed
            # CUDA attempt is kept, so results are never mixed.
            detector = self._detectors[str(request.get("detector"))]
            degraded = True
            try:
                entities = detector.detect(text)
            except Exception as retry:
                return {
                    "id": request_id,
                    "ok": False,
                    "error_code": getattr(retry, "error_code", "inference_failed"),
                }
        return {
            "id": request_id,
            "ok": True,
            "entities": _entities_to_json(list(entities)),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "device": self.device.selected if self.device is not None else "cpu",
            "degraded": degraded,
            "memory": self._memory(),
        }

    def _memory(self) -> Dict[str, Any]:
        """Current VRAM counters, so the parent's report never goes stale."""
        from services import model_device

        if self.device is None:
            return {}
        model_device.refresh_memory(self.device)
        return {
            key: getattr(self.device, key)
            for key in (
                "free_vram_bytes",
                "allocated_vram_bytes",
                "reserved_vram_bytes",
                "peak_allocated_vram_bytes",
            )
        }


def _batch_size() -> int | None:
    try:
        from services.local_model_detectors import detector_config_from_env

        return detector_config_from_env().batch_size
    except Exception:
        return None


def main() -> int:
    _configure_offline()
    worker = _Worker()
    worker.warm()
    # Announce readiness before the first request so the parent can wait on a
    # definite signal instead of sleeping.
    _respond({"id": 0, "ok": worker._ready, **worker.status()})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError
        except Exception:
            _respond({"id": None, "ok": False, "error_code": "malformed_request"})
            continue
        try:
            response = worker.handle(request)
        except Exception:
            # Never let a stack trace carrying request text reach a stream.
            traceback.clear_frames(sys.exc_info()[2])
            response = {
                "id": request.get("id"),
                "ok": False,
                "error_code": "worker_internal_error",
            }
        _respond(response)
        if response.get("shutdown"):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
