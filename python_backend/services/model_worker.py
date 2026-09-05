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

The protocol is newline-delimited JSON, one object per line, request and
response correlated by ``id``:

    {"id": 1, "op": "ping"}
    {"id": 1, "ok": true, "ready": true, "models": ["stanford", "gliner"]}

    {"id": 2, "op": "detect", "detector": "stanford", "text": "..."}
    {"id": 2, "ok": true, "entities": [{"start": 8, "end": 12, ...}]}

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

PROTOCOL_VERSION = 1
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
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
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

    def warm(self) -> None:
        """Load and checksum-verify both models once, before serving."""
        try:
            from services.local_model_detectors import (
                SOURCE_GLINER,
                SOURCE_STANFORD,
                GlinerPiiDetector,
                StanfordClinicalDetector,
                calibrated_config_for,
                load_gliner_model,
                load_stanford_pipeline,
            )

            # Loading verifies the pinned revision and weight checksum, and
            # verifies GLiNER's tokenizer backbone too.
            load_stanford_pipeline()
            load_gliner_model()
            self._detectors = {
                "stanford": StanfordClinicalDetector(
                    calibrated_config_for(SOURCE_STANFORD)
                ),
                "gliner": GlinerPiiDetector(calibrated_config_for(SOURCE_GLINER)),
            }
            self._ready = True
        except Exception as exc:
            # Carries a code, never the text or a stack containing it.
            self._load_error = getattr(exc, "error_code", type(exc).__name__)
            self._ready = False

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request_id = request.get("id")
        op = request.get("op")

        if op == "ping":
            return {
                "id": request_id,
                "ok": True,
                "ready": self._ready,
                "protocol": PROTOCOL_VERSION,
                "models": sorted(self._detectors),
                "error_code": self._load_error,
            }
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

        started = time.perf_counter()
        try:
            entities = detector.detect(text)
        except Exception as exc:
            return {
                "id": request_id,
                "ok": False,
                "error_code": getattr(exc, "error_code", "inference_failed"),
            }
        return {
            "id": request_id,
            "ok": True,
            "entities": _entities_to_json(list(entities)),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def main() -> int:
    _configure_offline()
    worker = _Worker()
    worker.warm()
    # Announce readiness before the first request so the parent can wait on a
    # definite signal instead of sleeping.
    _respond(
        {
            "id": 0,
            "ok": worker._ready,
            "ready": worker._ready,
            "protocol": PROTOCOL_VERSION,
            "models": sorted(worker._detectors),
            "error_code": worker._load_error,
        }
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
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
