"""Measure what a real upload costs through the actual ingest API.

The evaluation harness reports a per-document mean, but it runs three
detector passes in a loop with no HTTP layer, no routing and no release
decision, so it answers a different question. This drives the real
``/api/v1/ingest`` route and reports where the time goes.

    python benchmarks/latency_benchmark.py --help
    python benchmarks/latency_benchmark.py --requests 20
    python benchmarks/latency_benchmark.py --scenario concurrency

Results are written under ``benchmarks/results/`` which is git-ignored.
Nothing here touches a network service: the pinned models are loaded from the
local cache in offline mode, or the mocked detectors are used when the model
mode says so.
"""

from __future__ import annotations

import argparse
import ctypes
import io
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

RESULTS_DIR = BACKEND_ROOT / "benchmarks" / "results"

#: Anything slower than this is recorded as a timeout rather than a latency.
DEFAULT_TIMEOUT_SECONDS = 120.0


# -- outcome classification -------------------------------------------------

OUTCOME_SUCCESS = "success"
OUTCOME_REJECTED = "rejected"
OUTCOME_MANUAL_REVIEW = "manual_review"
OUTCOME_FAILED = "failed"
OUTCOME_TIMEOUT = "timeout"
#: Completed correctly, just slower than the budget. Distinct from a timeout,
#: which is reserved for a request that was actually aborted.
OUTCOME_OVER_BUDGET = "over_budget"

OUTCOMES = (
    OUTCOME_SUCCESS,
    OUTCOME_REJECTED,
    OUTCOME_MANUAL_REVIEW,
    OUTCOME_FAILED,
    OUTCOME_TIMEOUT,
    OUTCOME_OVER_BUDGET,
)


def classify(status_code: int, body: Optional[Dict[str, Any]]) -> str:
    """Bucket a response. Counted separately so a fast block is not a fast pass."""
    if status_code == 200:
        decision = (body or {}).get("release_decision") or {}
        if decision.get("releasable") is True:
            return OUTCOME_SUCCESS
        return OUTCOME_MANUAL_REVIEW
    if status_code == 422:
        return OUTCOME_MANUAL_REVIEW
    if 400 <= status_code < 500:
        return OUTCOME_REJECTED
    return OUTCOME_FAILED


# -- statistics -------------------------------------------------------------


def percentile(values: List[float], fraction: float) -> float:
    """Nearest-rank percentile. Explicit so the numbers are reproducible."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(fraction * len(ordered))))
    return ordered[rank - 1]


def summarize(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {
            k: 0.0 for k in ("count", "mean", "min", "p50", "p90", "p95", "p99", "max")
        }
    return {
        "count": len(latencies_ms),
        "mean": round(statistics.fmean(latencies_ms), 2),
        "min": round(min(latencies_ms), 2),
        "p50": round(percentile(latencies_ms, 0.50), 2),
        "p90": round(percentile(latencies_ms, 0.90), 2),
        "p95": round(percentile(latencies_ms, 0.95), 2),
        "p99": round(percentile(latencies_ms, 0.99), 2),
        "max": round(max(latencies_ms), 2),
    }


def throughput_per_minute(latencies_ms: List[float], wall_seconds: float) -> float:
    if wall_seconds <= 0:
        return 0.0
    return round(len(latencies_ms) / wall_seconds * 60.0, 2)


# -- environment ------------------------------------------------------------


def peak_memory_bytes() -> Optional[int]:
    """Peak working set of this process, or None if it cannot be read.

    Uses the Win32 API directly rather than adding a dependency for one
    number. Returns None elsewhere instead of guessing.
    """
    if not sys.platform.startswith("win"):
        try:
            import resource  # noqa: PLC0415

            # ru_maxrss is kilobytes on Linux, bytes on macOS.
            raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return raw * 1024 if sys.platform.startswith("linux") else raw
        except Exception:
            return None

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        probe = kernel32.K32GetProcessMemoryInfo
        probe.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_Counters),
            ctypes.c_uint32,
        ]
        probe.restype = ctypes.c_int
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        handle = kernel32.GetCurrentProcess()
        if not probe(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    except Exception:
        return None


def execution_mode() -> Dict[str, Any]:
    """CPU or GPU, reported from what torch actually says, not assumed."""
    mode: Dict[str, Any] = {"device": "cpu", "torch": None, "cuda_available": False}
    try:
        import torch  # noqa: PLC0415

        mode["torch"] = torch.__version__
        mode["cuda_available"] = bool(torch.cuda.is_available())
        mode["device"] = "cuda" if mode["cuda_available"] else "cpu"
    except Exception:
        # torch lives in the model virtualenv; when detection runs in the
        # worker this process legitimately has no torch. Say so rather than
        # claiming CPU on no evidence.
        mode["torch"] = "not importable in this process"
    return mode


# -- the harness ------------------------------------------------------------


class ApiHarness:
    """Drives the real FastAPI ingest route in-process."""

    def __init__(self) -> None:
        self._client = None
        self.startup_seconds = 0.0
        self.import_seconds = 0.0
        self.readiness_seconds = 0.0
        self.readiness = None

    def start(self) -> None:
        from fastapi.testclient import TestClient  # noqa: PLC0415

        started = time.perf_counter()
        import main  # noqa: PLC0415

        self._main = main
        self.import_seconds = time.perf_counter() - started

        started = time.perf_counter()
        started_ready = time.perf_counter()
        self._client = TestClient(main.app)
        self._client.__enter__()  # runs the lifespan, so prewarming happens here
        self.readiness_seconds = time.perf_counter() - started_ready
        self.startup_seconds = time.perf_counter() - started
        try:
            from services.model_lifecycle import state as _state

            self.readiness = _state().to_public_dict()
        except Exception:
            self.readiness = None

    def post(self, payload: bytes, timeout: float) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            response = self._client.post(
                "/api/v1/ingest",
                files={"file": ("note.txt", io.BytesIO(payload), "text/plain")},
                data={"profile": "strict"},
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            over_budget = elapsed_ms > timeout * 1000
            try:
                body = response.json()
            except Exception:
                body = None
            return {
                "outcome": (
                    OUTCOME_OVER_BUDGET
                    if over_budget
                    else classify(response.status_code, body)
                ),
                "latency_ms": elapsed_ms,
                "status_code": response.status_code,
                "over_budget": over_budget,
            }
        except Exception as exc:
            return {
                "outcome": OUTCOME_FAILED,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "error_type": type(exc).__name__,
            }

    def stage_totals(self) -> Dict[str, Dict[str, float]]:
        """Mean duration and call count per stage across retained requests."""
        records = getattr(self._main, "latest_request_timings", lambda: [])()
        totals: Dict[str, List[float]] = {}
        for record in records:
            for stage in record.stages:
                totals.setdefault(stage.name, []).append(stage.duration_ms)
        return {
            name: {
                "calls_total": len(values),
                "calls_per_request": round(len(values) / max(1, len(records)), 2),
                "mean_ms": round(statistics.fmean(values), 3),
            }
            for name, values in sorted(totals.items())
        }


def _run_sequential(
    harness: ApiHarness,
    payload: bytes,
    count: int,
    timeout: float,
) -> Dict[str, Any]:
    results, started = [], time.perf_counter()
    for _ in range(count):
        results.append(harness.post(payload, timeout))
    wall = time.perf_counter() - started
    return _assemble(results, wall)


def _run_concurrent(
    harness: ApiHarness,
    payload: bytes,
    count: int,
    workers: int,
    timeout: float,
) -> Dict[str, Any]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(harness.post, payload, timeout) for _ in range(count)]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - started
    assembled = _assemble(results, wall)
    # Serial time divided by observed wall time shows how much overlap the
    # single-slot worker actually allowed. Queueing is inferred, not timed
    # inside the worker, and is labelled as such.
    serial_ms = sum(r["latency_ms"] for r in results)
    assembled["inferred_queueing_ms"] = round(
        max(0.0, (wall * 1000) - (serial_ms / max(1, workers))), 2
    )
    assembled["concurrency"] = workers
    return assembled


def _assemble(results: List[Dict[str, Any]], wall_seconds: float) -> Dict[str, Any]:
    counts = {outcome: 0 for outcome in OUTCOMES}
    for result in results:
        counts[result["outcome"]] = counts.get(result["outcome"], 0) + 1
    # Percentiles describe completed requests; a timeout has no meaningful
    # latency and would otherwise flatter or distort the tail.
    measured = [r["latency_ms"] for r in results if r["outcome"] != OUTCOME_TIMEOUT]
    over_budget = sum(1 for r in results if r.get("over_budget"))
    return {
        "requests": len(results),
        "latency_ms": summarize(measured),
        "outcomes": counts,
        "over_budget": over_budget,
        "wall_seconds": round(wall_seconds, 3),
        "throughput_per_minute": throughput_per_minute(measured, wall_seconds),
    }


# -- scenarios --------------------------------------------------------------


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from benchmarks.fixtures import SEED, build_document, size_targets  # noqa: PLC0415

    report: Dict[str, Any] = {
        "schema_version": 1,
        "seed": SEED,
        "execution": execution_mode(),
        "model_mode": os.getenv("PHI_MODEL_MODE", "unset"),
        "worker_enabled": os.getenv("PHI_MODEL_WORKER", "0") == "1",
        "requests_per_scenario": args.requests,
        "warmup_requests": args.warmup,
        "scenarios": {},
    }

    harness = ApiHarness()
    cold_started = time.perf_counter()
    harness.start()
    report["readiness"] = harness.readiness
    report["startup"] = {
        "readiness_seconds": round(harness.readiness_seconds, 3),
        "import_seconds": round(harness.import_seconds, 3),
        "app_construction_seconds": round(harness.startup_seconds, 3),
        "total_seconds": round(time.perf_counter() - cold_started, 3),
    }

    tiny = build_document("mixed_clinical", 1_024)

    # Cold: the first request pays model load and any lazy initialisation.
    first = harness.post(tiny, args.timeout)
    report["scenarios"]["cold_first_request"] = _assemble(
        [first], first["latency_ms"] / 1000
    )
    report["model_load"] = {
        "note": (
            "Model load is not separable from the first request in-process; "
            "it is included in cold_first_request and excluded from all warm "
            "scenarios by the warm-up requests."
        ),
        "cold_first_request_ms": round(first["latency_ms"], 2),
    }

    for _ in range(args.warmup):
        harness.post(tiny, args.timeout)
    report["model_ready_after_warmup"] = True

    if args.scenario in {"all", "sizes"}:
        wanted = set(args.sizes.split(",")) if args.sizes else None
        for name, target in size_targets(include_large=not args.no_large):
            if wanted is not None and name not in wanted:
                continue
            payload = build_document("mixed_clinical", target)
            report["scenarios"][f"warm_{name}_text"] = {
                **_run_sequential(harness, payload, args.requests, args.timeout),
                "document_bytes": len(payload),
            }

    if args.scenario in {"all", "concurrency"}:
        for workers in (1, 2, 4):
            payload = build_document("mixed_clinical", args.payload_size)
            report["scenarios"][f"concurrency_{workers}"] = _run_concurrent(
                harness, payload, args.requests, workers, args.timeout
            )

    if args.scenario in {"all", "content"}:
        from benchmarks.fixtures import CONTENT_KINDS  # noqa: PLC0415

        for kind in CONTENT_KINDS:
            payload = build_document(kind, 4_096)
            report["scenarios"][f"content_{kind}"] = _run_sequential(
                harness, payload, max(3, args.requests // 4), args.timeout
            )

    report["stages"] = harness.stage_totals()
    peak = peak_memory_bytes()
    report["peak_memory_bytes"] = peak
    report["peak_memory_mb"] = round(peak / 1_048_576, 1) if peak else None
    return report


def print_summary(report: Dict[str, Any]) -> None:
    print()
    print(
        f"execution      : {report['execution']['device']} "
        f"(torch {report['execution']['torch']})"
    )
    print(
        f"model mode     : {report['model_mode']}  "
        f"worker={report['worker_enabled']}"
    )
    peak = report.get("peak_memory_mb")
    print(f"peak memory    : {peak} MB" if peak else "peak memory    : unavailable")
    print(f"startup        : {report['startup']['total_seconds']} s")
    print()
    header = f"{'scenario':<28}{'reqs':>5}{'p50':>9}{'p90':>9}{'p95':>9}{'p99':>9}{'max':>9}{'thr/min':>9}{'fail':>6}"
    print(header)
    print("-" * len(header))
    for name, data in report["scenarios"].items():
        stats = data["latency_ms"]
        failures = data["outcomes"].get(OUTCOME_FAILED, 0) + data["outcomes"].get(
            OUTCOME_TIMEOUT, 0
        )
        print(
            f"{name:<28}{data['requests']:>5}{stats['p50']:>9.1f}"
            f"{stats['p90']:>9.1f}{stats['p95']:>9.1f}{stats['p99']:>9.1f}"
            f"{stats['max']:>9.1f}{data['throughput_per_minute']:>9.1f}"
            f"{failures:>6}"
        )
    print()
    print("slowest stages (mean ms, calls per request):")
    ranked = sorted(
        report["stages"].items(), key=lambda kv: kv[1]["mean_ms"], reverse=True
    )
    for name, data in ranked[:10]:
        print(f"  {name:<38}{data['mean_ms']:>9.2f}  x{data['calls_per_request']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=20,
        help="measured requests per scenario (default 20)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="warm-up requests excluded from percentiles",
    )
    parser.add_argument(
        "--scenario", default="all", choices=("all", "sizes", "concurrency", "content")
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="per-request timeout in seconds",
    )
    parser.add_argument(
        "--no-large", action="store_true", help="skip the largest size bucket"
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help="comma-separated buckets to measure, e.g. tiny,small",
    )
    parser.add_argument(
        "--payload-size",
        type=int,
        default=1_024,
        help="bytes for the concurrency payload (default 1024)",
    )
    parser.add_argument(
        "--out", default=None, help="JSON output path (default benchmarks/results/)"
    )
    args = parser.parse_args()

    report = run(args)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = (
        Path(args.out)
        if args.out
        else (RESULTS_DIR / f"latency-{time.strftime('%Y%m%d-%H%M%S')}.json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print_summary(report)
    print()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
