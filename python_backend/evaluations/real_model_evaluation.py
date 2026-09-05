"""Real-model evaluation and threshold calibration.

Run one detector configuration over one corpus partition and write a JSON
report. One configuration per process, because the two pinned models together
do not comfortably fit in memory on a small machine and a swapping process
produces latency numbers that mean nothing.

    py evaluations/real_model_evaluation.py --detector stanford --partition calibration
    py evaluations/real_model_evaluation.py --detector combined --partition test --out report.json

Threshold sweeping is done by *filtering*, not by re-running inference. Both
adapters apply their threshold as a post-hoc score filter, so the models are
run once at threshold 0.0 with every candidate captured, and each candidate
threshold is then evaluated against that single pass. This is exactly
equivalent and avoids re-running a 1.16 GB model per threshold.

Thresholds may only be chosen on the ``calibration`` partition. The ``test``
partition exists to be run once, after the configuration is locked.

Nothing here reports a matched value. Detections carry offsets, categories,
sources and scores; failures carry an error code.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

CACHE_ROOT = BACKEND_ROOT / ".model-cache"


def _configure_environment(offline: bool = True) -> None:
    """Point at the local cache and force offline before anything imports."""
    os.environ["HF_HOME"] = str(CACHE_ROOT)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_ROOT / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "hub")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("BIOBLOCK_STUDY_SALT", "evaluation-salt")


_configure_environment()

from evaluations import detector_runners as runners  # noqa: E402
from evaluations.labelled_corpus import (  # noqa: E402
    CORPUS_VERSION,
    PARTITION_CALIB,
    PARTITIONS,
    partition_documents,
)
from evaluations.metrics import (  # noqa: E402
    Detection,
    aggregate,
    missed_categories,
    percentile,
)

MODEL_DETECTORS = ("stanford", "gliner")
ALL_DETECTORS = ("rules", "spacy", "stanford", "gliner", "combined")

#: Candidate thresholds swept during calibration.
DEFAULT_SWEEP = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def _environment_fingerprint() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "logical_cpus": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
        info["cuda_available"] = False
        info["device"] = "cpu"
    for name in ("transformers", "gliner", "huggingface_hub", "spacy"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except Exception:
            info[name] = None
    return info


def _manifest_fingerprint() -> Dict[str, Any]:
    manifest = json.loads(
        (BACKEND_ROOT / "config" / "model_manifest.json").read_text(encoding="utf-8")
    )
    return {
        name: {
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "license": spec["license"],
            "weight_sha256": spec["weight_sha256"],
        }
        for name, spec in manifest.items()
    }


def _filter_by_threshold(
    detections_by_doc: Dict[str, List[Detection]],
    threshold: float,
) -> Dict[str, List[Detection]]:
    """Apply a candidate threshold post-hoc.

    A detection with no score is rule-based and is never filtered: the
    deterministic patterns do not produce a confidence to threshold on.
    """
    return {
        doc_id: [
            d for d in detections if d.score is None or d.score >= threshold
        ]
        for doc_id, detections in detections_by_doc.items()
    }


def run_detector(detector: str, partition: str) -> Dict[str, Any]:
    """Run one configuration at threshold 0.0, capturing every candidate."""
    documents = partition_documents(partition)

    if detector in MODEL_DETECTORS:
        from services.local_model_detectors import DetectorConfig

        # Capture everything; the sweep filters afterwards.
        config = DetectorConfig(candidate_threshold=0.0, redaction_threshold=0.0)
        runner = (
            runners.run_stanford if detector == "stanford" else runners.run_gliner
        )
        result = runner(documents, config)
    elif detector == "rules":
        result = runners.run_rules(documents)
    elif detector == "spacy":
        result = runners.run_spacy(documents)
    elif detector == "combined":
        # Do NOT force a threshold here. The combined chain must run under
        # whatever calibration is locked, because that is the configuration
        # being measured. Forcing 0.0 silently overrides the locked file via
        # the environment and reports the uncalibrated chain instead.
        runners.clear_threshold_overrides()
        result = runners.run_combined(documents)
    else:
        raise SystemExit(f"unknown detector: {detector}")

    report = aggregate(documents, result.detections_by_doc)
    latencies = result.latencies_ms

    effective = None
    if detector == "combined":
        from services.local_model_detectors import calibrated_config_for

        effective = {
            name: calibrated_config_for(name).candidate_threshold
            for name in ("stanford_deidentifier", "gliner_multi_pii")
        }

    return {
        "detector": detector,
        "partition": partition,
        "corpus_version": CORPUS_VERSION,
        "threshold": 0.0,
        "effective_thresholds": effective,
        "metrics": report,
        "missed_categories": missed_categories(report),
        "performance": {
            "model_load_seconds": round(result.load_seconds, 2),
            "documents": len(documents),
            "latency_ms_mean": round(sum(latencies) / len(latencies), 2)
            if latencies
            else 0.0,
            "latency_ms_p50": percentile(latencies, 0.50),
            "latency_ms_p90": percentile(latencies, 0.90),
            "latency_ms_p99": percentile(latencies, 0.99),
            "latency_ms_max": round(max(latencies), 2) if latencies else 0.0,
            "peak_rss_mib": round(result.peak_rss_bytes / (1024 * 1024), 1),
        },
        "failures": result.failures,
        "unsupported_documents": result.unsupported,
        # Raw detections are kept so the sweep can filter without re-running.
        "_detections": {
            doc_id: [
                {
                    "start": d.start,
                    "end": d.end,
                    "category": d.category,
                    "source": d.source,
                    "score": d.score,
                }
                for d in detections
            ]
            for doc_id, detections in result.detections_by_doc.items()
        },
    }


def sweep_thresholds(
    run: Dict[str, Any],
    thresholds: Sequence[float] = DEFAULT_SWEEP,
) -> Dict[str, Any]:
    """Evaluate candidate thresholds against a single captured inference pass."""
    partition = run["partition"]
    if partition != PARTITION_CALIB:
        raise SystemExit(
            "Thresholds may only be swept on the calibration partition. "
            f"Refusing to sweep on '{partition}'."
        )
    documents = partition_documents(partition)
    detections_by_doc = {
        doc_id: [Detection(**d) for d in entries]
        for doc_id, entries in run["_detections"].items()
    }

    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        filtered = _filter_by_threshold(detections_by_doc, threshold)
        report = aggregate(documents, filtered)
        rows.append(
            {
                "threshold": threshold,
                "precision": report["precision"],
                "span_recall": report["span_recall"],
                "typed_recall": report["typed_recall"],
                "f1": report["f1"],
                "false_negatives": report["false_negatives"],
                "false_positives": report["false_positives"],
                "document_leakage_rate": report["document_leakage_rate"],
                "useful_text_preservation": report["useful_text_preservation"],
                "composite_cost": report["composite_cost"],
                "missed_categories": missed_categories(report),
                "by_category": {
                    name: stats["span_recall"]
                    for name, stats in report["by_category"].items()
                },
            }
        )

    # Selection rule, applied in this order and stated explicitly so the choice
    # is auditable rather than a judgement call:
    #   1. never accept a threshold that loses span recall relative to 0.0 -
    #      a false negative is a disclosure;
    #   2. exclude 0.0 itself. Zero is not a calibrated value, it is the
    #      absence of one: it accepts every candidate the model emits,
    #      including the score-0 single-character spans observed in practice;
    #   3. among the rest, take the lowest composite cost (FN weighted 10x);
    #   4. break ties by preferring the LOWEST threshold, because a lower
    #      threshold keeps more low-confidence candidates, and a low-confidence
    #      possible Safe Harbor identifier must be redacted, never retained.
    baseline_recall = rows[0]["span_recall"]
    no_recall_loss = [r for r in rows if r["span_recall"] >= baseline_recall]
    admissible = [r for r in no_recall_loss if r["threshold"] > 0.0]
    if not admissible:
        # Every positive threshold loses recall. Selecting one would trade a
        # disclosure for tidiness, so this is reported rather than resolved.
        return {
            "detector": run["detector"],
            "partition": partition,
            "corpus_version": CORPUS_VERSION,
            "baseline_span_recall": baseline_recall,
            "selected_threshold": None,
            "selection_status": "no_safe_threshold_available",
            "selection_rule": (
                "every positive threshold loses span recall against 0.0; no "
                "calibrated threshold can be selected without accepting a "
                "false negative"
            ),
            "sweep": rows,
        }
    selected = min(
        admissible,
        key=lambda r: (r["composite_cost"], r["threshold"]),
    )

    return {
        "detector": run["detector"],
        "partition": partition,
        "corpus_version": CORPUS_VERSION,
        "baseline_span_recall": baseline_recall,
        "selection_rule": (
            "no span-recall loss versus 0.0; exclude 0.0 as uncalibrated; "
            "then lowest (10*false_negatives + false_positives); ties go to "
            "the lower threshold so low-confidence candidates stay redacted"
        ),
        "selected_threshold": selected["threshold"],
        "selection_status": "selected",
        "selected_metrics": {
            key: selected[key]
            for key in (
                "precision",
                "span_recall",
                "typed_recall",
                "f1",
                "false_negatives",
                "false_positives",
                "document_leakage_rate",
                "useful_text_preservation",
                "composite_cost",
            )
        },
        "sweep": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True, choices=ALL_DETECTORS + ("residual",))
    parser.add_argument("--partition", default=PARTITION_CALIB, choices=PARTITIONS)
    parser.add_argument("--sweep", action="store_true", help="sweep thresholds (calibration only)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="apply a locked threshold instead of sweeping")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--keep-detections", action="store_true",
                        help="keep raw detections in the written report")
    args = parser.parse_args()

    started = time.perf_counter()

    if args.detector == "residual":
        if args.threshold is not None:
            runners.set_model_thresholds(args.threshold)
        documents = partition_documents(args.partition)
        payload: Dict[str, Any] = {
            "detector": "residual_validator",
            "partition": args.partition,
            "corpus_version": CORPUS_VERSION,
            "threshold": args.threshold,
            "result": runners.run_residual_validator(documents),
        }
    else:
        run = run_detector(args.detector, args.partition)
        if args.sweep:
            payload = sweep_thresholds(run)
        else:
            if args.threshold is not None:
                documents = partition_documents(args.partition)
                detections = {
                    doc_id: [Detection(**d) for d in entries]
                    for doc_id, entries in run["_detections"].items()
                }
                filtered = _filter_by_threshold(detections, args.threshold)
                report = aggregate(documents, filtered)
                run["threshold"] = args.threshold
                run["metrics"] = report
                run["missed_categories"] = missed_categories(report)
            payload = run
            if not args.keep_detections:
                payload.pop("_detections", None)

    payload["environment"] = _environment_fingerprint()
    payload["models"] = _manifest_fingerprint()
    payload["wall_seconds"] = round(time.perf_counter() - started, 2)

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
        summary = payload.get("metrics") or payload.get("selected_metrics") or {}
        if summary:
            print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
