"""Joint threshold calibration for the combined detector chain.

Calibrating each model in isolation asks the wrong question. GLiNER at
threshold 0.0 has perfect span recall on the calibration partition but 420
false positives and zero useful-text preservation: it redacts nearly
everything. Every positive threshold costs it some recall of its own. Judged
alone, no threshold is admissible.

But neither model is the release authority. They are two detectors in a chain
alongside deterministic patterns and spaCy NER, and a span one model drops is
very often still caught by another. The question that actually matters is:

    what thresholds keep the **combined chain's** span recall intact while
    giving back the most useful text?

So this sweeps a grid of (stanford, gliner) thresholds, reconstructs the
combined chain at each point using the pipeline's own overlap resolver, and
selects against combined recall. Inference is run once per model at threshold
0.0; the grid then filters those captured candidates, which is exactly
equivalent and avoids re-running a 1.16 GB model 169 times.

The selection rule is defined once in ``SELECTION_RULE`` and applied by
``select_from_grid``, which is kept separate so a selection can be re-derived
from a saved grid without re-running inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

CACHE_ROOT = BACKEND_ROOT / ".model-cache"
os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_ROOT / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PHI_MODEL_MODE", "offline")
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "evaluation-salt")

from evaluations import detector_runners as runners  # noqa: E402
from evaluations.labelled_corpus import (  # noqa: E402
    CORPUS_VERSION,
    PARTITION_CALIB,
    partition_documents,
)
from evaluations.metrics import Detection, aggregate, missed_categories  # noqa: E402

REPORTS = BACKEND_ROOT / "evaluations" / "reports"

GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)


def _capture(partition: str) -> Dict[str, Dict[str, List[Detection]]]:
    """Run every detector once at threshold 0.0 and keep all candidates."""
    documents = partition_documents(partition)
    from services.local_model_detectors import DetectorConfig

    permissive = DetectorConfig(candidate_threshold=0.0, redaction_threshold=0.0)

    captured: Dict[str, Dict[str, List[Detection]]] = {}
    print("capturing rules ...", flush=True)
    captured["rules"] = runners.run_rules(documents).detections_by_doc
    print("capturing spacy ...", flush=True)
    captured["spacy"] = runners.run_spacy(documents).detections_by_doc
    print("capturing stanford ...", flush=True)
    captured["stanford"] = runners.run_stanford(documents, permissive).detections_by_doc
    runners.release_memory()
    print("capturing gliner ...", flush=True)
    captured["gliner"] = runners.run_gliner(documents, permissive).detections_by_doc
    runners.release_memory()
    return captured


def _combine(
    captured: Dict[str, Dict[str, List[Detection]]],
    documents: Sequence[Any],
    stanford_threshold: float,
    gliner_threshold: float,
) -> Dict[str, List[Detection]]:
    """Rebuild the chain at a threshold pair, using the pipeline's resolver."""
    from services.phi_detection import DetectedEntity, resolve_overlaps

    merged: Dict[str, List[Detection]] = {}
    for document in documents:
        pool: List[Detection] = []
        pool.extend(captured["rules"].get(document.doc_id, []))
        pool.extend(captured["spacy"].get(document.doc_id, []))
        pool.extend(
            d
            for d in captured["stanford"].get(document.doc_id, [])
            if d.score is None or d.score >= stanford_threshold
        )
        pool.extend(
            d
            for d in captured["gliner"].get(document.doc_id, [])
            if d.score is None or d.score >= gliner_threshold
        )
        entities = [
            DetectedEntity(
                entity_type=d.category,
                start=d.start,
                end=d.end,
                source=d.source,
                score=d.score,
            )
            for d in pool
        ]
        resolved = resolve_overlaps(entities, len(document.text))
        merged[document.doc_id] = [
            Detection(
                start=e.start,
                end=e.end,
                category=e.entity_type,
                source=e.source,
                score=e.score,
            )
            for e in resolved
        ]
    return merged


SELECTION_RULE = (
    "1. combined span recall must not fall below the all-zero baseline - a "
    "false negative is a disclosure and is never traded away. "
    "2. zero is excluded for either model: it is not a calibrated value, it "
    "is the absence of one. "
    "3. among the rest, maximise typed recall, which improves how a finding "
    "is labelled without trading away any span. "
    "4. ties go to the LOWEST thresholds, keeping the widest safety margin "
    "on unseen data, because precision gained by raising a threshold is "
    "bought with candidates that would otherwise have been redacted."
)


def select_from_grid(
    rows: Sequence[Dict[str, Any]],
    baseline_recall: float,
) -> Tuple[Dict[str, Any], str]:
    """Apply the documented selection rule to a completed grid.

    Kept separate from the sweep so a selection can be re-derived from a saved
    grid without re-running inference, and so the rule itself is testable.
    """
    admissible = [
        r
        for r in rows
        if r["span_recall"] >= baseline_recall
        and r["stanford"] > 0.0
        and r["gliner"] > 0.0
    ]
    if not admissible:
        # Nothing preserves combined recall. Report rather than trade a
        # disclosure for a tidier score.
        return (
            max(rows, key=lambda r: (r["span_recall"], -r["composite_cost"])),
            "no_threshold_preserves_combined_recall",
        )
    selected = max(
        admissible,
        key=lambda r: (
            r["typed_recall"],
            r["useful_text_preservation"],
            -r["stanford"],
            -r["gliner"],
        ),
    )
    return selected, "selected"


def calibrate(partition: str = PARTITION_CALIB) -> Dict[str, Any]:
    if partition != PARTITION_CALIB:
        raise SystemExit("joint calibration runs on the calibration partition only")

    started = time.perf_counter()
    documents = partition_documents(partition)
    captured = _capture(partition)

    baseline = aggregate(documents, _combine(captured, documents, 0.0, 0.0))
    baseline_recall = baseline["span_recall"]
    print(f"combined baseline span recall @ (0.0, 0.0) = {baseline_recall}", flush=True)

    rows: List[Dict[str, Any]] = []
    for stanford_threshold in GRID:
        for gliner_threshold in GRID:
            report = aggregate(
                documents,
                _combine(captured, documents, stanford_threshold, gliner_threshold),
            )
            rows.append(
                {
                    "stanford": stanford_threshold,
                    "gliner": gliner_threshold,
                    "span_recall": report["span_recall"],
                    "typed_recall": report["typed_recall"],
                    "precision": report["precision"],
                    "f1": report["f1"],
                    "false_negatives": report["false_negatives"],
                    "false_positives": report["false_positives"],
                    "document_leakage_rate": report["document_leakage_rate"],
                    "useful_text_preservation": report["useful_text_preservation"],
                    "composite_cost": report["composite_cost"],
                    "missed_categories": missed_categories(report),
                }
            )

    selected, status = select_from_grid(rows, baseline_recall)

    return {
        "corpus_version": CORPUS_VERSION,
        "partition": partition,
        "baseline_combined_span_recall": baseline_recall,
        "baseline_useful_text_preservation": baseline["useful_text_preservation"],
        "baseline_false_positives": baseline["false_positives"],
        "selection_status": status,
        "selection_rule": SELECTION_RULE,
        "selected": {
            "stanford_candidate_threshold": selected["stanford"],
            "gliner_candidate_threshold": selected["gliner"],
        },
        "selected_metrics": {
            key: selected[key]
            for key in (
                "span_recall",
                "typed_recall",
                "precision",
                "f1",
                "false_negatives",
                "false_positives",
                "document_leakage_rate",
                "useful_text_preservation",
                "composite_cost",
                "missed_categories",
            )
        },
        "grid": rows,
        "wall_seconds": round(time.perf_counter() - started, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPORTS / "joint_calibration.local.json"))
    parser.add_argument(
        "--reselect",
        action="store_true",
        help="re-apply the selection rule to a saved grid without re-running inference",
    )
    args = parser.parse_args()

    if args.reselect:
        result = json.loads(Path(args.out).read_text(encoding="utf-8"))
        selected, status = select_from_grid(
            result["grid"], result["baseline_combined_span_recall"]
        )
        result["selection_status"] = status
        result["selection_rule"] = SELECTION_RULE
        result["selected"] = {
            "stanford_candidate_threshold": selected["stanford"],
            "gliner_candidate_threshold": selected["gliner"],
        }
        result["selected_metrics"] = {
            key: selected[key]
            for key in (
                "span_recall", "typed_recall", "precision", "f1",
                "false_negatives", "false_positives", "document_leakage_rate",
                "useful_text_preservation", "composite_cost", "missed_categories",
            )
        }
    else:
        result = calibrate()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nwrote {out}")
    print(f"status   : {result['selection_status']}")
    print(f"selected : {result['selected']}")
    print(json.dumps(result["selected_metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
