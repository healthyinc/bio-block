"""Privacy and utility evaluation, version 2.

Phase 9 measured privacy only, and shipped a configuration with zero expected
leakage and useful-text preservation of 0.214. This reports the two as
separate groups, because they answer different questions and a good number in
one does not excuse a bad number in the other.

    <model-venv-python> evaluations/phase10_evaluation.py --partition calibration
    <model-venv-python> evaluations/phase10_evaluation.py --partition heldout_v2

Each evaluated document also gets a transformation manifest recording what was
changed and what the two verdicts were. Manifests carry categories, counts and
statuses; never an original value.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

CACHE = BACKEND_ROOT / ".model-cache"
os.environ.setdefault("HF_HOME", str(CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE / "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "phase10-evaluation-salt")

from evaluations.labelled_corpus import (  # noqa: E402
    CORPUS_VERSION,
    PARTITIONS,
    partition_documents,
)
from evaluations.metrics import Detection, aggregate, missed_categories  # noqa: E402
from evaluations.transformation_manifest import build_manifest  # noqa: E402


def evaluate(partition: str, model_mode: str) -> Dict[str, Any]:
    os.environ["PHI_MODEL_MODE"] = model_mode
    from services import text_anonymization as ta

    ta._build_detectors.cache_clear()

    from services.ner_phi_detector import configured_model_name
    from services.text_anonymization import (
        anonymize_clinical_text,
        residual_phi_categories,
    )
    from services.utility_contract import contract_for

    documents = partition_documents(partition)
    contract = contract_for("text")
    model_name = configured_model_name()

    detections_by_doc: Dict[str, List[Detection]] = {}
    manifests: List[Dict[str, Any]] = []
    utility_rows: List[Dict[str, float]] = []
    residual_docs: List[str] = []
    surviving_gold = 0
    review_docs: List[str] = []
    latencies: List[float] = []
    failures: List[Dict[str, str]] = []

    for document in documents:
        started = time.perf_counter()
        try:
            entities = ta._detect_entities(document.text, model_name, "strict")
            outcome = anonymize_clinical_text(document.text, profile="strict")
            residual = residual_phi_categories(outcome["anonymized_text"])
        except Exception as exc:
            failures.append(
                {"doc_id": document.doc_id, "error_code": str(getattr(exc, "detail", type(exc).__name__))}
            )
            detections_by_doc[document.doc_id] = []
            continue
        latencies.append(round((time.perf_counter() - started) * 1000, 2))

        detections_by_doc[document.doc_id] = [
            Detection(e.start, e.end, e.entity_type, e.source, e.score) for e in entities
        ]

        redacted = outcome["anonymized_text"]
        survived = [s for s in document.spans if document.value(s) in redacted]
        surviving_gold += len(survived)
        if residual:
            residual_docs.append(document.doc_id)
        if outcome.get("review_required_reasons"):
            review_docs.append(document.doc_id)

        utility = outcome["utility_metrics"]
        utility_rows.append(utility)
        verdict = contract.evaluate(utility)

        manifests.append(
            build_manifest(
                artifact_id=document.doc_id,
                modality="text",
                policy="safe_harbor_v1",
                detected=outcome["detected_entities"],
                sources=outcome["detection_sources"],
                surrogate_counts=outcome.get("surrogate_counts", {}),
                utility=utility,
                utility_passed=verdict.passed,
                residual_categories=residual,
                review_reasons=outcome.get("review_required_reasons", []),
                model_mode=model_mode,
            )
        )

    privacy = aggregate(documents, detections_by_doc)
    manual_review = [
        m["artifact_id"]
        for m in manifests
        if m["release_decision"] != "releasable"
    ]

    def mean(name: str) -> float:
        values = [row[name] for row in utility_rows if name in row]
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "partition": partition,
        "corpus_version": CORPUS_VERSION,
        "model_mode": model_mode,
        "privacy": {
            "phi_precision": privacy["precision"],
            "phi_recall": privacy["span_recall"],
            "typed_recall": privacy["typed_recall"],
            "f1": privacy["f1"],
            "false_negatives": privacy["false_negatives"],
            "false_positives": privacy["false_positives"],
            "document_leakage_rate": privacy["document_leakage_rate"],
            "residual_canaries": surviving_gold,
            "documents_with_residual_findings": len(residual_docs),
            "manual_review_rate": round(len(manual_review) / len(documents), 4),
            "missed_categories": missed_categories(privacy),
        },
        "utility": {
            "clinical_term_preservation": mean("clinical_term_preservation"),
            "content_token_preservation": mean("content_token_preservation"),
            "numeric_preservation": mean("numeric_preservation"),
            "useful_text_preservation": privacy["useful_text_preservation"],
            "negative_terms_redacted": privacy["negative_terms_redacted"],
            "negative_terms": privacy["negative_terms"],
        },
        "performance": {
            "documents": len(documents),
            "latency_ms_mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "latency_ms_max": round(max(latencies), 2) if latencies else 0.0,
        },
        "failures": failures,
        "manifests": manifests,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", default="calibration", choices=PARTITIONS)
    parser.add_argument("--model-mode", default="offline", choices=("offline", "legacy_test"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = evaluate(args.partition, args.model_mode)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {path}")
    print(json.dumps({"privacy": report["privacy"], "utility": report["utility"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
