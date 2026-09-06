"""Privacy and utility evaluation, version 3.

Version 2 measured both halves on ten documents per partition, which cannot
express a rate finer than ten per cent. Version 3 runs the same two groups
over the two-hundred-document partitions of corpus v4.0, against the
configuration pinned in ``config/evaluation_freeze.json``.

    <model-venv-python> evaluations/phase11_evaluation.py --partition calibration
    <model-venv-python> evaluations/phase11_evaluation.py --partition heldout_v3

The held-out partition is run **once**. If the result shows a defect, the
failed result is what gets reported: fixing it and re-running the same
partition would produce a number about a configuration that had already seen
its own test set, and any further tuning requires a version 4 corpus with a
fresh held-out partition.

Every evaluated document also gets a transformation manifest recording what
changed and what the two verdicts were. Manifests carry categories, counts,
statuses and versions; never an original value.
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
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "phase11-evaluation-salt")

from evaluations.labelled_corpus import (  # noqa: E402
    CORPUS_VERSION,
    PARTITIONS,
    partition_documents,
)
from evaluations.metrics import Detection, aggregate, missed_categories  # noqa: E402
from services.transformation_manifest import build_manifest  # noqa: E402

FREEZE_PATH = BACKEND_ROOT / "config" / "evaluation_freeze.json"


def _frozen_configuration() -> Dict[str, Any]:
    """The pinned configuration, reported alongside every number.

    A result without its configuration is not evidence about anything.
    """
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def evaluate(partition: str, model_mode: str) -> Dict[str, Any]:
    os.environ["PHI_MODEL_MODE"] = model_mode
    from services import text_anonymization as ta

    ta._build_detectors.cache_clear()

    from services.ner_phi_detector import configured_model_name
    from services.text_anonymization import anonymize_clinical_text
    from services.utility_contract import contract_for

    documents = partition_documents(partition)
    contract = contract_for("text")
    model_name = configured_model_name()

    detections_by_doc: Dict[str, List[Detection]] = {}
    manifests: List[Dict[str, Any]] = []
    utility_rows: List[Dict[str, float]] = []
    surviving_gold = 0
    residual_docs: List[str] = []
    review_docs: List[str] = []
    latencies: List[float] = []
    failures: List[Dict[str, str]] = []
    classification_counts: Dict[str, int] = {}

    for document in documents:
        started = time.perf_counter()
        try:
            entities, _ = ta._detect_entities(document.text, model_name, "strict")
            outcome = anonymize_clinical_text(document.text, profile="strict")
        except Exception as exc:
            failures.append(
                {
                    "doc_id": document.doc_id,
                    "error_code": str(getattr(exc, "detail", type(exc).__name__)),
                }
            )
            detections_by_doc[document.doc_id] = []
            continue
        latencies.append(round((time.perf_counter() - started) * 1000, 2))

        detections_by_doc[document.doc_id] = [
            Detection(e.start, e.end, e.entity_type, e.source, e.score)
            for e in entities
        ]

        # The pipeline's own second-pass verdict. Re-scanning here without the
        # provenance map would report every surrogate as surviving text.
        residual = outcome["residual_phi_categories"]
        for finding in outcome["residual_findings"]:
            name = finding["classification"]
            classification_counts[name] = classification_counts.get(name, 0) + 1

        redacted = outcome["anonymized_text"]
        surviving_gold += sum(
            1 for span in document.spans if document.value(span) in redacted
        )
        if residual:
            residual_docs.append(document.doc_id)
        review_reasons = outcome.get("review_required_reasons") or []
        if review_reasons and not residual:
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
                review_reasons=review_reasons,
                model_mode=model_mode,
                generated_regions=outcome.get("provenance_counts", {}),
            )
        )

    privacy = aggregate(documents, detections_by_doc)
    held = [m["artifact_id"] for m in manifests if m["release_decision"] != "releasable"]
    total = len(documents) or 1

    def mean(name: str) -> float:
        values = [row[name] for row in utility_rows if name in row]
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "partition": partition,
        "corpus_version": CORPUS_VERSION,
        "model_mode": model_mode,
        "frozen_configuration": _frozen_configuration(),
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
            "missed_categories": missed_categories(privacy),
        },
        "release": {
            "documents": len(documents),
            "documents_held": len(held),
            "manual_review_rate": round(len(held) / total, 4),
            "blocked_by_residual_findings": round(len(residual_docs) / total, 4),
            "held_for_review_only": round(len(review_docs) / total, 4),
            "residual_findings_by_classification": dict(
                sorted(classification_counts.items())
            ),
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
            "latency_ms_mean": round(sum(latencies) / len(latencies), 2)
            if latencies
            else 0.0,
            "latency_ms_max": round(max(latencies), 2) if latencies else 0.0,
        },
        "failures": failures,
        "manifests": manifests,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", default="calibration", choices=PARTITIONS)
    parser.add_argument(
        "--model-mode", default="offline", choices=("offline", "legacy_test")
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = evaluate(args.partition, args.model_mode)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {path}")

    print(
        json.dumps(
            {
                "partition": report["partition"],
                "corpus_version": report["corpus_version"],
                "privacy": report["privacy"],
                "release": report["release"],
                "utility": report["utility"],
                "performance": report["performance"],
                "failures": len(report["failures"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
