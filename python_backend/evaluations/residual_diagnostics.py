"""Why documents are still held, one finding at a time.

The Phase 10 configuration sent sixty per cent of documents to manual review
with no expected synthetic PHI surviving. "Sixty per cent" is not a diagnosis,
so this script opens every remaining block and records what the validator
actually saw.

Each finding is reported as six fields and nothing else:

* detector;
* normalized category;
* evidence type;
* location type;
* whether it overlaps a generated replacement;
* final classification.

No detected value and no surrounding sentence is recorded. That constraint is
not decoration - a diagnostic report on a privacy pipeline is precisely the
document that gets pasted into a ticket - and ``test_residual_validator``
asserts the record shape independently.

    py -3.11 evaluations/residual_diagnostics.py --partition calibration
    py -3.11 evaluations/residual_diagnostics.py --partition development --json out.json

Runs on the mocked detector path by default so it needs no weights. Pass
``--model-mode offline`` to diagnose the real chain, from the model virtualenv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("PHI_MODEL_MODE", "legacy_test")
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "residual-diagnostics-salt")


def diagnose(partition: str) -> Dict[str, Any]:
    from evaluations.labelled_corpus import CORPUS_VERSION, partition_documents
    from services.text_anonymization import anonymize_clinical_text

    documents = partition_documents(partition)
    blocked: List[Dict[str, Any]] = []
    review_only: List[Dict[str, Any]] = []
    classification_counts: Counter = Counter()
    category_counts: Counter = Counter()
    detector_counts: Counter = Counter()
    evidence_counts: Counter = Counter()
    location_counts: Counter = Counter()
    failures: List[Dict[str, str]] = []

    for document in documents:
        try:
            outcome = anonymize_clinical_text(document.text, profile="strict")
        except Exception as exc:
            failures.append(
                {
                    "doc_id": document.doc_id,
                    "error_code": str(getattr(exc, "detail", type(exc).__name__)),
                }
            )
            continue

        # The pipeline's own classified findings. Re-scanning here would
        # repeat the very defect this phase removed: without the provenance
        # map, every surrogate reads as surviving text.
        findings = outcome["residual_findings"]
        for finding in findings:
            classification_counts[finding["classification"]] += 1
            category_counts[finding["category"]] += 1
            detector_counts[finding["detector"]] += 1
            evidence_counts[finding["evidence_type"]] += 1
            location_counts[finding["location_type"]] += 1

        blocking = [f for f in findings if f["blocking"]]
        review_reasons = outcome.get("review_required_reasons") or []

        if blocking:
            blocked.append({"doc_id": document.doc_id, "findings": blocking})
        elif review_reasons:
            review_only.append(
                {"doc_id": document.doc_id, "review_reasons": sorted(review_reasons)}
            )

    total = len(documents) or 1
    return {
        "partition": partition,
        "corpus_version": CORPUS_VERSION,
        "documents": len(documents),
        "documents_blocked_by_residual_findings": len(blocked),
        "documents_held_for_review_only": len(review_only),
        "documents_released_by_both_gates": len(documents)
        - len(blocked)
        - len(review_only)
        - len(failures),
        "block_rate": round(len(blocked) / total, 4),
        "manual_review_rate": round((len(blocked) + len(review_only)) / total, 4),
        "findings_by_classification": dict(sorted(classification_counts.items())),
        "findings_by_category": dict(sorted(category_counts.items())),
        "findings_by_detector": dict(sorted(detector_counts.items())),
        "findings_by_evidence_type": dict(sorted(evidence_counts.items())),
        "findings_by_location": dict(sorted(location_counts.items())),
        "blocked_documents": blocked,
        "review_only_documents": review_only,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", default="calibration")
    parser.add_argument(
        "--model-mode",
        default=None,
        choices=("offline", "legacy_test"),
        help="Overrides PHI_MODEL_MODE. 'offline' needs the model virtualenv.",
    )
    parser.add_argument("--json", default=None, help="Write the full report here.")
    args = parser.parse_args()

    if args.model_mode:
        os.environ["PHI_MODEL_MODE"] = args.model_mode

    report = diagnose(args.partition)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
        print(f"wrote {path}")

    summary = {
        key: report[key]
        for key in (
            "partition",
            "corpus_version",
            "documents",
            "documents_blocked_by_residual_findings",
            "documents_held_for_review_only",
            "block_rate",
            "manual_review_rate",
            "findings_by_classification",
            "findings_by_category",
            "findings_by_detector",
            "findings_by_evidence_type",
            "findings_by_location",
        )
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
