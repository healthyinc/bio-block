"""Run the synthetic-canary evaluation across every hardened path.

    py -3.11 evaluations/run_evaluation.py            # summary
    py -3.11 evaluations/run_evaluation.py --json     # machine-readable

By default this runs in `legacy_test` model mode, so it uses the rule-based and
spaCy detectors and downloads nothing. Set PHI_MODEL_MODE=offline to evaluate
with the pinned local models, which must already be cached (see
docs/MODEL_MANIFEST.md).

What is measured, per case:

* **category recall** - which expected PHI categories the detector proposed.
* **residual canaries** - which canary values survived into anything the
  pipeline was willing to release, index, or preview. This is the number that
  matters, and it must be zero.
* **release posture** - what the release decision actually was.

Reports name canaries by index, never by value, so no report format normalizes
carrying a matched string.

Zero residual canaries is an acceptance condition for this suite. It is not
proof that an artifact, model, modality, or deployment has zero PHI leakage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluations.canary_corpus import (  # noqa: E402
    CORPUS,
    GATE_CORPUS,
    CanaryCase,
    canary_labels,
    residual_canaries,
)

EXIT_OK = 0
EXIT_RESIDUAL_CANARY = 1
EXIT_POSTURE_MISMATCH = 2


def _released_surface(response: Dict[str, Any]) -> str:
    """Everything the pipeline was willing to hand back, as one string.

    A blocked decision must not carry content, so scanning the whole response
    is exactly the right test: anything that reaches here reached a caller.
    """
    return json.dumps(response, default=str)


def _run_ingest_case(case: CanaryCase, profile: str) -> Dict[str, Any]:
    from services.ingestion import IngestionError, route_for_ingestion

    try:
        payload = case.build()
    except ImportError as exc:
        return {"case_id": case.case_id, "status": "skipped", "reason": str(exc)}

    kwargs: Dict[str, Any] = {
        "filename": case.filename,
        "content_type": case.content_type,
        "header": payload[:4096],
        "profile": profile,
    }
    if case.modality == "text":
        kwargs["text_content"] = payload
        kwargs["study_salt"] = "evaluation-salt"
    else:
        kwargs["file_content"] = payload

    try:
        response = route_for_ingestion(**kwargs)
    except IngestionError as exc:
        return {
            "case_id": case.case_id,
            "modality": case.modality,
            "profile": profile,
            "status": "blocked_at_ingest",
            "detail": exc.detail,
            "residual_canaries": [],
            "releasable": False,
        }

    surface = _released_surface(response)
    survived = residual_canaries(surface, case.canaries)
    detected = set(response.get("detected_entities", {}) or {})
    expected = set(case.expected_categories)

    return {
        "case_id": case.case_id,
        "modality": case.modality,
        "profile": profile,
        "status": "processed",
        "anonymization_status": response.get("anonymization_status"),
        "releasable": bool(response.get("release_decision", {}).get("releasable")),
        "reason_codes": response.get("release_decision", {}).get("reason_codes", []),
        "expected_categories": sorted(expected),
        "detected_categories": sorted(detected),
        "missed_categories": sorted(expected - detected),
        "category_recall": (
            round(len(expected & detected) / len(expected), 4) if expected else None
        ),
        "residual_canaries": canary_labels(survived),
        "residual_canary_count": len(survived),
    }


def _run_gate_case(case: CanaryCase) -> Dict[str, Any]:
    from services.release_gate import ReleaseGateError, sanitized_preview

    try:
        payload = case.build()
    except ImportError as exc:
        return {"case_id": case.case_id, "status": "skipped", "reason": str(exc)}

    try:
        outcome = sanitized_preview(
            payload, filename=case.filename, content_type=case.content_type
        )
    except ReleaseGateError as exc:
        return {
            "case_id": case.case_id,
            "status": "blocked_at_gate",
            "detail": exc.detail,
            "residual_canaries": [],
            "releasable": False,
        }

    surface = outcome.content.decode("latin-1") if outcome.content else ""
    surface += json.dumps(outcome.safe_summary(), default=str)
    survived = residual_canaries(surface, case.canaries)

    return {
        "case_id": case.case_id,
        "modality": case.modality,
        "status": "processed",
        "preview_status": outcome.status,
        "releasable": outcome.released,
        "reason_codes": list(outcome.reason_codes),
        "residual_canaries": canary_labels(survived),
        "residual_canary_count": len(survived),
    }


def _run_index_gate() -> Dict[str, Any]:
    """The index path is the one a blocked upload could previously bypass."""
    from evaluations.canary_corpus import CLINICAL_NOTE, NOTE_CANARIES
    from services.release_gate import sanitize_for_index

    result = sanitize_for_index(
        {
            "dataset_title": "Evaluation cohort",
            "summary": "Aggregate outcomes.",
            "extracted_content": CLINICAL_NOTE,
        }
    )
    surface = json.dumps(dict(result.fields), default=str)
    surface += json.dumps(result.safe_summary(), default=str)
    survived = residual_canaries(surface, NOTE_CANARIES)

    return {
        "case_id": "index_gate_clinical_note",
        "modality": "index",
        "status": "processed",
        "sanitization_status": result.status,
        "releasable": result.cleared,
        "detected_categories": sorted(result.detected_entities),
        "residual_canaries": canary_labels(survived),
        "residual_canary_count": len(survived),
    }


def run_evaluation() -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []

    for case in CORPUS:
        cases.append(_run_ingest_case(case, profile="strict"))

    # Research must never release, preview, or index anything, on any modality.
    for case in CORPUS:
        research = _run_ingest_case(case, profile="research")
        research["case_id"] = f"{case.case_id}__research"
        cases.append(research)

    for case in GATE_CORPUS:
        cases.append(_run_gate_case(case))

    cases.append(_run_index_gate())

    processed = [case for case in cases if case.get("status") != "skipped"]
    residual_total = sum(case.get("residual_canary_count", 0) for case in processed)
    research_released = [
        case["case_id"]
        for case in processed
        if case["case_id"].endswith("__research") and case.get("releasable")
    ]

    posture_mismatches: List[str] = []
    expected_posture = {case.case_id: case.expect_releasable for case in CORPUS}
    for case in processed:
        expected = expected_posture.get(case["case_id"])
        if expected is None:
            continue
        if bool(case.get("releasable")) != bool(expected):
            posture_mismatches.append(case["case_id"])

    recalls = [
        case["category_recall"]
        for case in processed
        if case.get("category_recall") is not None
    ]

    return {
        "model_mode": os.getenv("PHI_MODEL_MODE", "offline"),
        "cases_run": len(processed),
        "cases_skipped": len(cases) - len(processed),
        "residual_canary_total": residual_total,
        "research_cases_released": research_released,
        "release_posture_mismatches": posture_mismatches,
        "mean_category_recall": (
            round(sum(recalls) / len(recalls), 4) if recalls else None
        ),
        "acceptance": {
            "zero_residual_canaries": residual_total == 0,
            "research_never_releases": not research_released,
            "release_posture_as_expected": not posture_mismatches,
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the full report")
    args = parser.parse_args()

    os.environ.setdefault("PHI_MODEL_MODE", "legacy_test")
    os.environ.setdefault("BIOBLOCK_STUDY_SALT", "evaluation-salt")

    report = run_evaluation()

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"model mode           : {report['model_mode']}")
        print(f"cases run / skipped  : {report['cases_run']} / {report['cases_skipped']}")
        print(f"residual canaries    : {report['residual_canary_total']}")
        print(f"mean category recall : {report['mean_category_recall']}")
        for name, passed in report["acceptance"].items():
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        for case in report["cases"]:
            if case.get("status") == "skipped":
                print(f"  skipped {case['case_id']}: {case.get('reason')}")

    if report["residual_canary_total"]:
        return EXIT_RESIDUAL_CANARY
    if report["research_cases_released"] or report["release_posture_mismatches"]:
        return EXIT_POSTURE_MISMATCH
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
