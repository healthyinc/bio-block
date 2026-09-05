"""Transformation manifests.

One record per evaluated artifact describing what was changed and why it was
or was not released. A manifest is a provenance record, so it is written to
reports and could reach a reviewer, a metadata index or a chain of custody -
which is exactly why it must never contain an original value. It carries
categories, counts, statuses and versions only.

There is deliberately no field for the surrogate mapping. Recording that
"PATIENT_001 was Jane Doe" in a manifest would undo the whole point of the
surrogate, and a field that exists will eventually be filled.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

MANIFEST_VERSION = "transformation-manifest-v1"

#: Keys that must never appear in a manifest, checked by test.
FORBIDDEN_KEYS = frozenset(
    {"original", "originals", "values", "mapping", "surrogate_mapping", "text", "raw"}
)


def build_manifest(
    artifact_id: str,
    modality: str,
    policy: str,
    detected: Mapping[str, int],
    sources: Mapping[str, int],
    surrogate_counts: Mapping[str, int],
    utility: Mapping[str, float],
    utility_passed: bool,
    residual_categories: Mapping[str, int],
    review_reasons: Sequence[str],
    model_mode: str,
    contract_version: str = "utility-contract-v1",
) -> Dict[str, Any]:
    """Build one manifest. Counts and categories only, never a value."""
    privacy_passed = not residual_categories
    needs_review = bool(review_reasons)

    # Privacy first and it is never traded against utility; then review; then
    # utility. Only when all three clear can the policy release anything.
    if not privacy_passed:
        decision = "blocked_privacy"
        reasons = ["privacy_requirements_not_met"]
    elif needs_review:
        decision = "manual_review_required"
        reasons = sorted(set(review_reasons))
    elif not utility_passed:
        decision = "manual_review_required"
        reasons = ["utility_validation_failed"]
    else:
        decision = "releasable"
        reasons = ["safe_harbor_technical_checks_passed"]

    return {
        "manifest_version": MANIFEST_VERSION,
        "artifact_id": artifact_id,
        "modality": modality,
        "policy_version": policy,
        "contract_version": contract_version,
        "model_mode": model_mode,
        "categories_transformed": dict(sorted(detected.items())),
        "transformation_types": {
            "replaced_with_surrogate": sum(surrogate_counts.values()),
            "redacted_fixed_placeholder": max(
                0, sum(detected.values()) - sum(surrogate_counts.values())
            ),
            "generalized": int(detected.get("AGE_OVER_89", 0)),
        },
        "fields_changed": {
            "removed_or_replaced": sum(detected.values()),
            "distinct_entities_replaced": sum(surrogate_counts.values()),
            "by_surrogate_kind": dict(sorted(surrogate_counts.items())),
        },
        "regions_modified": {
            # Text has no pixel regions; the key is present so the manifest
            # shape is stable across modalities.
            "pixel_regions": 0,
            "text_spans": sum(detected.values()),
        },
        "detection_sources": dict(sorted(sources.items())),
        "privacy_checks": {
            "passed": privacy_passed,
            "residual_categories": dict(sorted(residual_categories.items())),
        },
        "utility_checks": {
            "passed": utility_passed,
            "metrics": {k: v for k, v in sorted(utility.items())},
        },
        "release_decision": decision,
        "reason_codes": reasons,
    }
