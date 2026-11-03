"""Transformation manifests for every processed artifact.

One record per artifact describing what was changed, which versions of which
rules and models changed it, and why it was or was not released. A manifest is
a provenance record, so it travels: into reports, to a reviewer, potentially
into a metadata index or a chain of custody. That is exactly why it carries
categories, counts, statuses and versions only.

There is deliberately no field for the surrogate mapping, the original value,
the extracted text or the uploaded filename. Recording that "PATIENT_001 was
Jane Doe" in a manifest would undo the whole point of the surrogate, and a
field that exists will eventually be filled. ``FORBIDDEN_KEYS`` is asserted by
test rather than trusted to review.

The version block is the part that makes a manifest useful a year later. A
release decision is only reproducible if you can say which detector, which
threshold set, which vocabulary and which evidence model produced it; without
that, "this artifact passed" is a claim about a system that no longer exists.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

MANIFEST_VERSION = "transformation-manifest-v2"

#: Keys that must never appear anywhere in a manifest, checked by test.
FORBIDDEN_KEYS = frozenset(
    {
        "original",
        "originals",
        "values",
        "mapping",
        "surrogate_mapping",
        "text",
        "raw",
        "filename",
        "original_filename",
        "spans",
        "extracted_text",
    }
)

DECISION_RELEASABLE = "releasable"
DECISION_BLOCKED_PRIVACY = "blocked_privacy"
DECISION_MANUAL_REVIEW = "manual_review_required"


def component_versions(model_mode: str) -> Dict[str, Any]:
    """Which rules and models were in force. Names and revisions only.

    Read from the pinned manifest and the locked threshold file, never by
    loading a model: building a manifest must not pull weights into a process
    that was only asked to describe what it did.
    """
    from services.clinical_vocabulary import VOCABULARY_VERSION
    from services.detection_evidence import EVIDENCE_MODEL_VERSION
    from services.modality_utility import MEASUREMENT_VERSION
    from services.utility_contract import CONTRACT_VERSION

    versions: Dict[str, Any] = {
        "manifest": MANIFEST_VERSION,
        "utility_contract": CONTRACT_VERSION,
        "utility_measurement": MEASUREMENT_VERSION,
        "evidence_model": EVIDENCE_MODEL_VERSION,
        "clinical_vocabulary": VOCABULARY_VERSION,
        "model_mode": model_mode,
    }

    try:
        from services.local_model_detectors import (
            detector_specs,
            load_locked_thresholds,
        )

        versions["detectors"] = {
            name: {"repo": spec.repo_id, "revision": spec.revision}
            for name, spec in sorted(detector_specs().items())
        }
        versions["thresholds"] = {
            name: dict(sorted(entry.items()))
            for name, entry in sorted(load_locked_thresholds().items())
        }
    except Exception:
        # The pinned configuration is unavailable in this process - the mocked
        # test path, for instance. Say so; do not report an empty dict as if
        # the models had no revisions.
        versions["detectors"] = "not_recorded"
        versions["thresholds"] = "not_recorded"

    return versions


def _review_reason(decision: str, reasons: Sequence[str]) -> Optional[str]:
    """The code a reviewer is actually being asked to act on.

    A decision often carries several codes, some of which record checks that
    *passed* on the way to a block. Naming one of those as the reason for
    review sends the reviewer looking for a problem in the one thing that
    went right.
    """
    if decision != DECISION_MANUAL_REVIEW:
        return None
    actionable = [code for code in reasons if not code.endswith("_passed")]
    return (actionable or list(reasons) or [DECISION_MANUAL_REVIEW])[0]


def build_manifest(
    artifact_id: str,
    modality: str,
    policy: str,
    detected: Mapping[str, int],
    sources: Mapping[str, int],
    surrogate_counts: Mapping[str, int],
    utility: Mapping[str, Any],
    utility_passed: bool,
    residual_categories: Mapping[str, int],
    review_reasons: Sequence[str],
    model_mode: str,
    contract_version: str = "utility-contract-v1",
    pixel_regions_modified: int = 0,
    generated_regions: Optional[Mapping[str, int]] = None,
    unsupported_reasons: Sequence[str] = (),
) -> Dict[str, Any]:
    """Build one manifest. Counts, categories and versions only, never a value.

    ``artifact_id`` is a caller-chosen identifier - a corpus document id or a
    content hash. It must not be the uploaded filename, which is itself a
    common carrier of a patient's name.
    """
    privacy_passed = not residual_categories
    needs_review = bool(review_reasons) or bool(unsupported_reasons)

    # Privacy first and it is never traded against utility; then review; then
    # utility. Only when all three clear can the policy release anything.
    if not privacy_passed:
        decision = DECISION_BLOCKED_PRIVACY
        reasons = ["privacy_requirements_not_met"]
    elif needs_review:
        decision = DECISION_MANUAL_REVIEW
        reasons = sorted(set(list(review_reasons) + list(unsupported_reasons)))
    elif not utility_passed:
        decision = DECISION_MANUAL_REVIEW
        reasons = ["utility_validation_failed"]
    else:
        decision = DECISION_RELEASABLE
        reasons = ["safe_harbor_technical_checks_passed"]

    regions = dict(generated_regions or {})
    return {
        "manifest_version": MANIFEST_VERSION,
        "artifact_id": artifact_id,
        "modality": modality,
        "policy_version": policy,
        "contract_version": contract_version,
        "model_mode": model_mode,
        "component_versions": component_versions(model_mode),
        "categories_transformed": dict(sorted(detected.items())),
        "transformation_types": {
            "replaced_with_surrogate": sum(surrogate_counts.values()),
            "redacted_fixed_token": max(
                0, sum(detected.values()) - sum(surrogate_counts.values())
            ),
            "generalized": int(detected.get("AGE_OVER_89", 0)),
            "pixel_regions_covered": int(pixel_regions_modified),
        },
        "fields_changed": {
            "removed_or_replaced": sum(detected.values()),
            "distinct_entities_replaced": sum(surrogate_counts.values()),
            "by_surrogate_kind": dict(sorted(surrogate_counts.items())),
        },
        "regions_modified": {
            "pixel_regions": int(pixel_regions_modified),
            "text_spans": sum(detected.values()),
            # Regions the sanitizer wrote into the output, by kind, taken from
            # the transformation-provenance map. This is what the second-pass
            # validator attributes findings against.
            "generated_regions": dict(sorted(regions.items())),
            "generated_region_total": sum(regions.values()),
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
        # Named separately from reason_codes so a reviewer reading a manifest
        # does not have to work out which of several codes is the one asking
        # for their attention.
        "manual_review_reason": _review_reason(decision, reasons),
    }
