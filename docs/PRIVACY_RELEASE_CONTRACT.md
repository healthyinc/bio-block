# Privacy release contract

## Phase 2 boundary

`safe_harbor_v1` is the only policy eligible to authorize automatic release.
The existing `strict` request value is retained as a compatibility alias for
that policy. Eligibility is necessary but not sufficient: a sanitizer-issued
release decision must also bind the exact validated artifact digest.

`research` maps to an expert-determination workflow. Public ingestion and
download endpoints return `expert_determination_required`, with no previewable,
downloadable, indexable, or releasable content. They do not invoke modality
handlers for research requests.

Until modality-specific writers and final-byte validators are implemented,
DICOM, NIfTI, WSI, and tabular ingestion results remain
`manual_review_required` and all downstream statuses are blocked. Text retains
the existing response shape after typed detection and deterministic redaction.

## Intentional compatibility change

`POST /anonymize_dicom` now returns HTTP 409 with a release decision instead of
a DICOM attachment. The previous endpoint sanitized metadata but did not prove
that pixel data and final serialized bytes were safe. Download behavior will be
restored only through the unified pipeline after those validation steps exist.

Legacy text and PDF JSON no longer includes original detected values, raw
offsets, or original page text. Entity metadata is limited to normalized types
and counts.

## Evaluation claim boundary

Zero residual synthetic canaries is an acceptance condition for the evaluation
suite. It is not proof that an artifact, model, modality, or deployment has zero
PHI leakage. Unsupported processing and uncertain findings remain blocked or
require manual review.
