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

## Current state (phases 3-8)

The Phase 2 boundary above still holds. What changed since:

- **Text** now carries a third validator, `residual_phi_rescan`. A release is
  refused with `privacy_requirements_not_met` if anything survives redaction.
- **PDF and workbooks** are recognized modalities rather than rejected uploads.
  Both are inventoried and scanned across every readable surface, and both stay
  `manual_review_required` at best: no validated writer exists for either.
- **DICOM and NIfTI** now validate against the bytes they would hand out, and
  both carry `facial_reconstruction_not_mitigated` as a standing reason code.
  No defacing step exists, so improving the pixel scan does not lift it.
- **CSV** validates its serialized output against both the removal plan and the
  original input. The ingest route's posture is unchanged; the old reason code
  `serialized_output_validation_pending` is now split into
  `serialized_output_validation_passed` plus
  `tabular_release_policy_review_pending`.
- **`/anonymize_csv`** is policy-gated. It previously accepted no privacy
  profile at all and streamed rows regardless of the privacy verdict.
- **The index and preview paths** are gated. `/store` wrote client-supplied
  content straight into the searchable vector store, and `/simple_preview`
  streamed uploads back unmodified. Neither is possible now.

`research` still releases, previews, and indexes nothing, on any modality.

See [PHI_HARDENING.md](PHI_HARDENING.md) for the full release matrix, the
fail-closed inventory, and the known gaps.
