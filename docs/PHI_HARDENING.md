# Multimodal PHI hardening

How PHI is detected, removed, validated, and gated across every modality this
backend accepts, and what is deliberately still blocked.

Companion documents:

- [PRIVACY_RELEASE_CONTRACT.md](PRIVACY_RELEASE_CONTRACT.md) — the release contract
- [MODEL_MANIFEST.md](MODEL_MANIFEST.md) — pinned local models
- [DOCUMENT_HARDENING.md](DOCUMENT_HARDENING.md) — TXT and PDF
- [IMAGING_HARDENING.md](IMAGING_HARDENING.md) — DICOM, NIfTI, WSI, raster
- [TABULAR_ROUTING.md](TABULAR_ROUTING.md) — CSV and workbooks
- [DOWNSTREAM_ENFORCEMENT.md](DOWNSTREAM_ENFORCEMENT.md) — index and preview gates

## The shape of the pipeline

```
upload ──▶ modality detection ──▶ policy resolution ──▶ handler
                                        │                  │
                                        │                  ▼
                                        │            detectors propose spans
                                        │            (patterns, spaCy NER,
                                        │             Stanford, GLiNER)
                                        │                  │
                                        │                  ▼
                                        │            Safe Harbor transformation
                                        │                  │
                                        │                  ▼
                                        │            validation of the bytes
                                        │            that would be handed out
                                        │                  │
                                        └────────▶  release decision
                                                           │
                                    ┌──────────────────────┴────────────┐
                                    ▼                                   ▼
                              releasable                       blocked / manual
                                    │                          review / expert
                                    ▼                          determination
                          index gate · preview gate
```

Two properties hold throughout:

1. **Models are detectors, never release authorities.** Stanford deidentifier
   and GLiNER propose candidate spans that are normalized into one internal
   taxonomy. The Week 6 Safe Harbor logic decides what happens to them.
2. **Validation is on the bytes, not the plan.** Every path re-reads what it is
   about to hand out and checks it, rather than trusting the report of what it
   did.

## Release matrix

| Modality | Automatic release | Why |
|---|---|---|
| TXT | **Yes**, under `safe_harbor_v1` | Redacted, then re-scanned for residual PHI |
| CSV | No — `manual_review_required` | Serialized output is validated; whether generalized quasi-identifiers clear the release bar is an open policy question |
| PDF | No — `manual_review_required` at best | No validated PDF writer |
| Workbook | No — `manual_review_required` at best | No validated workbook writer |
| DICOM | No | `facial_reconstruction_not_mitigated` — no defacing step |
| NIfTI | No | Same standing blocker |
| WSI | No | No validated slide writer |
| Raster preview | Yes, verified | Every detected text region filled, then verified twice |
| Any modality, `research` | **Never** | `expert_determination_required` |

## Status vocabulary

Only these appear as status strings:

`safe_harbor_technical_checks_passed`, `residual_phi_not_detected`,
`privacy_requirements_not_met`, `manual_review_required`,
`expert_determination_required`, `unsupported_or_unscannable`.

Nothing in this codebase claims HIPAA compliance. These are technical checks
against the Safe Harbor identifier list, not a legal determination.

## Fail-closed inventory

Every one of these blocks. None falls back to original or partially sanitized
content.

| Condition | Where |
|---|---|
| Model load, checksum, inference, timeout, malformed output | `local_model_detectors` |
| Unknown `PHI_MODEL_MODE` | `local_model_detectors` |
| Undecodable or NUL-bearing text | `ingestion.anonymize_text` |
| Residual PHI after redaction | `text_anonymization.residual_phi_categories` |
| Encrypted, unparseable, oversized, or macro/attachment-bearing documents | `document_sanitization`, `workbook_sanitization` |
| A PDF page with raster content | `document_sanitization` |
| DICOM text detected but not cleared | `ocr_redaction` |
| DICOM redaction absent from the re-read bytes | `ocr_redaction._validate_sanitized_dicom` |
| Raster OCR unavailable, failing, or leaving readable text | `raster_redaction` |
| NIfTI scrub absent from the re-read bytes | `nifti_anonymization` |
| Removed CSV column or identifier value in the serialized output | `tabular_validation` |
| PHI in text or metadata headed for the index | `release_gate.sanitize_for_index` |
| A preview modality with no sanitized form | `release_gate.sanitized_preview` |

## Disclosure hygiene

No matched value ever reaches a log, an exception, a telemetry field, or a test
snapshot. Reports carry categories, counts, detector names, offsets into
transient input, and status codes. The evaluation harness goes further and
names canaries by index rather than by value, so no report format normalizes
carrying a matched string.

Endpoints that previously interpolated `str(e)` into a 500 response — the PDF,
image, store, and preview routes — now return fixed messages, because that
interpolation could echo document content.

## Evaluation

```
py -3.11 evaluations/run_evaluation.py            # summary
py -3.11 evaluations/run_evaluation.py --json     # full report
py -3.11 -m pytest tests/test_evaluation_harness.py   # same, as a gate
```

The harness runs a synthetic canary corpus through every hardened path and
measures:

- **category recall** — which expected categories the detectors proposed;
- **residual canaries** — which canary values survived into anything the
  pipeline was willing to release, index, or preview;
- **release posture** — what the release decision actually was, including that
  `research` releases nothing on any modality.

Every canary is invented: `.invalid` domains, 555-01xx numbers, `SYN-`
identifiers, obviously fictional names.

It runs in `legacy_test` model mode by default and downloads nothing. The
real-model path is opt-in:

```
PHI_RUN_REAL_MODEL_EVAL=1 py -3.11 -m pytest tests/test_real_model_evaluation.py -m real_models
PHI_MODEL_MODE=offline py -3.11 evaluations/run_evaluation.py
```

### What the harness found

Writing the evaluation surfaced two defects that the unit tests had not:

- **An IP address ending a sentence was never detected.** The boundary was
  `(?![\d.])`, so `accessed from 203.0.113.42.` — an entirely ordinary
  phrasing — failed the lookahead on the trailing period and the address
  survived redaction. Now `(?!\d)(?!\.\d)`, which still rejects longer dotted
  numbers.
- **The residual validator blocked on its own leftovers.** Masking placeholders
  changes sentence shape, which exposed ordinary capitalized words like
  "Portal" to the high-recall proper-noun heuristic. In strict mode the first
  pass already ran that heuristic and redacted everything it flagged, so a hit
  on the second pass is an artifact, not surviving PHI. That one source is now
  excluded from the residual scan; every evidence-based detector still counts.

### Claim boundary

Zero residual canaries is an acceptance condition for this suite. It is **not**
proof that an artifact, model, modality, or deployment has zero PHI leakage.
The corpus is small and synthetic, category recall is measured against expected
categories rather than a labelled gold standard, and thresholds remain
uncalibrated (`PHI_THRESHOLDS_CALIBRATED=0`). Uncertain findings block.

## Known gaps

These are deliberate and blocked, not overlooked:

1. **No defacing.** Cross-sectional head imaging permits facial reconstruction.
   DICOM and NIfTI carry `facial_reconstruction_not_mitigated` permanently
   until a defacing step exists.
2. **No validated writer for PDF, workbooks, or WSI.** Those modalities can be
   inventoried and scanned but never automatically released.
3. **PDF and workbook raster content is not OCR'd.** Any page with an image
   blocks with `pdf_raster_requires_ocr` rather than being scanned.
4. **CSV release posture is undecided.** Serialized-output validation passes;
   whether generalized quasi-identifiers clear the release bar is a
   re-identification-risk judgment that has not been made.
5. **Thresholds are uncalibrated.** The conservative default is zero, so every
   model candidate is redacted. No configured value is described as validated.
6. **NIfTI and WSI previews are blocked entirely**, which is a functional
   regression for any client that previewed them.
