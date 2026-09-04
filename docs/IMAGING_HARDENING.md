# Imaging hardening: DICOM, NIfTI, WSI, raster

## The standing blocker on volumetric imaging

Cross-sectional head imaging permits facial reconstruction, which Safe Harbor
treats as a comparable image. No defacing step exists in this pipeline, so
DICOM and NIfTI carry `facial_reconstruction_not_mitigated` as a permanent
reason code on their release decision. Improving the pixel scan does not lift
it, and nothing in this phase makes either modality automatically releasable.

## DICOM pixel path

### What was wrong

`redact_dicom_pixels` set `sanitized_bytes = file_bytes` and only replaced it
when at least one box was actually redacted. Two consequences:

- When OCR **detected** text but every box fell below the confidence threshold,
  the status was `completed_no_boxes_redacted` and the input bytes — still
  carrying that text — were returned under the key `sanitized_dicom_bytes`.
  That is now `privacy_requirements_not_met` with no bytes at all.
- Redacted pixels were written back with `dataset.PixelData = ...` while the
  original transfer syntax stood. Under a compressed syntax that produces a
  file no decoder can read.

### What it does now

1. Redacted regions are recorded as coordinates and fill values — never OCR
   text.
2. If text was detected and none was cleared, the scan blocks and returns no
   bytes.
3. When pixels are rewritten, a compressed transfer syntax is replaced with
   `ExplicitVRLittleEndian` first.
4. **Final-byte validation**: the bytes about to be handed out are re-read,
   the pixel array is decoded, the shape is compared to what was scanned, and
   every recorded region is confirmed to hold the fill value. Anything that
   fails blocks with `pixel_validation_status: verification_failed` and no
   bytes.

Validation is structural rather than a second OCR pass. The question it answers
is "did the redaction survive serialization", which is deterministic and does
not depend on an OCR engine's behaviour varying between runs. Semantic re-OCR
belongs in the opt-in evaluation suite.

`pixel_validation_status` is one of `not_applicable`, `not_attempted`,
`serialization_failed`, `verification_failed`, `verified`.

## Raster images (JPEG/PNG)

### What was wrong

`/anonymize_image` masked only OCR tokens whose **exact text** matched a spaCy
entity. If OCR returned nothing, or if spaCy did not classify a burned-in
identifier, the image came back byte-for-byte unmodified — streamed under an
`anonymized_` filename with a 200. If Presidio raised, the code caught it,
printed, and fell through to that weaker path.

### What it does now

`services/raster_redaction.py` takes the opposite position: on a medical raster
**any** burned-in text is presumptively identifying, so every region the engine
reports at or above the profile confidence threshold is filled. Then:

- Structural check: re-read the encoded output and confirm each filled region
  is uniformly black.
- Semantic check: re-run OCR over the encoded output. Any text still readable
  at or above the threshold blocks the release.
- Metadata: output is re-encoded from raw pixels as lossless PNG, so input
  EXIF, XMP, and PNG text chunks are never carried forward. This is verified on
  the re-read.

If OCR is unavailable, OCR fails, detected text was not cleared, or either
check fails, **no image bytes are returned** — the endpoint answers 422 with a
blocked summary carrying counts and reason codes only.

`/anonymize_image_presidio` still uses Presidio, but its output now goes
through the same semantic verification: Presidio redacts selectively by PHI
classification, so a misclassified identifier would otherwise survive.

The former `mask_phi_in_image_presidio` and `mask_phi_in_image_legacy` helpers
were removed rather than left unreferenced, so the fail-open implementation is
not available to a future caller.

| Limit | Value |
|---|---|
| `MAX_RASTER_BYTES` | 32 MiB |
| `MAX_RASTER_PIXELS` | 64 M |
| Accepted input formats | JPEG, PNG |
| Output format | PNG (lossless, so a fill cannot be perturbed by compression) |

## NIfTI

`anonymize_nifti_metadata` scrubbed the text header fields and then reported
`image_data_preserved: True` without ever serializing or re-reading. It now
serializes the scrubbed volume, reads it back, and confirms:

- every field in `TEXT_HEADER_FIELDS` is actually empty,
- no extensions remain (an extension can embed an entire DICOM header),
- the shape is unchanged.

Any failure sets `anonymization_status` to `privacy_requirements_not_met` and
lists the specific `validation_failures`. A profile that preserves extensions
fails this check by construction, because a surviving extension is unscanned
metadata.

`defacing_status` is reported as `not_implemented`.

## WSI

Unchanged and still plan-only: `scan_wsi_bytes` produces a redaction plan,
`wsi_rewrite_status` stays `not_supported_yet`, and no sanitized bytes exist on
any path. Tests now assert this explicitly so a future writer cannot be added
without the release decision being revisited.
