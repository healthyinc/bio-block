"""Downstream enforcement: nothing reaches an index or a preview unsanitized.

Two paths let raw uploaded content out of the pipeline without ever touching
the sanitizer:

* ``/store`` and ``/store_enhanced`` accepted client-supplied
  ``extracted_content`` and wrote it straight into the vector store, which
  ``/search`` reads back. A caller whose upload was blocked by ``/api/v1/ingest``
  could post the same text here and have it indexed.
* ``/simple_preview`` streamed the uploaded image bytes back unmodified - the
  image preview generator documents this as "bypass behavior" - and
  ``/preview_dicom`` rendered raw DICOM pixels to PNG with burned-in text
  intact.

This module is the gate for both. Indexed text is redacted and re-scanned
before it is written; preview bytes are produced only from verified sanitized
pixels. Every failure blocks, and every report carries categories, counts, and
status codes only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional, Tuple

from services.dicom_anonymization import (
    DicomAnonymizationError,
    anonymize_dicom_file_bytes,
)
from services.ocr_redaction import redact_dicom_pixels
from services.privacy_policy import PrivacyPolicyError, resolve_privacy_policy
from services.raster_redaction import (
    RasterRedactionError,
    redact_raster_bytes,
    verify_redacted_raster_bytes,
)
from services.text_anonymization import (
    MAX_TEXT_BYTES,
    TextAnonymizationError,
    anonymize_clinical_text,
)

STATUS_SANITIZED = "safe_harbor_technical_checks_passed"
STATUS_BLOCKED = "privacy_requirements_not_met"
STATUS_UNSCANNABLE = "unsupported_or_unscannable"
STATUS_EXPERT_DETERMINATION = "expert_determination_required"

REASON_RESIDUAL_PHI = "residual_phi_after_redaction"
REASON_METADATA_PHI = "phi_detected_in_metadata_value"
REASON_UNSUPPORTED_PREVIEW = "preview_modality_not_sanitizable"
REASON_DICOM_METADATA_FAILED = "dicom_metadata_scrub_failed"
REASON_DICOM_PIXELS_UNVERIFIED = "dicom_pixel_redaction_unverified"
REASON_PREVIEW_RENDER_FAILED = "preview_render_failed"
REASON_FACIAL_RECONSTRUCTION = "facial_reconstruction_not_mitigated"
REASON_NO_VALIDATED_WRITER = "validated_writer_unavailable"

# Structural fields the platform needs verbatim to function: wallet addresses,
# content identifiers, and chunk bookkeeping. They are not free text and are
# excluded from the metadata scan rather than being rewritten.
STRUCTURAL_METADATA_KEYS = frozenset(
    {
        "cid",
        "owner_address",
        "file_type",
        "chunk_index",
        "total_chunks",
        "parent_doc_id",
        "doc_id",
        "document_id",
        "created_at",
        "updated_at",
    }
)

# Preview modalities that are sanitizable at all. NIfTI and WSI are absent by
# intent, see _blocked_preview_reason.
RASTER_PREVIEW_TYPES = frozenset({"image/jpeg", "image/jpg", "image/png"})
RASTER_PREVIEW_EXTENSIONS = frozenset({"jpg", "jpeg", "png"})
DICOM_PREVIEW_EXTENSIONS = frozenset({"dcm", "dicom"})
NIFTI_PREVIEW_EXTENSIONS = frozenset({"nii", "gz"})
WSI_PREVIEW_EXTENSIONS = frozenset({"svs", "ndpi", "tif", "tiff", "scn", "mrxs"})


class ReleaseGateError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class IndexableText:
    """Redacted text cleared for indexing, or a block."""

    status: str
    fields: Mapping[str, str] = field(default_factory=dict)
    detected_entities: Mapping[str, int] = field(default_factory=dict)
    residual_phi_categories: Mapping[str, int] = field(default_factory=dict)
    blocked_fields: Tuple[str, ...] = ()
    reason_codes: Tuple[str, ...] = ()

    @property
    def cleared(self) -> bool:
        return self.status == STATUS_SANITIZED

    def safe_summary(self) -> Dict[str, Any]:
        """Reportable fields only. Never includes the redacted text itself."""
        return {
            "sanitization_status": self.status,
            "detected_entities": dict(self.detected_entities),
            "residual_phi_categories": dict(self.residual_phi_categories),
            "blocked_fields": list(self.blocked_fields),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class PreviewBytes:
    """Verified preview bytes, or a block. ``content`` is set only on success."""

    status: str
    content: Optional[bytes] = field(default=None, repr=False)
    media_type: Optional[str] = None
    reason_codes: Tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def released(self) -> bool:
        return self.status == STATUS_SANITIZED and bool(self.content)

    def safe_summary(self) -> Dict[str, Any]:
        return {
            "preview_status": self.status,
            "reason_codes": list(self.reason_codes),
            **dict(self.detail),
        }


def _merge_counts(target: Dict[str, int], addition: Mapping[str, int]) -> None:
    for key, value in addition.items():
        target[key] = target.get(key, 0) + int(value)


def _extension(filename: Optional[str]) -> str:
    name = (filename or "").strip().lower()
    if name.endswith(".nii.gz"):
        return "nii.gz"
    return name.rsplit(".", 1)[-1] if "." in name else ""


# ---------------------------------------------------------------------------
# Indexing gate
# ---------------------------------------------------------------------------


def sanitize_for_index(
    text_fields: Mapping[str, Optional[str]],
    metadata: Optional[Mapping[str, Any]] = None,
    profile: str = "strict",
) -> IndexableText:
    """Redact free-text fields and scan metadata before anything is indexed.

    ``text_fields`` are redacted and re-scanned; anything surviving blocks.
    ``metadata`` values are scanned but never rewritten - the platform filters
    on them, so a silent rewrite would corrupt them. PHI found there blocks.
    """
    try:
        resolved = resolve_privacy_policy(profile)
    except PrivacyPolicyError as exc:
        raise ReleaseGateError(str(exc), status_code=400) from exc

    if not resolved.automatic_release_allowed:
        # Research content is never indexable, so nothing is written.
        return IndexableText(
            status=STATUS_EXPERT_DETERMINATION,
            reason_codes=(STATUS_EXPERT_DETERMINATION,),
        )

    detected: Dict[str, int] = {}
    residual: Dict[str, int] = {}
    redacted: Dict[str, str] = {}
    blocked: List[str] = []
    reasons: List[str] = []

    for name, value in text_fields.items():
        raw = str(value or "")
        if not raw.strip():
            redacted[name] = raw
            continue
        if len(raw.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ReleaseGateError(
                f"{name} exceeds the {MAX_TEXT_BYTES} byte limit",
                status_code=413,
            )
        try:
            result = anonymize_clinical_text(raw, profile=resolved.config_profile)
            field_residual = result["residual_phi_categories"]
        except TextAnonymizationError as exc:
            raise ReleaseGateError(exc.detail, status_code=exc.status_code) from exc

        _merge_counts(detected, result["detected_entities"])
        redacted[name] = result["anonymized_text"]
        if field_residual:
            _merge_counts(residual, field_residual)
            blocked.append(name)

    for key, value in (metadata or {}).items():
        if key in STRUCTURAL_METADATA_KEYS or not isinstance(value, str):
            continue
        if not value.strip():
            continue
        try:
            result = anonymize_clinical_text(value, profile=resolved.config_profile)
        except TextAnonymizationError as exc:
            raise ReleaseGateError(exc.detail, status_code=exc.status_code) from exc
        if result["detected_entities"]:
            _merge_counts(detected, result["detected_entities"])
            blocked.append(f"metadata.{key}")
            reasons.append(REASON_METADATA_PHI)

    if residual:
        reasons.append(REASON_RESIDUAL_PHI)

    if blocked:
        return IndexableText(
            status=STATUS_BLOCKED,
            detected_entities=detected,
            residual_phi_categories=residual,
            blocked_fields=tuple(sorted(set(blocked))),
            reason_codes=tuple(sorted(set(reasons))),
        )

    return IndexableText(
        status=STATUS_SANITIZED,
        fields=redacted,
        detected_entities=detected,
        reason_codes=("residual_phi_not_detected",),
    )


# ---------------------------------------------------------------------------
# Preview gate
# ---------------------------------------------------------------------------


def _blocked_preview_reason(extension: str, content_type: str) -> Optional[str]:
    """Reason a modality cannot have a sanitized preview at all."""
    if extension in NIFTI_PREVIEW_EXTENSIONS or extension == "nii.gz":
        # A slice through a head volume is itself a comparable image, and no
        # defacing step exists.
        return REASON_FACIAL_RECONSTRUCTION
    if extension in WSI_PREVIEW_EXTENSIONS:
        # The slide label is exactly where identifiers live, and there is no
        # validated writer for the sanitized slide.
        return REASON_NO_VALIDATED_WRITER
    return None


def _blocked_preview(status: str, *reasons: str, **detail: Any) -> PreviewBytes:
    return PreviewBytes(status=status, reason_codes=tuple(reasons), detail=detail)


def sanitized_preview(
    file_contents: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    profile: str = "strict",
) -> PreviewBytes:
    """Produce preview bytes only from verified sanitized pixels."""
    try:
        resolved = resolve_privacy_policy(profile)
    except PrivacyPolicyError as exc:
        raise ReleaseGateError(str(exc), status_code=400) from exc
    if not resolved.automatic_release_allowed:
        return _blocked_preview(
            STATUS_EXPERT_DETERMINATION, STATUS_EXPERT_DETERMINATION
        )

    if not file_contents:
        raise ReleaseGateError("Uploaded file is empty", status_code=400)

    extension = _extension(filename)
    mime = (content_type or "").strip().lower()

    unsupported = _blocked_preview_reason(extension, mime)
    if unsupported:
        return _blocked_preview(STATUS_UNSCANNABLE, unsupported)

    if mime in RASTER_PREVIEW_TYPES or extension in RASTER_PREVIEW_EXTENSIONS:
        return _raster_preview(file_contents, resolved.config_profile)

    if extension in DICOM_PREVIEW_EXTENSIONS or "dicom" in mime:
        return _dicom_preview(file_contents, resolved.config_profile)

    return _blocked_preview(STATUS_UNSCANNABLE, REASON_UNSUPPORTED_PREVIEW)


def _raster_preview(file_contents: bytes, profile: str) -> PreviewBytes:
    try:
        outcome = redact_raster_bytes(file_contents, profile=profile)
    except RasterRedactionError as exc:
        raise ReleaseGateError(exc.detail, status_code=exc.status_code) from exc

    if not outcome.released:
        return _blocked_preview(
            outcome.status,
            *outcome.reason_codes,
            boxes_detected=outcome.boxes_detected,
            boxes_redacted=outcome.boxes_redacted,
            residual_text_boxes=outcome.residual_text_boxes,
        )
    return PreviewBytes(
        status=STATUS_SANITIZED,
        content=outcome.released_bytes,
        media_type=outcome.media_type,
        reason_codes=("residual_phi_not_detected",),
        detail={
            "boxes_redacted": outcome.boxes_redacted,
            "validation_status": outcome.validation_status,
        },
    )


def _dicom_preview(file_contents: bytes, profile: str) -> PreviewBytes:
    """Scrub metadata, redact pixels, verify, then render. Never raw pixels."""
    try:
        metadata_result = anonymize_dicom_file_bytes(file_contents, profile=profile)
    except DicomAnonymizationError as exc:
        return _blocked_preview(
            STATUS_UNSCANNABLE, REASON_DICOM_METADATA_FAILED, detail_code=exc.detail
        )

    pixel_result = redact_dicom_pixels(
        metadata_result["anonymized_dicom_bytes"], profile=profile
    )
    sanitized = pixel_result.get("sanitized_dicom_bytes")
    if not sanitized or pixel_result.get("pixel_validation_status") != "verified":
        # No verified bytes means no preview. Rendering the input here is
        # exactly the bypass this gate exists to close.
        return _blocked_preview(
            STATUS_BLOCKED,
            REASON_DICOM_PIXELS_UNVERIFIED,
            pixel_redaction_status=pixel_result.get("pixel_redaction_status"),
            pixel_validation_status=pixel_result.get("pixel_validation_status"),
        )

    try:
        png_bytes = _render_dicom_png(sanitized)
    except Exception:
        return _blocked_preview(STATUS_UNSCANNABLE, REASON_PREVIEW_RENDER_FAILED)

    # The rendered PNG goes through the same residual text scan a raster
    # upload would, so anything the pixel pass missed still blocks.
    try:
        verified = verify_redacted_raster_bytes(png_bytes, profile=profile)
    except RasterRedactionError as exc:
        raise ReleaseGateError(exc.detail, status_code=exc.status_code) from exc
    if not verified.released:
        return _blocked_preview(
            verified.status,
            *verified.reason_codes,
            residual_text_boxes=verified.residual_text_boxes,
        )

    return PreviewBytes(
        status=STATUS_SANITIZED,
        content=verified.released_bytes,
        media_type=verified.media_type,
        reason_codes=("residual_phi_not_detected",),
        detail={
            "boxes_redacted": pixel_result.get("boxes_redacted", 0),
            "pixel_validation_status": pixel_result.get("pixel_validation_status"),
        },
    )


def _render_dicom_png(sanitized_dicom: bytes) -> bytes:
    import numpy as np
    import pydicom
    from PIL import Image

    dataset = pydicom.dcmread(BytesIO(sanitized_dicom))
    pixels = np.asarray(dataset.pixel_array)
    if pixels.ndim == 3 and int(getattr(dataset, "SamplesPerPixel", 1) or 1) == 1:
        pixels = pixels[0]

    minimum = float(pixels.min())
    maximum = float(pixels.max())
    if maximum == minimum:
        normalized = np.zeros(pixels.shape, dtype=np.uint8)
    else:
        normalized = (((pixels - minimum) / (maximum - minimum)) * 255).astype(
            np.uint8
        )

    mode = "RGB" if normalized.ndim == 3 else "L"
    buffer = BytesIO()
    Image.fromarray(normalized, mode=mode).convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()
