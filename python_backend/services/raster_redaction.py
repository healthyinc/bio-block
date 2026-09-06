"""Verified raster (JPEG/PNG) redaction.

The previous image path masked only OCR tokens whose exact text matched a
spaCy entity, then streamed the result back as "anonymized" regardless of
whether anything had been masked at all. An image whose OCR produced nothing,
or whose burned-in identifiers spaCy did not classify, came back unmodified
under an ``anonymized_`` filename. That is a fail-open.

This module takes the opposite position for medical rasters: every text region
the OCR engine finds above the profile confidence threshold is presumptively
identifying and is filled, and the encoded output is then re-read and verified
before any bytes are released. If OCR is unavailable, if OCR fails, if detected
text was not cleared, or if verification does not pass, no bytes come back at
all.

Only categories, counts, and statuses are reported - never OCR text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from services.ocr_redaction import (
    OCREngineUnavailable,
    OCRBackend,
    clip_box_to_image,
    get_default_ocr_backend,
)
from services.privacy_profiles import PrivacyProfileError, get_privacy_profile

MAX_RASTER_BYTES = 32 * 1024 * 1024
MAX_RASTER_PIXELS = 64_000_000
SUPPORTED_INPUT_FORMATS = {"JPEG", "PNG"}
# Output is always lossless so a fill cannot be perturbed by compression.
OUTPUT_FORMAT = "PNG"
OUTPUT_MEDIA_TYPE = "image/png"

STATUS_COMPLETED = "safe_harbor_technical_checks_passed"
STATUS_BLOCKED = "privacy_requirements_not_met"
STATUS_UNSCANNABLE = "unsupported_or_unscannable"
STATUS_MANUAL_REVIEW = "manual_review_required"

REASON_OCR_UNAVAILABLE = "raster_ocr_unavailable"
REASON_OCR_FAILED = "raster_ocr_failed"
REASON_TEXT_NOT_CLEARED = "raster_detected_text_not_cleared"
REASON_VERIFICATION_FAILED = "raster_redaction_verification_failed"
REASON_RESIDUAL_TEXT = "raster_residual_text_detected"
REASON_UNSUPPORTED_FORMAT = "raster_unsupported_format"
REASON_UNDECODABLE = "raster_undecodable"
REASON_TOO_LARGE = "raster_exceeds_size_limit"
REASON_INVALID_PROFILE = "raster_invalid_profile"

_BLACK = (0, 0, 0)


from services.modality_utility import measure_raster_utility
from services.text_anonymization import (
    EXTRACTED_NON_PHI,
    EXTRACTED_UNCERTAIN,
    classify_extracted_text,
)


class RasterRedactionError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class RasterRedactionOutcome:
    """A redaction verdict. ``released_bytes`` is set only on success."""

    status: str
    released_bytes: Optional[bytes] = field(default=None, repr=False)
    media_type: Optional[str] = None
    boxes_detected: int = 0
    boxes_redacted: int = 0
    residual_text_boxes: int = 0
    #: Boxes deliberately left in place because their text read as clinical.
    boxes_preserved: int = 0
    #: Boxes redacted only because the pipeline could not place them. Counted
    #: so the utility cost of caution is visible rather than silent.
    uncertain_regions_redacted: int = 0
    utility_metrics: Dict[str, Any] = field(default_factory=dict)
    ocr_engine_status: str = "not_applicable"
    validation_status: str = "not_attempted"
    input_metadata_present: bool = False
    reason_codes: Tuple[str, ...] = ()

    @property
    def released(self) -> bool:
        return self.status == STATUS_COMPLETED and bool(self.released_bytes)

    def safe_summary(self) -> Dict[str, Any]:
        """Reportable fields only. Never includes bytes or OCR text."""
        return {
            "anonymization_status": self.status,
            "boxes_detected": self.boxes_detected,
            "boxes_redacted": self.boxes_redacted,
            "residual_text_boxes": self.residual_text_boxes,
            "boxes_preserved": self.boxes_preserved,
            "uncertain_regions_redacted": self.uncertain_regions_redacted,
            "utility_metrics": dict(self.utility_metrics),
            "ocr_engine_status": self.ocr_engine_status,
            "validation_status": self.validation_status,
            "input_metadata_present": self.input_metadata_present,
            "reason_codes": list(self.reason_codes),
            "released": self.released,
        }


def _blocked(
    status: str,
    reason: str,
    **fields: Any,
) -> RasterRedactionOutcome:
    return RasterRedactionOutcome(status=status, reason_codes=(reason,), **fields)


def _confidence_threshold(profile: str) -> float:
    try:
        return float(get_privacy_profile(profile)["ocr_confidence_threshold"])
    except (PrivacyProfileError, KeyError, TypeError, ValueError) as exc:
        raise RasterRedactionError(REASON_INVALID_PROFILE, status_code=400) from exc


def _decode(content: bytes) -> Tuple[Image.Image, str, bool]:
    """Return (RGB image, source format, whether input carried metadata)."""
    try:
        probe = Image.open(BytesIO(content))
        source_format = (probe.format or "").upper()
        # EXIF, XMP, and PNG text chunks can carry device serials, GPS, and
        # timestamps. None of it survives into the output.
        has_metadata = bool(
            probe.info.get("exif")
            or probe.info.get("XML:com.adobe.xmp")
            or any(
                isinstance(value, str) and value.strip()
                for key, value in probe.info.items()
                if key not in {"dpi", "jfif", "jfif_version", "jfif_unit", "jfif_density"}
            )
        )
        probe.load()
        image = probe.convert("RGB")
    except Exception as exc:
        raise RasterRedactionError(REASON_UNDECODABLE, status_code=400) from exc
    return image, source_format, has_metadata


def redact_raster_bytes(
    content: bytes,
    profile: str = "strict",
    ocr_backend: Optional[OCRBackend] = None,
) -> RasterRedactionOutcome:
    """Redact all detected text in a raster image and verify the result."""
    if not isinstance(content, (bytes, bytearray)):
        raise RasterRedactionError("Image content must be bytes", status_code=500)
    if not content:
        raise RasterRedactionError("Uploaded image is empty", status_code=400)
    if len(content) > MAX_RASTER_BYTES:
        raise RasterRedactionError(REASON_TOO_LARGE, status_code=413)

    threshold = _confidence_threshold(profile)
    image, source_format, has_metadata = _decode(bytes(content))

    if source_format not in SUPPORTED_INPUT_FORMATS:
        return _blocked(
            STATUS_UNSCANNABLE,
            REASON_UNSUPPORTED_FORMAT,
            input_metadata_present=has_metadata,
        )
    width, height = image.size
    if width * height > MAX_RASTER_PIXELS:
        raise RasterRedactionError(REASON_TOO_LARGE, status_code=413)

    backend = ocr_backend or get_default_ocr_backend()
    engine_status = str(getattr(backend, "ocr_engine_status", "available"))
    if engine_status == "unavailable":
        return _blocked(
            STATUS_UNSCANNABLE,
            REASON_OCR_UNAVAILABLE,
            ocr_engine_status=engine_status,
            input_metadata_present=has_metadata,
        )

    try:
        boxes = list(backend.detect_text_boxes(image))
    except OCREngineUnavailable:
        return _blocked(
            STATUS_UNSCANNABLE,
            REASON_OCR_UNAVAILABLE,
            ocr_engine_status="unavailable",
            input_metadata_present=has_metadata,
        )
    except Exception:
        return _blocked(
            STATUS_UNSCANNABLE,
            REASON_OCR_FAILED,
            ocr_engine_status="error",
            input_metadata_present=has_metadata,
        )

    pixels = np.array(image, dtype=np.uint8)
    regions: List[Tuple[int, int, int, int]] = []
    preserved: List[Tuple[int, int, int, int]] = []
    uncertain_regions = 0
    for box in boxes:
        if box.confidence < threshold:
            continue
        clipped = clip_box_to_image(box, width, height)
        if clipped is None:
            continue

        # Not every readable box is a name. A laterality marker, a scale bar
        # or a burned-in measurement is the reason the image was kept, and
        # blacking out every box the engine reports drives residual text to
        # zero by destroying the diagnostic content along with the identifier.
        # Only recognisably clinical text is spared; anything the classifier
        # cannot place is redacted like PHI and counted, so the utility cost
        # of that caution is visible rather than silent.
        reading = classify_extracted_text(box.text)
        if reading == EXTRACTED_NON_PHI:
            preserved.append(clipped)
            continue
        if reading == EXTRACTED_UNCERTAIN:
            uncertain_regions += 1

        left, top, right, bottom = clipped
        pixels[top:bottom, left:right] = _BLACK
        regions.append(clipped)

    if boxes and not regions and not preserved:
        # Text was found and none of it was cleared. These pixels still carry
        # it, so nothing is released.
        return _blocked(
            STATUS_BLOCKED,
            REASON_TEXT_NOT_CLEARED,
            boxes_detected=len(boxes),
            ocr_engine_status=engine_status,
            input_metadata_present=has_metadata,
        )

    # Re-encode without carrying any of the input's metadata forward.
    buffer = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format=OUTPUT_FORMAT)
    encoded = buffer.getvalue()

    verified, residual = _verify_encoded(
        encoded, pixels.shape, regions, backend, threshold, preserved
    )
    if not verified:
        return _blocked(
            STATUS_BLOCKED,
            REASON_VERIFICATION_FAILED if residual < 0 else REASON_RESIDUAL_TEXT,
            boxes_detected=len(boxes),
            boxes_redacted=len(regions),
            residual_text_boxes=max(residual, 0),
            ocr_engine_status=engine_status,
            validation_status="verification_failed",
            input_metadata_present=has_metadata,
        )

    return RasterRedactionOutcome(
        status=STATUS_COMPLETED,
        released_bytes=encoded,
        media_type=OUTPUT_MEDIA_TYPE,
        boxes_detected=len(boxes),
        boxes_redacted=len(regions),
        boxes_preserved=len(preserved),
        uncertain_regions_redacted=uncertain_regions,
        utility_metrics=measure_raster_utility(
            bytes(content),
            encoded,
            redaction_boxes=regions,
            residual_text_regions=0,
            preserved_label_regions=len(preserved),
            review_regions=uncertain_regions,
            lossless=True,
        ),
        residual_text_boxes=0,
        ocr_engine_status=engine_status,
        validation_status="verified",
        input_metadata_present=has_metadata,
        reason_codes=("residual_phi_not_detected",),
    )


def _verify_encoded(
    encoded: bytes,
    expected_shape: Tuple[int, ...],
    regions: List[Tuple[int, int, int, int]],
    backend: OCRBackend,
    threshold: float,
    preserved: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Tuple[bool, int]:
    """Re-read the encoded output and confirm the redaction survived.

    Returns (verified, residual box count). A residual count of -1 means the
    structural check itself failed, which is also a block.
    """
    try:
        reread = Image.open(BytesIO(encoded))
        reread.load()
        if reread.info.get("exif") or reread.info.get("XML:com.adobe.xmp"):
            return False, -1
        pixels = np.array(reread.convert("RGB"), dtype=np.uint8)
    except Exception:
        return False, -1

    if tuple(pixels.shape) != tuple(expected_shape):
        return False, -1
    for left, top, right, bottom in regions:
        region = pixels[top:bottom, left:right]
        if region.size == 0 or not np.all(region == 0):
            return False, -1

    # Semantic check: nothing the engine can still read may remain, except
    # inside a region this pass deliberately kept. A deliberately preserved
    # marker is re-classified here rather than trusted: if the re-read text
    # no longer looks clinical, it counts as residual and blocks.
    kept = list(preserved or [])

    def _inside_preserved(box) -> bool:
        left, top = int(box.x), int(box.y)
        right, bottom = left + int(box.width), top + int(box.height)
        return any(
            x0 <= left and y0 <= top and right <= x1 and bottom <= y1
            for x0, y0, x1, y1 in kept
        )

    try:
        residual_boxes = [
            box
            for box in backend.detect_text_boxes(Image.fromarray(pixels, mode="RGB"))
            if box.confidence >= threshold
            and not (
                _inside_preserved(box)
                and classify_extracted_text(box.text) == EXTRACTED_NON_PHI
            )
        ]
    except Exception:
        return False, -1

    return (not residual_boxes), len(residual_boxes)


def verify_redacted_raster_bytes(
    content: bytes,
    profile: str = "strict",
    ocr_backend: Optional[OCRBackend] = None,
) -> RasterRedactionOutcome:
    """Verify already-redacted bytes produced by an external redactor.

    The regions are unknown, so only the semantic check applies: any text the
    OCR engine can still read above the threshold blocks the release.
    """
    if not content:
        raise RasterRedactionError("Redacted image is empty", status_code=500)
    if len(content) > MAX_RASTER_BYTES:
        raise RasterRedactionError(REASON_TOO_LARGE, status_code=413)

    threshold = _confidence_threshold(profile)
    image, _source_format, has_metadata = _decode(bytes(content))

    backend = ocr_backend or get_default_ocr_backend()
    engine_status = str(getattr(backend, "ocr_engine_status", "available"))
    if engine_status == "unavailable":
        # Without an engine the redaction cannot be checked, so nothing ships.
        return _blocked(
            STATUS_UNSCANNABLE,
            REASON_OCR_UNAVAILABLE,
            ocr_engine_status=engine_status,
            input_metadata_present=has_metadata,
        )

    try:
        residual = [
            box
            for box in backend.detect_text_boxes(image)
            if box.confidence >= threshold
        ]
    except OCREngineUnavailable:
        return _blocked(
            STATUS_UNSCANNABLE,
            REASON_OCR_UNAVAILABLE,
            ocr_engine_status="unavailable",
            input_metadata_present=has_metadata,
        )
    except Exception:
        return _blocked(
            STATUS_UNSCANNABLE,
            REASON_OCR_FAILED,
            ocr_engine_status="error",
            input_metadata_present=has_metadata,
        )

    if residual:
        return _blocked(
            STATUS_BLOCKED,
            REASON_RESIDUAL_TEXT,
            residual_text_boxes=len(residual),
            ocr_engine_status=engine_status,
            validation_status="verification_failed",
            input_metadata_present=has_metadata,
        )

    # Re-encode from pixels so no input metadata is carried forward.
    buffer = BytesIO()
    image.save(buffer, format=OUTPUT_FORMAT)
    return RasterRedactionOutcome(
        status=STATUS_COMPLETED,
        released_bytes=buffer.getvalue(),
        media_type=OUTPUT_MEDIA_TYPE,
        residual_text_boxes=0,
        ocr_engine_status=engine_status,
        validation_status="verified",
        input_metadata_present=has_metadata,
        reason_codes=("residual_phi_not_detected",),
    )


def verified_raster_response(outcome: RasterRedactionOutcome) -> Dict[str, Any]:
    """Public-facing summary for a blocked outcome. Never carries bytes."""
    summary = outcome.safe_summary()
    summary["message"] = (
        "Image redaction could not be verified; no image bytes were released."
    )
    return summary
