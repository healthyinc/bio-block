from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from services.privacy_profiles import PrivacyProfileError, get_privacy_profile

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    pydicom = None
    InvalidDicomError = Exception


DEFAULT_MIN_CONFIDENCE = 0.5


class OCREngineUnavailable(RuntimeError):
    pass


class OCRBackend(Protocol):
    ocr_engine_status: str

    def detect_text_boxes(self, image: Image.Image) -> Sequence["OCRBox"]:
        ...


@dataclass(frozen=True)
class OCRBox:
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


@dataclass
class ImageRedactionResult:
    image: Optional[Image.Image]
    boxes_detected: int
    boxes_redacted: int
    redaction_status: str
    ocr_engine_status: Optional[str] = None

    def safe_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "boxes_detected": self.boxes_detected,
            "boxes_redacted": self.boxes_redacted,
            "redaction_status": self.redaction_status,
        }
        if self.ocr_engine_status is not None:
            summary["ocr_engine_status"] = self.ocr_engine_status
        return summary


class UnavailableOCRBackend:
    ocr_engine_status = "unavailable"

    def detect_text_boxes(self, image: Image.Image) -> Sequence[OCRBox]:
        raise OCREngineUnavailable("OCR engine is unavailable")


class PytesseractOCRBackend:
    def __init__(self, tesseract_cmd: Optional[str] = None):
        self._pytesseract = None
        self._output = None
        self.ocr_engine_status = "unavailable"

        try:
            import pytesseract
            from pytesseract import Output
        except ImportError:
            return

        resolved_cmd = _resolve_tesseract_cmd(tesseract_cmd)
        if not resolved_cmd:
            return

        pytesseract.pytesseract.tesseract_cmd = resolved_cmd
        self._pytesseract = pytesseract
        self._output = Output
        self.ocr_engine_status = "available"

    def detect_text_boxes(self, image: Image.Image) -> Sequence[OCRBox]:
        if self.ocr_engine_status != "available" or self._pytesseract is None:
            raise OCREngineUnavailable("OCR engine is unavailable")

        pil_image = _ensure_pil_image(image)
        ocr_data = self._pytesseract.image_to_data(
            pil_image,
            output_type=self._output.DICT,
        )

        detections: List[OCRBox] = []
        for index, text in enumerate(ocr_data.get("text", [])):
            cleaned_text = str(text).strip()
            if not cleaned_text:
                continue

            confidence = _parse_confidence(ocr_data.get("conf", [])[index])
            if confidence < 0:
                continue

            detections.append(
                OCRBox(
                    text=cleaned_text,
                    confidence=confidence,
                    x=int(ocr_data.get("left", [])[index]),
                    y=int(ocr_data.get("top", [])[index]),
                    width=int(ocr_data.get("width", [])[index]),
                    height=int(ocr_data.get("height", [])[index]),
                )
            )
        return detections


def get_default_ocr_backend() -> OCRBackend:
    backend = PytesseractOCRBackend()
    if backend.ocr_engine_status == "available":
        return backend
    return UnavailableOCRBackend()


def redact_image_with_backend(
    image: Any,
    ocr_backend: Optional[OCRBackend] = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> ImageRedactionResult:
    backend = ocr_backend or get_default_ocr_backend()
    engine_status = _backend_status(backend)

    if engine_status == "unavailable":
        return ImageRedactionResult(
            image=None,
            boxes_detected=0,
            boxes_redacted=0,
            redaction_status="skipped_ocr_unavailable",
            ocr_engine_status=engine_status,
        )

    try:
        boxes = backend.detect_text_boxes(_ensure_pil_image(image))
    except OCREngineUnavailable:
        return ImageRedactionResult(
            image=None,
            boxes_detected=0,
            boxes_redacted=0,
            redaction_status="skipped_ocr_unavailable",
            ocr_engine_status="unavailable",
        )
    except Exception:
        return ImageRedactionResult(
            image=None,
            boxes_detected=0,
            boxes_redacted=0,
            redaction_status="ocr_failed",
            ocr_engine_status="error",
        )

    result = redact_image_regions(
        image=image,
        boxes=boxes,
        min_confidence=min_confidence,
    )
    result.ocr_engine_status = engine_status
    return result


def redact_image_regions(
    image: Any,
    boxes: Iterable[OCRBox],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    fill: Optional[Any] = None,
) -> ImageRedactionResult:
    pil_image = _ensure_pil_image(image)
    redacted = pil_image.copy()
    draw = ImageDraw.Draw(redacted)
    image_width, image_height = redacted.size
    detections = list(boxes)
    boxes_redacted = 0
    mask_fill = fill if fill is not None else _black_fill_for_mode(redacted.mode)

    for box in detections:
        if box.confidence < min_confidence:
            continue

        clipped = clip_box_to_image(box, image_width, image_height)
        if clipped is None:
            continue

        left, top, right, bottom = clipped
        draw.rectangle((left, top, right - 1, bottom - 1), fill=mask_fill)
        boxes_redacted += 1

    if boxes_redacted:
        status = "completed"
    elif detections:
        status = "completed_no_boxes_redacted"
    else:
        status = "completed_no_boxes"

    return ImageRedactionResult(
        image=redacted,
        boxes_detected=len(detections),
        boxes_redacted=boxes_redacted,
        redaction_status=status,
    )


def clip_box_to_image(
    box: OCRBox,
    image_width: int,
    image_height: int,
) -> Optional[Tuple[int, int, int, int]]:
    left = max(0, int(box.x))
    top = max(0, int(box.y))
    right = min(image_width, int(box.x) + max(0, int(box.width)))
    bottom = min(image_height, int(box.y) + max(0, int(box.height)))

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def redact_dicom_pixels(
    file_bytes: bytes,
    profile: str = "strict",
    ocr_backend: Optional[OCRBackend] = None,
    min_confidence: Optional[float] = None,
    include_sanitized_bytes: bool = True,
) -> Dict[str, Any]:
    if pydicom is None:
        return _dicom_result(
            pixel_redaction_status="dependency_unavailable",
            ocr_engine_status="unavailable",
            include_sanitized_bytes=include_sanitized_bytes,
        )
    if not file_bytes:
        return _dicom_result(
            pixel_redaction_status="empty_file",
            ocr_engine_status="not_applicable",
            include_sanitized_bytes=include_sanitized_bytes,
        )

    try:
        profile_settings = get_privacy_profile(profile)
    except PrivacyProfileError:
        return _dicom_result(
            pixel_redaction_status="invalid_profile",
            ocr_engine_status="not_applicable",
            include_sanitized_bytes=include_sanitized_bytes,
        )

    effective_min_confidence = (
        float(min_confidence)
        if min_confidence is not None
        else float(profile_settings["ocr_confidence_threshold"])
    )

    backend = ocr_backend or get_default_ocr_backend()
    engine_status = _backend_status(backend)
    if engine_status == "unavailable":
        return _dicom_result(
            pixel_redaction_status="skipped_ocr_unavailable",
            ocr_engine_status=engine_status,
            include_sanitized_bytes=include_sanitized_bytes,
        )

    try:
        dataset = pydicom.dcmread(BytesIO(file_bytes), force=False)
    except (InvalidDicomError, Exception):
        return _dicom_result(
            pixel_redaction_status="invalid_dicom",
            ocr_engine_status=engine_status,
            include_sanitized_bytes=include_sanitized_bytes,
        )

    if "PixelData" not in dataset:
        return _dicom_result(
            pixel_redaction_status="no_pixel_data",
            ocr_engine_status=engine_status,
            include_sanitized_bytes=include_sanitized_bytes,
        )

    try:
        pixel_array = np.array(dataset.pixel_array, copy=True)
    except Exception:
        return _dicom_result(
            pixel_redaction_status="unsupported_pixel_data",
            ocr_engine_status=engine_status,
            include_sanitized_bytes=include_sanitized_bytes,
        )

    frames = _iter_grayscale_frames(pixel_array, dataset)
    if frames is None:
        return _dicom_result(
            pixel_redaction_status="unsupported_pixel_format",
            ocr_engine_status=engine_status,
            include_sanitized_bytes=include_sanitized_bytes,
        )

    boxes_detected = 0
    boxes_redacted = 0
    frames_processed = 0

    try:
        for frame_index, frame in frames:
            ocr_image = _frame_to_ocr_image(frame)
            boxes = list(backend.detect_text_boxes(ocr_image))
            boxes_detected += len(boxes)

            image_width, image_height = ocr_image.size
            for box in boxes:
                if box.confidence < effective_min_confidence:
                    continue

                clipped = clip_box_to_image(box, image_width, image_height)
                if clipped is None:
                    continue

                left, top, right, bottom = clipped
                frame[top:bottom, left:right] = _redaction_pixel_value(frame)
                boxes_redacted += 1

            frames_processed += 1
    except OCREngineUnavailable:
        return _dicom_result(
            pixel_redaction_status="skipped_ocr_unavailable",
            ocr_engine_status="unavailable",
            include_sanitized_bytes=include_sanitized_bytes,
        )
    except Exception:
        return _dicom_result(
            pixel_redaction_status="ocr_failed",
            ocr_engine_status="error",
            frames_processed=frames_processed,
            include_sanitized_bytes=include_sanitized_bytes,
        )

    sanitized_bytes = file_bytes
    if boxes_redacted:
        dataset.PixelData = pixel_array.tobytes()
        sanitized_bytes = _dataset_to_bytes(dataset)

    if boxes_redacted:
        status = "completed"
    elif boxes_detected:
        status = "completed_no_boxes_redacted"
    else:
        status = "completed_no_text_detected"

    return _dicom_result(
        pixel_redaction_status=status,
        ocr_boxes_detected=boxes_detected,
        boxes_redacted=boxes_redacted,
        frames_processed=frames_processed,
        scanned_regions=frames_processed,
        ocr_engine_status=engine_status,
        sanitized_bytes=sanitized_bytes,
        include_sanitized_bytes=include_sanitized_bytes,
        ocr_confidence_threshold=effective_min_confidence,
    )


def safe_ocr_response(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"sanitized_dicom_bytes", "sanitized_bytes"}
    }


def _resolve_tesseract_cmd(tesseract_cmd: Optional[str]) -> Optional[str]:
    candidates = [
        tesseract_cmd,
        os.getenv("TESSERACT_CMD"),
        shutil.which("tesseract"),
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isfile(candidate):
            return candidate
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _parse_confidence(raw_confidence: Any) -> float:
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        return -1.0

    if confidence > 1.0:
        confidence = confidence / 100.0
    return confidence


def _ensure_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.copy()

    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array)
    if array.ndim == 3:
        return Image.fromarray(array)

    raise ValueError("OCR redaction requires a 2D grayscale or 3D image array")


def _black_fill_for_mode(mode: str) -> Any:
    if mode in {"1", "L", "I", "I;16", "F"}:
        return 0
    if mode == "RGBA":
        return (0, 0, 0, 255)
    return (0, 0, 0)


def _backend_status(backend: OCRBackend) -> str:
    status = getattr(backend, "ocr_engine_status", "available")
    if callable(status):
        status = status()
    return str(status or "available")


def _iter_grayscale_frames(pixel_array: np.ndarray, dataset: Any):
    samples_per_pixel = int(getattr(dataset, "SamplesPerPixel", 1) or 1)

    if samples_per_pixel != 1:
        return None
    if pixel_array.ndim == 2:
        return [(0, pixel_array)]
    if pixel_array.ndim == 3:
        return [(index, pixel_array[index]) for index in range(pixel_array.shape[0])]
    return None


def _frame_to_ocr_image(frame: np.ndarray) -> Image.Image:
    frame_min = float(np.min(frame))
    frame_max = float(np.max(frame))
    if frame_max == frame_min:
        normalized = np.zeros(frame.shape, dtype=np.uint8)
    else:
        normalized = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(
            np.uint8
        )
    return Image.fromarray(normalized, mode="L")


def _redaction_pixel_value(frame: np.ndarray) -> Any:
    if np.issubdtype(frame.dtype, np.integer):
        return np.iinfo(frame.dtype).min
    return 0


def _dataset_to_bytes(dataset: Any) -> bytes:
    buffer = BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def _dicom_result(
    pixel_redaction_status: str,
    ocr_boxes_detected: int = 0,
    boxes_redacted: int = 0,
    frames_processed: int = 0,
    scanned_regions: int = 0,
    ocr_engine_status: str = "not_applicable",
    sanitized_bytes: Optional[bytes] = None,
    include_sanitized_bytes: bool = True,
    ocr_confidence_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "pixel_redaction_status": pixel_redaction_status,
        "ocr_boxes_detected": ocr_boxes_detected,
        "boxes_redacted": boxes_redacted,
        "frames_processed": frames_processed,
        "scanned_regions": scanned_regions,
        "ocr_engine_status": ocr_engine_status,
    }
    if ocr_confidence_threshold is not None:
        result["ocr_confidence_threshold"] = ocr_confidence_threshold
    if include_sanitized_bytes and sanitized_bytes is not None:
        result["sanitized_dicom_bytes"] = sanitized_bytes
    return result

