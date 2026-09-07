import json
import os
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ocr_redaction import (  # noqa: E402
    OCRBox,
    redact_dicom_pixels,
    redact_image_regions,
    redact_image_with_backend,
    safe_ocr_response,
)


RAW_OCR_TEXT = "PATIENT^BURNED"


class FakeOCRBackend:
    ocr_engine_status = "available"

    def __init__(self, boxes):
        self.boxes = boxes

    def detect_text_boxes(self, image):
        return self.boxes


def build_grayscale_dicom(pixel_array: np.ndarray) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.8.498.41"
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.42"

    dataset = FileDataset(
        None,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = RAW_OCR_TEXT
    dataset.PatientID = "OCR-PAT-001"
    dataset.Rows = int(pixel_array.shape[0])
    dataset.Columns = int(pixel_array.shape[1])
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixel_array.astype(np.uint8).tobytes()

    buffer = BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def test_fake_ocr_box_gets_redacted():
    image = Image.fromarray(np.full((6, 6), 255, dtype=np.uint8), mode="L")
    result = redact_image_with_backend(
        image,
        ocr_backend=FakeOCRBackend([OCRBox(RAW_OCR_TEXT, 0.98, 1, 1, 3, 2)]),
    )
    pixels = np.asarray(result.image)

    assert result.boxes_detected == 1
    assert result.boxes_redacted == 1
    assert pixels[1:3, 1:4].max() == 0


def test_no_ocr_boxes_leaves_image_unchanged():
    image = np.full((5, 5), 127, dtype=np.uint8)
    result = redact_image_regions(image, [])

    assert result.boxes_detected == 0
    assert result.boxes_redacted == 0
    assert np.array_equal(np.asarray(result.image), image)


def test_low_confidence_boxes_are_skipped():
    image = np.full((5, 5), 255, dtype=np.uint8)
    result = redact_image_regions(
        image,
        [OCRBox(RAW_OCR_TEXT, 0.2, 0, 0, 3, 3)],
        min_confidence=0.5,
    )

    assert result.boxes_detected == 1
    assert result.boxes_redacted == 0
    assert np.asarray(result.image).min() == 255


def test_redaction_summary_does_not_contain_raw_ocr_text():
    result = redact_image_regions(
        np.full((5, 5), 255, dtype=np.uint8),
        [OCRBox(RAW_OCR_TEXT, 0.9, 0, 0, 2, 2)],
    )

    assert RAW_OCR_TEXT not in json.dumps(result.safe_summary())


def test_box_clipping_redacts_boundary_intersection():
    image = np.full((4, 4), 255, dtype=np.uint8)
    result = redact_image_regions(
        image,
        [OCRBox(RAW_OCR_TEXT, 0.99, 2, 2, 10, 10)],
    )
    pixels = np.asarray(result.image)

    assert result.boxes_redacted == 1
    assert pixels[2:, 2:].max() == 0
    assert pixels[:2, :].min() == 255
    assert pixels[:, :2].min() == 255


def test_dicom_pixel_redaction_changes_only_detected_region():
    pixels = np.full((8, 8), 20, dtype=np.uint8)
    pixels[0:2, 0:4] = 250
    dicom_bytes = build_grayscale_dicom(pixels)

    result = redact_dicom_pixels(
        dicom_bytes,
        ocr_backend=FakeOCRBackend([OCRBox(RAW_OCR_TEXT, 0.99, 0, 0, 4, 2)]),
    )
    sanitized = result["sanitized_dicom_bytes"]

    import pydicom

    dataset = pydicom.dcmread(BytesIO(sanitized))
    redacted_pixels = dataset.pixel_array

    assert result["pixel_redaction_status"] == "completed"
    assert result["ocr_boxes_detected"] == 1
    assert result["boxes_redacted"] == 1
    assert redacted_pixels[0:2, 0:4].max() == 0
    assert np.array_equal(redacted_pixels[2:, :], pixels[2:, :])
    assert np.array_equal(redacted_pixels[:, 4:], pixels[:, 4:])


def test_dicom_pixel_data_remains_valid_after_redaction():
    pixels = np.full((4, 4), 100, dtype=np.uint8)
    dicom_bytes = build_grayscale_dicom(pixels)

    result = redact_dicom_pixels(
        dicom_bytes,
        ocr_backend=FakeOCRBackend([OCRBox(RAW_OCR_TEXT, 0.99, 1, 1, 2, 2)]),
    )

    import pydicom

    dataset = pydicom.dcmread(BytesIO(result["sanitized_dicom_bytes"]))
    assert dataset.PixelData
    assert dataset.pixel_array.shape == (4, 4)


def test_dicom_safe_response_does_not_expose_ocr_text_or_bytes():
    result = redact_dicom_pixels(
        build_grayscale_dicom(np.full((4, 4), 100, dtype=np.uint8)),
        ocr_backend=FakeOCRBackend([OCRBox(RAW_OCR_TEXT, 0.99, 0, 0, 2, 2)]),
    )
    safe = safe_ocr_response(result)

    assert "sanitized_dicom_bytes" not in safe
    assert RAW_OCR_TEXT not in json.dumps(safe)


def test_dicom_pixel_redaction_uses_profile_confidence_thresholds():
    dicom_bytes = build_grayscale_dicom(np.full((6, 6), 255, dtype=np.uint8))
    backend = FakeOCRBackend([OCRBox(RAW_OCR_TEXT, 0.4, 1, 1, 3, 3)])

    strict = redact_dicom_pixels(
        dicom_bytes,
        profile="strict",
        ocr_backend=backend,
    )
    research = redact_dicom_pixels(
        dicom_bytes,
        profile="research",
        ocr_backend=backend,
    )

    assert strict["ocr_confidence_threshold"] == 0.3
    assert strict["boxes_redacted"] == 1
    assert research["ocr_confidence_threshold"] == 0.5
    assert research["boxes_redacted"] == 0
