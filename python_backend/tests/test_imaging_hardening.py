"""Imaging hardening tests (Phase 5): DICOM, NIfTI, WSI, and raster.

All fixtures are synthetic arrays and synthetic identifiers. No real patient
data appears here, and no assertion snapshots OCR text.
"""

import json
import os
import sys
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import raster_redaction as raster  # noqa: E402
from services.ingestion import route_for_ingestion  # noqa: E402
from services.ocr_redaction import OCRBox, redact_dicom_pixels  # noqa: E402

pydicom = pytest.importorskip("pydicom")

from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import (  # noqa: E402
    ExplicitVRLittleEndian,
    JPEGBaseline8Bit,
    SecondaryCaptureImageStorage,
)

SYNTHETIC_BURNED_TEXT = "SYNTH^BURNED^ID"


class FakeOCRBackend:
    ocr_engine_status = "available"

    def __init__(self, boxes):
        self.boxes = list(boxes)

    def detect_text_boxes(self, image):
        return self.boxes


class OneShotOCRBackend:
    """Reports boxes on the first pass, nothing on verification passes."""

    ocr_engine_status = "available"

    def __init__(self, boxes):
        self.boxes = list(boxes)
        self.calls = 0

    def detect_text_boxes(self, image):
        self.calls += 1
        return self.boxes if self.calls == 1 else []


class UnavailableOCRBackend:
    ocr_engine_status = "unavailable"

    def detect_text_boxes(self, image):
        raise AssertionError("unavailable backend must not run")


class FailingOCRBackend:
    ocr_engine_status = "available"

    def detect_text_boxes(self, image):
        raise RuntimeError("synthetic OCR failure")


def build_dicom(
    pixel_array: np.ndarray,
    transfer_syntax=ExplicitVRLittleEndian,
) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = transfer_syntax
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.8.498.41"
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.42"

    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = SYNTHETIC_BURNED_TEXT
    dataset.PatientID = "SYNTH-PAT-001"
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


# ---------------------------------------------------------------------------
# DICOM: detected-but-not-cleared text
# ---------------------------------------------------------------------------


def test_detected_text_left_in_place_never_yields_sanitized_bytes():
    # A box below the confidence threshold is detected but not redacted. The
    # bytes still carry that text, so they must not come back as sanitized.
    dicom_bytes = build_dicom(np.full((8, 8), 255, dtype=np.uint8))
    backend = FakeOCRBackend([OCRBox(SYNTHETIC_BURNED_TEXT, 0.4, 1, 1, 3, 3)])

    result = redact_dicom_pixels(dicom_bytes, profile="research", ocr_backend=backend)

    assert result["boxes_redacted"] == 0
    assert result["ocr_boxes_detected"] == 1
    assert result["pixel_redaction_status"] == "privacy_requirements_not_met"
    assert "sanitized_dicom_bytes" not in result
    assert result["pixel_validation_status"] == "not_attempted"


def test_no_text_detected_still_validates_before_returning_bytes():
    dicom_bytes = build_dicom(np.full((6, 6), 120, dtype=np.uint8))

    result = redact_dicom_pixels(dicom_bytes, ocr_backend=FakeOCRBackend([]))

    assert result["pixel_redaction_status"] == "completed_no_text_detected"
    assert result["pixel_validation_status"] == "verified"
    assert "sanitized_dicom_bytes" in result


# ---------------------------------------------------------------------------
# DICOM: final-byte validation
# ---------------------------------------------------------------------------


def test_redaction_is_verified_against_the_re_read_bytes():
    pixels = np.full((8, 8), 250, dtype=np.uint8)
    result = redact_dicom_pixels(
        build_dicom(pixels),
        ocr_backend=FakeOCRBackend([OCRBox(SYNTHETIC_BURNED_TEXT, 0.99, 0, 0, 4, 2)]),
    )

    assert result["pixel_redaction_status"] == "completed"
    assert result["pixel_validation_status"] == "verified"

    reread = pydicom.dcmread(BytesIO(result["sanitized_dicom_bytes"]))
    assert reread.pixel_array[0:2, 0:4].max() == 0


def test_validation_failure_blocks_and_withholds_bytes(monkeypatch):
    from services import ocr_redaction

    monkeypatch.setattr(
        ocr_redaction, "_validate_sanitized_dicom", lambda *_args, **_kwargs: False
    )

    result = redact_dicom_pixels(
        build_dicom(np.full((8, 8), 250, dtype=np.uint8)),
        ocr_backend=FakeOCRBackend([OCRBox(SYNTHETIC_BURNED_TEXT, 0.99, 0, 0, 4, 2)]),
    )

    assert result["pixel_redaction_status"] == "privacy_requirements_not_met"
    assert result["pixel_validation_status"] == "verification_failed"
    assert "sanitized_dicom_bytes" not in result


def test_validator_rejects_bytes_whose_redaction_did_not_survive():
    from services import ocr_redaction

    pixels = np.full((8, 8), 250, dtype=np.uint8)
    untouched = build_dicom(pixels)

    # Claim a region was cleared that is in fact still bright.
    assert not ocr_redaction._validate_sanitized_dicom(
        untouched, pixels.shape, [(0, 0, 0, 4, 2, 0)]
    )


def test_validator_rejects_a_shape_change():
    from services import ocr_redaction

    pixels = np.full((8, 8), 0, dtype=np.uint8)
    payload = build_dicom(pixels)

    assert not ocr_redaction._validate_sanitized_dicom(payload, (4, 4), [])


def test_validator_rejects_undecodable_bytes():
    from services import ocr_redaction

    assert not ocr_redaction._validate_sanitized_dicom(b"not a dicom", (4, 4), [])
    assert not ocr_redaction._validate_sanitized_dicom(b"", (4, 4), [])


def test_compressed_transfer_syntax_is_replaced_when_writing_pixels():
    # Writing raw redacted pixels while a compressed UID stands produces a file
    # no decoder can read, so the UID is forced to an uncompressed one.
    from services import ocr_redaction

    pixels = np.zeros((4, 4), dtype=np.uint8)
    dataset = pydicom.dcmread(BytesIO(build_dicom(pixels)))
    dataset.file_meta.TransferSyntaxUID = JPEGBaseline8Bit

    written = ocr_redaction._write_uncompressed_pixels(dataset, pixels)

    reread = pydicom.dcmread(BytesIO(written))
    assert reread.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
    assert reread.pixel_array.shape == (4, 4)


def test_serialization_failure_blocks_without_returning_bytes(monkeypatch):
    # pydicom refuses to write raw pixels under a compressed syntax, which is
    # what _write_uncompressed_pixels prevents. If a write fails anyway, the
    # scan must block rather than hand back the input.
    from services import ocr_redaction

    def exploding_write(_dataset, _pixels):
        raise ValueError("cannot serialize")

    monkeypatch.setattr(ocr_redaction, "_write_uncompressed_pixels", exploding_write)

    result = redact_dicom_pixels(
        build_dicom(np.full((8, 8), 250, dtype=np.uint8)),
        ocr_backend=FakeOCRBackend([OCRBox(SYNTHETIC_BURNED_TEXT, 0.99, 0, 0, 4, 2)]),
    )

    assert result["pixel_redaction_status"] == "serialization_failed"
    assert result["pixel_validation_status"] == "serialization_failed"
    assert "sanitized_dicom_bytes" not in result


def test_no_result_path_carries_ocr_text():
    for backend in (
        FakeOCRBackend([OCRBox(SYNTHETIC_BURNED_TEXT, 0.99, 0, 0, 4, 2)]),
        FakeOCRBackend([OCRBox(SYNTHETIC_BURNED_TEXT, 0.01, 0, 0, 4, 2)]),
        FailingOCRBackend(),
        UnavailableOCRBackend(),
    ):
        result = redact_dicom_pixels(
            build_dicom(np.full((8, 8), 250, dtype=np.uint8)),
            ocr_backend=backend,
        )
        reportable = {
            key: value
            for key, value in result.items()
            if key != "sanitized_dicom_bytes"
        }
        assert SYNTHETIC_BURNED_TEXT not in json.dumps(reportable)


# ---------------------------------------------------------------------------
# Raster redaction
# ---------------------------------------------------------------------------


def _png(color=200, size=(32, 32)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (color, color, color)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_unavailable_ocr_never_returns_raster_bytes():
    outcome = raster.redact_raster_bytes(_png(), ocr_backend=UnavailableOCRBackend())

    assert outcome.released is False
    assert outcome.released_bytes is None
    assert outcome.status == raster.STATUS_UNSCANNABLE
    assert raster.REASON_OCR_UNAVAILABLE in outcome.reason_codes


def test_failing_ocr_never_returns_raster_bytes():
    outcome = raster.redact_raster_bytes(_png(), ocr_backend=FailingOCRBackend())

    assert outcome.released is False
    assert outcome.released_bytes is None
    assert raster.REASON_OCR_FAILED in outcome.reason_codes


def test_detected_text_below_threshold_blocks_the_raster():
    backend = FakeOCRBackend([OCRBox(SYNTHETIC_BURNED_TEXT, 0.05, 2, 2, 8, 6)])

    outcome = raster.redact_raster_bytes(_png(), ocr_backend=backend)

    assert outcome.released is False
    assert outcome.status == raster.STATUS_BLOCKED
    assert raster.REASON_TEXT_NOT_CLEARED in outcome.reason_codes


def test_residual_text_after_redaction_blocks_the_raster():
    # A backend that keeps reporting the same box models a redaction that did
    # not actually clear the text.
    backend = FakeOCRBackend([OCRBox(SYNTHETIC_BURNED_TEXT, 0.9, 2, 2, 8, 6)])

    outcome = raster.redact_raster_bytes(_png(), ocr_backend=backend)

    assert outcome.released is False
    assert outcome.residual_text_boxes >= 1
    assert outcome.validation_status == "verification_failed"


def test_verified_redaction_releases_bytes_with_the_region_filled():
    backend = OneShotOCRBackend([OCRBox(SYNTHETIC_BURNED_TEXT, 0.9, 2, 2, 8, 6)])

    outcome = raster.redact_raster_bytes(_png(), ocr_backend=backend)

    assert outcome.released is True
    assert outcome.validation_status == "verified"
    assert outcome.boxes_redacted == 1
    pixels = np.array(Image.open(BytesIO(outcome.released_bytes)).convert("RGB"))
    assert pixels[2:8, 2:10].max() == 0


def test_clean_image_with_no_text_is_released():
    outcome = raster.redact_raster_bytes(_png(), ocr_backend=FakeOCRBackend([]))

    assert outcome.released is True
    assert outcome.boxes_detected == 0
    assert outcome.validation_status == "verified"


def test_jpeg_exif_is_detected_and_never_carried_into_the_output():
    from PIL import Image as PILImage

    source = PILImage.new("RGB", (24, 24), (180, 180, 180))
    exif = source.getexif()
    exif[0x010E] = "ImageDescription: Jordan Fictional"  # ImageDescription
    buffer = BytesIO()
    source.save(buffer, format="JPEG", exif=exif.tobytes())

    outcome = raster.redact_raster_bytes(
        buffer.getvalue(), ocr_backend=FakeOCRBackend([])
    )

    assert outcome.released is True
    assert outcome.input_metadata_present is True
    assert b"Jordan Fictional" not in outcome.released_bytes
    reread = PILImage.open(BytesIO(outcome.released_bytes))
    assert not reread.info.get("exif")


def test_output_never_carries_input_metadata():
    source = Image.new("RGB", (24, 24), (180, 180, 180))
    buffer = BytesIO()
    # PNG text chunks are a metadata carrier just like EXIF.
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("Patient", "Jordan Fictional")
    source.save(buffer, format="PNG", pnginfo=info)

    outcome = raster.redact_raster_bytes(
        buffer.getvalue(), ocr_backend=FakeOCRBackend([])
    )

    assert outcome.released is True
    assert outcome.input_metadata_present is True
    assert b"Jordan Fictional" not in outcome.released_bytes


def test_oversized_raster_is_rejected():
    with pytest.raises(raster.RasterRedactionError) as exc:
        raster.redact_raster_bytes(b"x" * (raster.MAX_RASTER_BYTES + 1))

    assert exc.value.status_code == 413


def test_undecodable_raster_is_rejected():
    with pytest.raises(raster.RasterRedactionError) as exc:
        raster.redact_raster_bytes(b"not an image at all")

    assert exc.value.detail == raster.REASON_UNDECODABLE


def test_raster_summary_never_carries_bytes_or_ocr_text():
    backend = FakeOCRBackend([OCRBox(SYNTHETIC_BURNED_TEXT, 0.9, 2, 2, 8, 6)])

    outcome = raster.redact_raster_bytes(_png(), ocr_backend=backend)
    summary = json.dumps(outcome.safe_summary())

    assert SYNTHETIC_BURNED_TEXT not in summary
    assert "released_bytes" not in summary
    assert SYNTHETIC_BURNED_TEXT not in repr(outcome)


def test_external_redactor_output_is_independently_verified():
    # Bytes an external redactor produced, where text is still readable.
    backend = FakeOCRBackend([OCRBox(SYNTHETIC_BURNED_TEXT, 0.9, 2, 2, 8, 6)])

    outcome = raster.verify_redacted_raster_bytes(_png(), ocr_backend=backend)

    assert outcome.released is False
    assert raster.REASON_RESIDUAL_TEXT in outcome.reason_codes
    assert outcome.residual_text_boxes == 1


def test_external_redactor_output_passes_when_nothing_is_readable():
    outcome = raster.verify_redacted_raster_bytes(
        _png(), ocr_backend=FakeOCRBackend([])
    )

    assert outcome.released is True
    assert outcome.validation_status == "verified"


def test_external_verification_blocks_without_an_ocr_engine():
    outcome = raster.verify_redacted_raster_bytes(
        _png(), ocr_backend=UnavailableOCRBackend()
    )

    assert outcome.released is False
    assert raster.REASON_OCR_UNAVAILABLE in outcome.reason_codes


# ---------------------------------------------------------------------------
# Standing blockers on volumetric imaging
# ---------------------------------------------------------------------------


def test_dicom_release_records_the_facial_reconstruction_blocker(monkeypatch):
    from services import ingestion

    monkeypatch.setitem(
        ingestion.HANDLER_REGISTRY,
        "dicom",
        lambda file_content, profile: {
            "handler": "anonymize_dicom",
            "routing_status": "handler_selected",
            "anonymization_status": "completed",
            "message": "ok",
            "metadata_summary": {},
            "pixel_redaction_status": "completed",
        },
    )
    payload = build_dicom(np.full((4, 4), 10, dtype=np.uint8))

    response = route_for_ingestion(
        filename="scan.dcm",
        content_type="application/dicom",
        header=payload[:4096],
        profile="strict",
        file_content=payload,
    )

    decision = response["release_decision"]
    assert decision["releasable"] is False
    assert ingestion.FACIAL_RECONSTRUCTION_REASON in decision["reason_codes"]


def test_blocked_pixel_status_is_carried_into_the_reason_codes(monkeypatch):
    from services import ingestion

    monkeypatch.setitem(
        ingestion.HANDLER_REGISTRY,
        "dicom",
        lambda file_content, profile: {
            "handler": "anonymize_dicom",
            "routing_status": "handler_selected",
            "anonymization_status": "completed",
            "message": "ok",
            "metadata_summary": {},
            "pixel_redaction_status": "privacy_requirements_not_met",
        },
    )
    payload = build_dicom(np.full((4, 4), 10, dtype=np.uint8))

    response = route_for_ingestion(
        filename="scan.dcm",
        content_type="application/dicom",
        header=payload[:4096],
        profile="strict",
        file_content=payload,
    )

    assert "privacy_requirements_not_met" in (
        response["release_decision"]["reason_codes"]
    )


def test_nifti_reports_defacing_as_not_implemented():
    nib = pytest.importorskip("nibabel")
    from services.nifti_anonymization import anonymize_nifti_metadata

    volume = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4))
    volume.header["descrip"] = b"Patient Jordan Fictional"

    result = anonymize_nifti_metadata(
        volume.to_bytes(), filename="scan.nii", profile="strict"
    )
    summary = result["metadata_summary"]

    assert summary["defacing_status"] == "not_implemented"
    assert summary["metadata_validation_status"] == "verified"
    assert summary["validation_failures"] == []
    assert "Jordan Fictional" not in json.dumps(summary)


def test_nifti_scrub_is_verified_against_re_read_bytes(monkeypatch):
    nib = pytest.importorskip("nibabel")
    from services import nifti_anonymization

    monkeypatch.setattr(
        nifti_anonymization,
        "_scrub_header_fields",
        lambda header: {"fields_scrubbed": 0, "scrubbed_field_counts": {}},
    )
    volume = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4))
    volume.header["descrip"] = b"Patient Jordan Fictional"

    result = nifti_anonymization.anonymize_nifti_metadata(
        volume.to_bytes(), filename="scan.nii", profile="strict"
    )

    # The scrub was skipped, so the re-read must catch the surviving field.
    assert result["anonymization_status"] == "privacy_requirements_not_met"
    assert "descrip_not_cleared" in result["metadata_summary"]["validation_failures"]


# ---------------------------------------------------------------------------
# WSI
# ---------------------------------------------------------------------------


def test_wsi_never_produces_releasable_bytes():
    from services.wsi_tiling import scan_wsi_bytes

    result = scan_wsi_bytes(b"not a slide", filename="slide.svs")

    assert result["wsi_rewrite_status"] == "not_supported_yet"
    assert "sanitized_bytes" not in result
    assert "sanitized_wsi_bytes" not in result


def test_wsi_ingestion_stays_blocked():
    payload = b"II*\x00" + b"\x00" * 64

    response = route_for_ingestion(
        filename="slide.svs",
        content_type="image/tiff",
        header=payload[:4096],
        profile="strict",
        file_content=payload,
    )

    decision = response["release_decision"]
    assert decision["releasable"] is False
    assert "validated_wsi_writer_unavailable" in decision["reason_codes"]
    assert decision["artifact_sha256"] is None


# ---------------------------------------------------------------------------
# Image endpoint posture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient

    from main import app

    return TestClient(app)


def test_image_endpoint_blocks_instead_of_streaming_when_ocr_is_unavailable(
    api_client, monkeypatch
):
    import main

    monkeypatch.setattr(
        main, "redact_raster_bytes", lambda content, profile="strict": (
            raster.redact_raster_bytes(
                content, profile=profile, ocr_backend=UnavailableOCRBackend()
            )
        )
    )

    response = api_client.post(
        "/anonymize_image",
        files={"file": ("scan.png", BytesIO(_png()), "image/png")},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["released"] is False
    assert raster.REASON_OCR_UNAVAILABLE in body["reason_codes"]
    assert response.headers["content-type"].startswith("application/json")


def test_image_endpoint_streams_only_a_verified_redaction(api_client):
    response = api_client.post(
        "/anonymize_image",
        files={"file": ("scan.png", BytesIO(_png()), "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["X-Redaction-Validation"] == "verified"


def test_image_endpoint_rejects_unsupported_media_types(api_client):
    response = api_client.post(
        "/anonymize_image",
        files={"file": ("scan.gif", BytesIO(b"GIF89a"), "image/gif")},
    )

    assert response.status_code == 400


def test_image_endpoint_never_echoes_content_in_errors(api_client):
    response = api_client.post(
        "/anonymize_image",
        files={"file": ("scan.png", BytesIO(b"Jordan Fictional not an image"), "image/png")},
    )

    assert "Jordan Fictional" not in response.text
