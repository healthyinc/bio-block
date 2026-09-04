"""Downstream enforcement tests (Phase 7).

These cover the two paths that let raw uploaded content out of the pipeline
without touching the sanitizer: the store/index path and the preview path.

All fixtures use synthetic identifiers. No real patient data appears here.
"""

import json
import os
import sys
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import release_gate as gate  # noqa: E402

pydicom = pytest.importorskip("pydicom")

from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import (  # noqa: E402
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
)

SYNTHETIC_NAME = "Jordan Fictional"
SYNTHETIC_SSN = "123-45-6789"
SYNTHETIC_EMAIL = "jordan.fictional@example.invalid"
SYNTHETIC_MRN = "MRN-000101"


class FakeOCRBackend:
    ocr_engine_status = "available"

    def __init__(self, boxes):
        self.boxes = list(boxes)

    def detect_text_boxes(self, image):
        return self.boxes


class UnavailableOCRBackend:
    ocr_engine_status = "unavailable"

    def detect_text_boxes(self, image):
        raise AssertionError("unavailable backend must not run")


def _png(color=200, size=(32, 32)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (color, color, color)).save(buffer, format="PNG")
    return buffer.getvalue()


def build_dicom(pixel_array: np.ndarray) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.8.498.41"
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.42"

    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = SYNTHETIC_NAME
    dataset.PatientID = SYNTHETIC_MRN
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
# Indexing gate
# ---------------------------------------------------------------------------


def test_free_text_is_redacted_before_indexing():
    result = gate.sanitize_for_index(
        {
            "dataset_title": "Cohort study",
            "summary": f"Lead contact {SYNTHETIC_NAME}",
            "extracted_content": f"Patient MRN: 123456 called {SYNTHETIC_EMAIL}",
        }
    )

    assert result.cleared is True
    indexed = " ".join(result.fields.values())
    assert SYNTHETIC_NAME not in indexed
    assert SYNTHETIC_EMAIL not in indexed
    assert "123456" not in indexed
    assert result.detected_entities


def test_clean_content_passes_through_unchanged():
    result = gate.sanitize_for_index(
        {
            "dataset_title": "Cohort study",
            "summary": "Aggregate outcomes by treatment arm.",
            "extracted_content": "Counts only.",
        }
    )

    assert result.cleared is True
    assert result.fields["extracted_content"] == "Counts only."
    assert result.residual_phi_categories == {}


def test_residual_phi_after_redaction_blocks_indexing(monkeypatch):
    from services import release_gate

    monkeypatch.setattr(
        release_gate,
        "residual_phi_categories",
        lambda text: {"US_SSN": 1},
    )

    result = gate.sanitize_for_index({"summary": "anything"})

    assert result.cleared is False
    assert result.status == gate.STATUS_BLOCKED
    assert gate.REASON_RESIDUAL_PHI in result.reason_codes
    assert result.residual_phi_categories == {"US_SSN": 1}
    assert result.fields == {}


def test_blocked_result_carries_no_indexable_text():
    from services import release_gate

    result = release_gate.IndexableText(
        status=gate.STATUS_BLOCKED,
        residual_phi_categories={"PERSON": 1},
        blocked_fields=("summary",),
    )

    assert result.fields == {}
    assert "summary" in result.safe_summary()["blocked_fields"]
    assert SYNTHETIC_NAME not in json.dumps(result.safe_summary())


def test_phi_in_a_metadata_value_blocks_the_store():
    result = gate.sanitize_for_index(
        {"summary": "Aggregate outcomes."},
        metadata={"submitter_note": f"ask {SYNTHETIC_EMAIL}"},
    )

    assert result.cleared is False
    assert gate.REASON_METADATA_PHI in result.reason_codes
    assert "metadata.submitter_note" in result.blocked_fields
    assert SYNTHETIC_EMAIL not in json.dumps(result.safe_summary())


def test_structural_metadata_keys_are_not_scanned():
    # Wallet addresses and CIDs are structural. Rewriting or blocking on them
    # would break filtering without protecting anything.
    result = gate.sanitize_for_index(
        {"summary": "Aggregate outcomes."},
        metadata={
            "owner_address": "0x1234567890abcdef1234567890abcdef12345678",
            "cid": "bafybeigdyrztktx5j7ulnbtqmcqrqm7v4ge2ykqbn3sn5vbmqoiu5w4qbu",
            "file_type": "spreadsheet",
        },
    )

    assert result.cleared is True
    assert result.blocked_fields == ()


def test_research_profile_is_never_indexable():
    result = gate.sanitize_for_index(
        {"summary": f"Contact {SYNTHETIC_NAME}"}, profile="research"
    )

    assert result.cleared is False
    assert result.status == gate.STATUS_EXPERT_DETERMINATION
    assert result.fields == {}


def test_unknown_profile_is_rejected():
    with pytest.raises(gate.ReleaseGateError) as exc:
        gate.sanitize_for_index({"summary": "x"}, profile="public")

    assert exc.value.status_code == 400


def test_oversized_field_is_rejected():
    from services.text_anonymization import MAX_TEXT_BYTES

    with pytest.raises(gate.ReleaseGateError) as exc:
        gate.sanitize_for_index({"summary": "a" * (MAX_TEXT_BYTES + 1)})

    assert exc.value.status_code == 413


def test_detector_failure_blocks_indexing(monkeypatch):
    from services import release_gate
    from services.text_anonymization import TextAnonymizationError

    def failing(*_args, **_kwargs):
        raise TextAnonymizationError("phi_detection_failed", status_code=500)

    monkeypatch.setattr(release_gate, "anonymize_clinical_text", failing)

    with pytest.raises(gate.ReleaseGateError) as exc:
        gate.sanitize_for_index({"summary": "Patient note"})

    assert exc.value.detail == "phi_detection_failed"
    assert exc.value.status_code == 500


# ---------------------------------------------------------------------------
# Preview gate
# ---------------------------------------------------------------------------


def _png_with_metadata(text=f"Patient {SYNTHETIC_NAME}") -> bytes:
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("Patient", text)
    buffer = BytesIO()
    Image.new("RGB", (32, 32), (200, 200, 200)).save(
        buffer, format="PNG", pnginfo=info
    )
    return buffer.getvalue()


def test_raster_preview_is_sanitized_not_passed_through():
    # The old generator streamed the input bytes back verbatim. A clean image
    # can legitimately re-encode to the same pixels, so what proves the bytes
    # went through the sanitizer is that the input's metadata is gone and the
    # output was verified.
    original = _png_with_metadata()

    outcome = gate.sanitized_preview(
        original, filename="scan.png", content_type="image/png"
    )

    assert outcome.released is True
    assert outcome.media_type == "image/png"
    assert SYNTHETIC_NAME.encode() in original
    assert SYNTHETIC_NAME.encode() not in outcome.content
    assert outcome.detail["validation_status"] == "verified"


def test_raster_preview_blocks_when_ocr_is_unavailable(monkeypatch):
    from services import release_gate
    from services.raster_redaction import redact_raster_bytes

    monkeypatch.setattr(
        release_gate,
        "redact_raster_bytes",
        lambda content, profile="strict": redact_raster_bytes(
            content, profile=profile, ocr_backend=UnavailableOCRBackend()
        ),
    )

    outcome = gate.sanitized_preview(_png(), filename="scan.png", content_type="image/png")

    assert outcome.released is False
    assert outcome.content is None


def test_raster_preview_blocks_when_text_is_not_cleared(monkeypatch):
    from services import release_gate
    from services.ocr_redaction import OCRBox
    from services.raster_redaction import redact_raster_bytes

    monkeypatch.setattr(
        release_gate,
        "redact_raster_bytes",
        lambda content, profile="strict": redact_raster_bytes(
            content,
            profile=profile,
            # Below threshold: detected, never cleared.
            ocr_backend=FakeOCRBackend([OCRBox("SYNTH", 0.01, 2, 2, 8, 6)]),
        ),
    )

    outcome = gate.sanitized_preview(_png(), filename="scan.png", content_type="image/png")

    assert outcome.released is False
    assert outcome.status == gate.STATUS_BLOCKED


def test_dicom_preview_renders_only_verified_pixels():
    payload = build_dicom(np.full((16, 16), 90, dtype=np.uint8))

    outcome = gate.sanitized_preview(payload, filename="scan.dcm")

    assert outcome.released is True
    assert outcome.media_type == "image/png"
    # A PNG, not the original DICOM bytes.
    assert outcome.content.startswith(b"\x89PNG")
    assert SYNTHETIC_NAME.encode() not in outcome.content
    assert SYNTHETIC_MRN.encode() not in outcome.content


def test_dicom_preview_blocks_when_pixels_are_unverified(monkeypatch):
    from services import release_gate

    monkeypatch.setattr(
        release_gate,
        "redact_dicom_pixels",
        lambda content, profile="strict": {
            "pixel_redaction_status": "privacy_requirements_not_met",
            "pixel_validation_status": "verification_failed",
        },
    )

    outcome = gate.sanitized_preview(
        build_dicom(np.full((8, 8), 90, dtype=np.uint8)), filename="scan.dcm"
    )

    assert outcome.released is False
    assert gate.REASON_DICOM_PIXELS_UNVERIFIED in outcome.reason_codes
    assert outcome.content is None


def test_dicom_preview_blocks_when_metadata_scrub_fails(monkeypatch):
    from services import release_gate
    from services.dicom_anonymization import DicomAnonymizationError

    def failing(*_args, **_kwargs):
        raise DicomAnonymizationError("Invalid DICOM file format")

    monkeypatch.setattr(release_gate, "anonymize_dicom_file_bytes", failing)

    outcome = gate.sanitized_preview(b"not a dicom", filename="scan.dcm")

    assert outcome.released is False
    assert gate.REASON_DICOM_METADATA_FAILED in outcome.reason_codes


def test_nifti_preview_is_blocked_for_facial_reconstruction():
    outcome = gate.sanitized_preview(b"nifti bytes", filename="head.nii.gz")

    assert outcome.released is False
    assert gate.REASON_FACIAL_RECONSTRUCTION in outcome.reason_codes


def test_wsi_preview_is_blocked_without_a_validated_writer():
    outcome = gate.sanitized_preview(b"slide bytes", filename="slide.svs")

    assert outcome.released is False
    assert gate.REASON_NO_VALIDATED_WRITER in outcome.reason_codes


def test_unknown_preview_modality_is_blocked():
    outcome = gate.sanitized_preview(b"%PDF-1.7", filename="note.pdf")

    assert outcome.released is False
    assert gate.REASON_UNSUPPORTED_PREVIEW in outcome.reason_codes


def test_research_profile_never_previews():
    outcome = gate.sanitized_preview(
        _png(), filename="scan.png", content_type="image/png", profile="research"
    )

    assert outcome.released is False
    assert outcome.status == gate.STATUS_EXPERT_DETERMINATION


def test_empty_preview_upload_is_rejected():
    with pytest.raises(gate.ReleaseGateError):
        gate.sanitized_preview(b"", filename="scan.png", content_type="image/png")


def test_preview_summary_never_carries_bytes():
    outcome = gate.sanitized_preview(
        b"slide bytes", filename="slide.svs"
    )

    assert "content" not in json.dumps(outcome.safe_summary())
    assert "slide bytes" not in repr(outcome)


# ---------------------------------------------------------------------------
# Endpoint posture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient

    from main import app

    return TestClient(app)


def test_store_redacts_before_indexing(api_client):
    response = api_client.post(
        "/store",
        json={
            "summary": f"Study lead {SYNTHETIC_NAME}",
            "dataset_title": "Cohort A",
            "cid": "test-cid-gate-1",
            "metadata": {"dataType": "Institution"},
            "extracted_content": f"Patient MRN: 123456 email {SYNTHETIC_EMAIL}",
        },
    )

    assert response.status_code in (200, 201)
    body = response.json()
    assert body["sanitization"]["sanitization_status"] == gate.STATUS_SANITIZED
    assert body["sanitization"]["detected_entities"]
    for identifier in (SYNTHETIC_NAME, SYNTHETIC_EMAIL, "123456"):
        assert identifier not in response.text


def test_stored_content_is_not_searchable_by_the_raw_identifier(api_client):
    api_client.post(
        "/store",
        json={
            "summary": "Aggregate outcomes.",
            "dataset_title": "Cohort B",
            "cid": "test-cid-gate-2",
            "metadata": {},
            "extracted_content": f"Contact {SYNTHETIC_EMAIL} for details.",
        },
    )

    search = api_client.post("/search", json={"query": SYNTHETIC_EMAIL, "n_results": 5})

    assert search.status_code == 200
    # The redacted text was indexed, so the raw value cannot come back.
    assert SYNTHETIC_EMAIL not in search.text


def test_store_blocks_on_metadata_phi(api_client):
    response = api_client.post(
        "/store",
        json={
            "summary": "Aggregate outcomes.",
            "dataset_title": "Cohort C",
            "cid": "test-cid-gate-3",
            "metadata": {"submitter_note": f"reach {SYNTHETIC_EMAIL}"},
            "extracted_content": "Counts only.",
        },
    )

    assert response.status_code == 422
    assert SYNTHETIC_EMAIL not in response.text


def test_store_enhanced_is_gated_the_same_way(api_client):
    response = api_client.post(
        "/store_enhanced",
        json={
            "summary": f"Study lead {SYNTHETIC_NAME}",
            "dataset_title": "Cohort D",
            "cid": "test-cid-gate-4",
            "metadata": {},
            "extracted_content": f"SSN {SYNTHETIC_SSN}",
        },
    )

    assert response.status_code in (200, 201)
    assert SYNTHETIC_SSN not in response.text
    assert SYNTHETIC_NAME not in response.text


def test_simple_preview_no_longer_streams_the_upload_verbatim(api_client):
    original = _png_with_metadata()

    response = api_client.post(
        "/simple_preview",
        files={"file": ("scan.png", BytesIO(original), "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["X-BioBlock-Preview-Status"] == gate.STATUS_SANITIZED
    # The upload's metadata was in the bytes and is not in the response.
    assert SYNTHETIC_NAME.encode() in original
    assert SYNTHETIC_NAME.encode() not in response.content


def test_preview_blocks_a_modality_it_cannot_sanitize(api_client):
    response = api_client.post(
        "/simple_preview",
        files={"file": ("head.nii.gz", BytesIO(b"nifti bytes"), "application/gzip")},
    )

    assert response.status_code == 422
    body = response.json()
    assert gate.REASON_FACIAL_RECONSTRUCTION in body["reason_codes"]


def test_preview_dicom_returns_sanitized_pixels(api_client):
    payload = build_dicom(np.full((16, 16), 90, dtype=np.uint8))

    response = api_client.post(
        "/preview_dicom",
        files={"file": ("scan.dcm", BytesIO(payload), "application/dicom")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")
    assert SYNTHETIC_NAME.encode() not in response.content


def test_preview_never_echoes_content_in_errors(api_client):
    response = api_client.post(
        "/simple_preview",
        files={
            "file": (
                "scan.png",
                BytesIO(SYNTHETIC_SSN.encode() + b" not an image"),
                "image/png",
            )
        },
    )

    assert SYNTHETIC_SSN not in response.text
