import json
import os
import sys
from io import BytesIO

import pydicom
import pytest
from fastapi.testclient import TestClient
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "week3-test-salt")

from main import app  # noqa: E402
from services.dicom_anonymization import (  # noqa: E402
    anonymize_dicom_file_bytes,
    DicomAnonymizationError,
    anonymize_dicom_metadata,
    anonymize_dicom_file_bytes,
)


client = TestClient(app)

TOP_LEVEL_NAME = "SYNTHETIC^PERSON"
NESTED_NAME = "NESTED^SYNTH"
PATIENT_ID = "SYN-PAT-001"
PRIVATE_VALUE = "SYN_PRIV"
PIXEL_BYTES = b"\x01\x02\x03\x04"


def build_dicom_bytes() -> bytes:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.8.498.1"
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.2"

    dataset = FileDataset(
        None,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = TOP_LEVEL_NAME
    dataset.PatientID = PATIENT_ID
    dataset.PatientBirthDate = "19600101"
    dataset.PatientAge = "045Y"
    dataset.PatientSex = "F"
    dataset.AccessionNumber = "SYN-ACC-01"
    dataset.StudyDate = "20240101"
    dataset.InstitutionName = "SYN_INST"

    nested_item = Dataset()
    nested_item.PatientName = NESTED_NAME
    nested_item.PatientID = "NEST-ID-01"
    dataset.ReferencedPatientSequence = Sequence([nested_item])

    dataset.add_new((0x0043, 0x0010), "LO", "SYN_CREATOR")
    dataset.add_new((0x0043, 0x1010), "LO", PRIVATE_VALUE)

    dataset.Rows = 2
    dataset.Columns = 2
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = PIXEL_BYTES

    buffer = BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def test_dicom_phi_scrubbing_returns_safe_summary_only():
    result = anonymize_dicom_metadata(build_dicom_bytes())
    response_text = json.dumps(result)
    summary = result["metadata_summary"]

    assert result["anonymization_status"] == "completed"
    assert summary["fields_scrubbed"] >= 6
    assert summary["scrubbed_field_counts"]["PatientName"] == 2
    assert summary["scrubbed_field_counts"]["PatientID"] == 2
    assert TOP_LEVEL_NAME not in response_text
    assert PATIENT_ID not in response_text
    assert "SYN_INST" not in response_text


def test_dicom_nested_sequence_scrubbing_is_counted():
    result = anonymize_dicom_metadata(build_dicom_bytes())

    assert result["metadata_summary"]["scrubbed_field_counts"]["PatientName"] == 2
    assert result["metadata_summary"]["scrubbed_field_counts"]["PatientID"] == 2


def test_dicom_private_tags_removed_in_strict_mode():
    result = anonymize_dicom_metadata(build_dicom_bytes(), profile="strict")
    response_text = json.dumps(result)

    assert result["metadata_summary"]["private_tags_removed"] == 2
    assert PRIVATE_VALUE not in response_text


def test_invalid_dicom_is_rejected():
    with pytest.raises(DicomAnonymizationError) as exc:
        anonymize_dicom_metadata(b"not a dicom file")

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid DICOM file format"


def test_dicom_pixel_data_is_not_modified():
    result = anonymize_dicom_metadata(build_dicom_bytes())

    assert result["metadata_summary"]["pixel_data_present"] is True
    assert result["metadata_summary"]["pixel_data_preserved"] is True
    assert result["pixel_redaction_status"] == "metadata_only"


def test_dicom_file_bytes_are_metadata_scrubbed_and_readable():
    result = anonymize_dicom_file_bytes(build_dicom_bytes())
    safe_result = {
        key: value
        for key, value in result.items()
        if key != "anonymized_dicom_bytes"
    }
    dataset = pydicom.dcmread(BytesIO(result["anonymized_dicom_bytes"]))

    assert dataset.PatientName == ""
    assert dataset.PatientID == ""
    assert dataset.PixelData == PIXEL_BYTES
    assert TOP_LEVEL_NAME not in json.dumps(safe_result)
    assert PATIENT_ID not in json.dumps(safe_result)
    assert result["metadata_summary"]["pixel_data_preserved"] is True

def test_dicom_api_returns_completed_without_raw_phi(monkeypatch):
    def fake_pixel_redaction(file_content, profile="strict"):
        dataset = pydicom.dcmread(BytesIO(file_content))
        assert dataset.PatientName == ""
        assert dataset.PatientID == ""
        assert dataset.PixelData == PIXEL_BYTES
        return {
            "pixel_redaction_status": "completed",
            "ocr_boxes_detected": 1,
            "boxes_redacted": 1,
            "frames_processed": 1,
            "scanned_regions": 1,
            "ocr_engine_status": "available",
            "sanitized_dicom_bytes": b"internal-only",
        }

    monkeypatch.setattr(
        "services.ingestion.redact_dicom_pixels",
        fake_pixel_redaction,
    )

    response = client.post(
        "/api/v1/ingest",
        files={
            "file": (
                "scan.dcm",
                BytesIO(build_dicom_bytes()),
                "application/dicom",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    response_text = json.dumps(body)
    assert body["detected_modality"] == "dicom"
    assert body["handler"] == "anonymize_dicom"
    assert body["anonymization_status"] == "completed"
    assert body["pixel_redaction_status"] == "completed"
    assert body["ocr_boxes_detected"] == 1
    assert body["boxes_redacted"] == 1
    assert body["ocr_engine_status"] == "available"
    assert TOP_LEVEL_NAME not in response_text
    assert PATIENT_ID not in response_text
    assert "BURNED_IN_LABEL" not in response_text
    assert "sanitized_dicom_bytes" not in body
    assert "file_bytes" not in body

def test_dicom_download_endpoint_returns_readable_anonymized_file(monkeypatch):
    monkeypatch.setattr(
        "main.audit_logger.log_operation",
        lambda *args, **kwargs: None,
    )

    response = client.post(
        "/anonymize_dicom",
        files={
            "file": (
                "scan.dcm",
                BytesIO(build_dicom_bytes()),
                "application/dicom",
            )
        },
        data={"profile": "strict"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/dicom")
    assert "anonymized_scan.dcm" in response.headers["content-disposition"]
    assert response.headers["x-bioblock-anonymization-status"] == "completed"
    assert int(response.headers["x-bioblock-fields-scrubbed"]) >= 6
    assert int(response.headers["x-bioblock-private-tags-removed"]) == 2

    import pydicom

    downloaded = pydicom.dcmread(BytesIO(response.content), force=False)
    assert str(downloaded.PatientName) == ""
    assert str(downloaded.PatientID) == ""
    assert str(downloaded.PatientBirthDate) == "19000101"
    assert str(downloaded.ReferencedPatientSequence[0].PatientName) == ""
    assert str(downloaded.ReferencedPatientSequence[0].PatientID) == ""
    assert str(downloaded.PatientIdentityRemoved) == "YES"
    assert (0x0043, 0x1010) not in downloaded
    assert bytes(downloaded.PixelData) == PIXEL_BYTES
    assert TOP_LEVEL_NAME.encode("utf-8") not in response.content
    assert PATIENT_ID.encode("utf-8") not in response.content



def test_dicom_research_shifts_dates_generalizes_demographics_and_removes_private_tags():
    result = anonymize_dicom_file_bytes(build_dicom_bytes(), profile="research")
    summary = result["metadata_summary"]

    assert summary["profile"] == "research"
    assert summary["date_strategy"] == "shift"
    assert summary["dates_shifted"] >= 2
    assert summary["generalized_demographics"] == 2
    assert summary["private_tags_removed"] == 2
    assert summary["preserve_dicom_technical_metadata"] is True
    assert "Rows" in summary["technical_metadata_preserved"]
    assert "Columns" in summary["technical_metadata_preserved"]

    import pydicom

    downloaded = pydicom.dcmread(BytesIO(result["anonymized_dicom_bytes"]), force=False)
    assert str(downloaded.PatientName) == ""
    assert str(downloaded.PatientID) == ""
    assert str(downloaded.PatientBirthDate) not in {"19600101", "19000101"}
    assert str(downloaded.StudyDate) not in {"20240101", "19000101"}
    assert str(downloaded.PatientAge) == "040Y"
    assert str(downloaded.PatientSex) == "F"
    assert (0x0043, 0x1010) not in downloaded




