import asyncio
import json
import os
import sys
import tempfile
import unittest
from io import BytesIO
from unittest.mock import patch

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "week2-test-salt")

from main import app, ingest_file  # noqa: E402
from services.ingestion import TEXT_READ_LIMIT_BYTES  # noqa: E402

client = TestClient(app)


def fake_dicom_pixel_redaction(file_content, profile="strict"):
    return {
        "pixel_redaction_status": "completed",
        "ocr_boxes_detected": 1,
        "boxes_redacted": 1,
        "frames_processed": 1,
        "scanned_regions": 1,
        "ocr_engine_status": "available",
        "sanitized_dicom_bytes": b"internal-only",
    }


def fake_wsi_scan(file_content, filename):
    return {
        "pixel_redaction_status": "redaction_plan_ready",
        "ocr_boxes_detected": 2,
        "boxes_redacted": 0,
        "redaction_plan_boxes": 2,
        "tiles_scanned": 4,
        "priority_regions_scanned": [
            "corner_top_left",
            "corner_top_right",
        ],
        "image_dimensions": {"width": 2048, "height": 2048},
        "tile_size": 1024,
        "ocr_engine_status": "available",
        "wsi_rewrite_status": "not_supported_yet",
    }


class FakeUploadFile:
    def __init__(self, filename, content, content_type):
        self.filename = filename
        self.content = content
        self.content_type = content_type
        self.position = 0
        self.seek_offsets = []

    async def read(self, size=-1):
        if size is None or size < 0:
            size = len(self.content) - self.position

        start = self.position
        end = min(start + size, len(self.content))
        self.position = end
        return self.content[start:end]

    async def seek(self, offset):
        self.seek_offsets.append(offset)
        self.position = offset


def upload_file(
    filename, content, content_type="application/octet-stream", profile=None
):
    data = {}
    if profile is not None:
        data["profile"] = profile

    return client.post(
        "/api/v1/ingest",
        files={"file": (filename, BytesIO(content), content_type)},
        data=data,
    )


def build_dicom_bytes():
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.8.498.11"
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.12"

    dataset = FileDataset(
        None,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "SYN^ROUTE"
    dataset.PatientID = "SYN-ROUTE-ID"

    nested_item = Dataset()
    nested_item.PatientName = "NEST^ROUTE"
    dataset.ReferencedPatientSequence = Sequence([nested_item])

    dataset.Rows = 2
    dataset.Columns = 2
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = b"\x01\x02\x03\x04"

    buffer = BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def build_nifti_bytes(suffix=".nii"):
    data = np.arange(8, dtype=np.int16).reshape((2, 2, 2))
    image = nib.Nifti1Image(data, np.eye(4))
    image.header["descrip"] = b"SYNTHETIC_ROUTING_HEADER"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        nib.save(image, temp_path)
        with open(temp_path, "rb") as saved_file:
            return saved_file.read()
    finally:
        os.unlink(temp_path)


class TestIngestionRouting(unittest.TestCase):
    def assert_routes_to(self, response, modality, handler):
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["detected_modality"], modality)
        self.assertEqual(body["handler"], handler)
        self.assertEqual(body["routing_status"], "handler_selected")
        self.assertEqual(body["anonymization_status"], "placeholder")
        self.assertEqual(body["downstream"]["ipfs_chunking"], "pending")
        self.assertEqual(body["downstream"]["cid_encryption"], "pending")
        self.assertEqual(body["downstream"]["metadata_indexing"], "pending")
        self.assertEqual(body["downstream"]["blockchain_transaction"], "pending")

    def assert_completed_metadata_route(self, response, modality, handler):
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["detected_modality"], modality)
        self.assertEqual(body["handler"], handler)
        self.assertEqual(body["routing_status"], "handler_selected")
        self.assertEqual(body["anonymization_status"], "completed")
        self.assertIn("metadata_summary", body)
        self.assertEqual(body["downstream"]["ipfs_chunking"], "pending")
        self.assertEqual(body["downstream"]["cid_encryption"], "pending")
        self.assertEqual(body["downstream"]["metadata_indexing"], "pending")
        self.assertEqual(body["downstream"]["blockchain_transaction"], "pending")

    def test_csv_routes_to_csv_handler(self):
        response = upload_file("sample.csv", b"age,diagnosis\n42,test\n", "text/csv")
        self.assert_routes_to(response, "csv", "anonymize_csv")

    def test_text_routes_to_text_handler(self):
        response = upload_file(
            "note.txt",
            (b"Patient has MRN: 123456 and diabetes. " b"Email john.doe@example.com."),
            "text/plain",
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["detected_modality"], "text")
        self.assertEqual(body["handler"], "anonymize_text")
        self.assertEqual(body["routing_status"], "handler_selected")
        self.assertEqual(body["anonymization_status"], "completed")
        self.assertNotIn("123456", body["anonymized_text"])
        self.assertNotIn("john.doe@example.com", body["anonymized_text"])
        self.assertIn("MRN_", body["anonymized_text"])
        self.assertIn("<REDACTED_EMAIL>", body["anonymized_text"])
        self.assertIn("diabetes", body["anonymized_text"])
        self.assertEqual(body["detected_entities"]["MEDICAL_RECORD_NUMBER"], 1)
        self.assertEqual(body["detected_entities"]["EMAIL_ADDRESS"], 1)
        self.assertEqual(body["downstream"]["ipfs_chunking"], "pending")
        self.assertEqual(body["downstream"]["cid_encryption"], "pending")
        self.assertEqual(body["downstream"]["metadata_indexing"], "pending")
        self.assertEqual(body["downstream"]["blockchain_transaction"], "pending")

    def test_text_routes_by_extension_despite_mime_mismatch(self):
        response = upload_file(
            "note.txt",
            b"Patient ID PT-1001 was admitted.",
            "application/octet-stream",
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["detected_modality"], "text")
        self.assertEqual(body["anonymization_status"], "completed")
        self.assertNotIn("PT-1001", body["anonymized_text"])
        self.assertIn("PATIENT_ID_", body["anonymized_text"])

    def test_dicom_extension_routes_to_dicom_handler(self):
        with patch("services.ingestion.redact_dicom_pixels", fake_dicom_pixel_redaction):
            response = upload_file("scan.dcm", build_dicom_bytes())
        self.assert_completed_metadata_route(response, "dicom", "anonymize_dicom")
        body = response.json()
        self.assertEqual(body["pixel_redaction_status"], "completed")
        self.assertEqual(body["ocr_boxes_detected"], 1)
        self.assertNotIn("sanitized_dicom_bytes", body)

    def test_dicom_octet_stream_routes_by_extension(self):
        with patch("services.ingestion.redact_dicom_pixels", fake_dicom_pixel_redaction):
            response = upload_file(
                "scan.dcm",
                build_dicom_bytes(),
                "application/octet-stream",
            )
        self.assert_completed_metadata_route(response, "dicom", "anonymize_dicom")
        self.assertEqual(response.json()["pixel_redaction_status"], "completed")

    def test_nifti_nii_routes_to_nifti_handler(self):
        response = upload_file("brain.nii", build_nifti_bytes(".nii"))
        self.assert_completed_metadata_route(response, "nifti", "anonymize_nifti")

    def test_nifti_nii_gz_routes_to_nifti_handler(self):
        response = upload_file("brain.nii.gz", build_nifti_bytes(".nii.gz"))
        self.assert_completed_metadata_route(response, "nifti", "anonymize_nifti")

    def test_wsi_routes_to_wsi_handler(self):
        with patch("services.ingestion.scan_wsi_bytes", fake_wsi_scan):
            response = upload_file("slide.svs", b"wsi routing scaffold")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["detected_modality"], "wsi")
        self.assertEqual(body["handler"], "anonymize_wsi")
        self.assertEqual(body["routing_status"], "handler_selected")
        self.assertEqual(body["anonymization_status"], "redaction_plan_ready")
        self.assertEqual(body["pixel_redaction_status"], "redaction_plan_ready")
        self.assertEqual(body["tiles_scanned"], 4)
        self.assertEqual(body["boxes_redacted"], 0)
        self.assertEqual(body["wsi_rewrite_status"], "not_supported_yet")
        self.assertNotIn("ocr_text", json.dumps(body).lower())

    def test_unsupported_file_type_is_rejected(self):
        response = upload_file("archive.zip", b"zip scaffold")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Unsupported file modality")

    def test_empty_file_is_rejected(self):
        response = upload_file("empty.csv", b"", "text/csv")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Uploaded file is empty")

    def test_invalid_profile_is_rejected(self):
        response = upload_file("sample.csv", b"a,b\n1,2\n", "text/csv", "open")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid privacy profile", response.json()["detail"])

    def test_text_upload_with_unsupported_encoding_is_rejected(self):
        response = upload_file("note.txt", b"\xff\xfe\x00", "text/plain")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "Text uploads must be UTF-8 encoded",
        )

    def test_large_text_upload_is_rejected(self):
        response = upload_file(
            "large-note.txt",
            b"a" * (TEXT_READ_LIMIT_BYTES + 1),
            "text/plain",
        )

        self.assertEqual(response.status_code, 413)
        self.assertIn("Text uploads must be", response.json()["detail"])

    def test_dicom_preamble_detection_routes_to_dicom_handler(self):
        with patch("services.ingestion.redact_dicom_pixels", fake_dicom_pixel_redaction):
            response = upload_file(
                "scan.bin",
                build_dicom_bytes(),
                "application/octet-stream",
            )
        self.assert_completed_metadata_route(response, "dicom", "anonymize_dicom")
        self.assertNotEqual(
            response.json()["pixel_redaction_status"],
            "not_started_week4",
        )

    def test_endpoint_resets_upload_stream_after_header_read(self):
        fake_file = FakeUploadFile(
            filename="sample.csv",
            content=b"age,diagnosis\n42,test\n",
            content_type="text/csv",
        )

        result = asyncio.run(ingest_file(file=fake_file, profile="strict"))

        self.assertEqual(fake_file.seek_offsets, [0])
        self.assertEqual(fake_file.position, 0)
        self.assertEqual(result["detected_modality"], "csv")
        self.assertEqual(result["handler"], "anonymize_csv")


if __name__ == "__main__":
    unittest.main()
