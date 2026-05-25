import os
import sys
import unittest
from io import BytesIO

from fastapi.testclient import TestClient


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402


client = TestClient(app)


def upload_file(filename, content, content_type="application/octet-stream", profile=None):
    data = {}
    if profile is not None:
        data["profile"] = profile

    return client.post(
        "/api/v1/ingest",
        files={"file": (filename, BytesIO(content), content_type)},
        data=data,
    )


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

    def test_csv_routes_to_csv_handler(self):
        response = upload_file("sample.csv", b"age,diagnosis\n42,test\n", "text/csv")
        self.assert_routes_to(response, "csv", "anonymize_csv")

    def test_text_routes_to_text_handler(self):
        response = upload_file("note.txt", b"clinical note scaffold\n", "text/plain")
        self.assert_routes_to(response, "text", "anonymize_text")

    def test_dicom_extension_routes_to_dicom_handler(self):
        response = upload_file("scan.dcm", b"not-real-dicom-routing-only")
        self.assert_routes_to(response, "dicom", "anonymize_dicom")

    def test_dicom_octet_stream_routes_by_extension(self):
        response = upload_file(
            "scan.dcm",
            b"dicom routing scaffold",
            "application/octet-stream",
        )
        self.assert_routes_to(response, "dicom", "anonymize_dicom")

    def test_nifti_nii_routes_to_nifti_handler(self):
        response = upload_file("brain.nii", b"nifti routing scaffold")
        self.assert_routes_to(response, "nifti", "anonymize_nifti")

    def test_nifti_nii_gz_routes_to_nifti_handler(self):
        response = upload_file("brain.nii.gz", b"compressed nifti scaffold")
        self.assert_routes_to(response, "nifti", "anonymize_nifti")

    def test_wsi_routes_to_wsi_handler(self):
        response = upload_file("slide.svs", b"wsi routing scaffold")
        self.assert_routes_to(response, "wsi", "anonymize_wsi")

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

    def test_dicom_preamble_detection_routes_to_dicom_handler(self):
        dicom_header = b"\x00" * 128 + b"DICM" + b"routing scaffold"
        response = upload_file("scan.bin", dicom_header, "application/octet-stream")
        self.assert_routes_to(response, "dicom", "anonymize_dicom")


if __name__ == "__main__":
    unittest.main()
