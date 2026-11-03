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
        self.assertEqual(body["downstream"]["ipfs_chunking"], "blocked")
        self.assertEqual(body["downstream"]["cid_encryption"], "blocked")
        self.assertEqual(body["downstream"]["metadata_indexing"], "blocked")
        self.assertEqual(body["downstream"]["blockchain_transaction"], "blocked")
        self.assertFalse(body["release_decision"]["releasable"])

    def assert_completed_metadata_route(self, response, modality, handler):
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["detected_modality"], modality)
        self.assertEqual(body["handler"], handler)
        self.assertEqual(body["routing_status"], "handler_selected")
        self.assertEqual(body["anonymization_status"], "completed")
        self.assertIn("metadata_summary", body)
        self.assertEqual(body["downstream"]["ipfs_chunking"], "blocked")
        self.assertEqual(body["downstream"]["cid_encryption"], "blocked")
        self.assertEqual(body["downstream"]["metadata_indexing"], "blocked")
        self.assertEqual(body["downstream"]["blockchain_transaction"], "blocked")
        self.assertFalse(body["release_decision"]["releasable"])

    def test_csv_upload_returns_completed_safe_tabular_summary(self):
        csv_content = (
            b"name,email,phone,mrn,age,gender,diagnosis\n"
            b"Alice Adams,alice@example.com,555-111-2222,MRN-001,31,F,flu\n"
            b"Bob Baker,bob@example.com,555-111-3333,MRN-002,32,F,cold\n"
            b"Carol Chen,carol@example.com,555-111-4444,MRN-003,33,M,flu\n"
            b"Dan Diaz,dan@example.com,555-111-5555,MRN-004,34,M,cold\n"
            b"Eve Evans,eve@example.com,555-111-6666,MRN-005,35,F,flu\n"
        )
        response = upload_file("sample.csv", csv_content, "text/csv")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        response_text = json.dumps(body)
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["detected_modality"], "csv")
        self.assertEqual(body["handler"], "anonymize_csv")
        self.assertEqual(body["routing_status"], "handler_selected")
        self.assertEqual(
            body["anonymization_status"],
            "completed_with_warnings",
        )
        self.assertNotIn("placeholder", response_text.lower())
        self.assertIn("tabular_summary", body)
        summary = body["tabular_summary"]
        self.assertEqual(summary["rows_in"], 5)
        self.assertEqual(summary["rows_out"], 5)
        self.assertEqual(summary["k"], 5)
        self.assertEqual(summary["l"], 2)
        self.assertEqual(
            summary["direct_identifiers_removed"],
            ["name", "email", "phone", "mrn"],
        )
        self.assertEqual(summary["quasi_identifiers_used"], ["age", "gender"])
        self.assertEqual(summary["sensitive_column"], "diagnosis")
        self.assertTrue(summary["k_anonymity_satisfied"])
        self.assertTrue(summary["l_diversity_satisfied"])
        self.assertEqual(
            summary["safe_harbor_report"]["safe_harbor_validation_status"],
            "passed_with_warnings",
        )
        self.assertEqual(
            summary["safe_harbor_report"]["unresolved_identifier_categories"],
            [],
        )
        self.assertEqual(body["downstream"]["ipfs_chunking"], "blocked")
        self.assertEqual(body["downstream"]["cid_encryption"], "blocked")
        self.assertEqual(body["downstream"]["metadata_indexing"], "blocked")
        self.assertEqual(body["downstream"]["blockchain_transaction"], "blocked")
        self.assertFalse(body["release_decision"]["releasable"])
        for raw_value in (
            "Alice Adams",
            "alice@example.com",
            "555-111-2222",
            "MRN-001",
            "flu",
            "cold",
        ):
            self.assertNotIn(raw_value, response_text)
        self.assertNotIn("_internal_anonymized_csv", body)
        self.assertNotIn("file_bytes", body)

    def test_synthea_csv_ingestion_summary_is_safe_and_complete(self):
        csv_content = (
            b"Id,BIRTHDATE,SSN,DRIVERS,PASSPORT,FIRST,LAST,ADDRESS,"
            b"GENDER,ZIP,LAT,LON,HEALTHCARE_EXPENSES\n"
            b"uuid-a,1980-01-01,111-22-3333,DL-A,P-A,Alice,Alpha,1 Fake St,"
            b"F,02139,42.3601,-71.0589,1000\n"
            b"uuid-b,1981-01-02,222-33-4444,DL-B,P-B,Bob,Beta,2 Fake St,"
            b"M,02140,42.3611,-71.0599,1100\n"
            b"uuid-c,1982-01-03,333-44-5555,DL-C,P-C,Carol,Gamma,3 Fake St,"
            b"F,02141,42.3621,-71.0609,1200\n"
            b"uuid-d,1983-01-04,444-55-6666,DL-D,P-D,Dan,Delta,4 Fake St,"
            b"M,02142,42.3631,-71.0619,1300\n"
            b"uuid-e,1984-01-05,555-66-7777,DL-E,P-E,Eve,Epsilon,5 Fake St,"
            b"F,02143,42.3641,-71.0629,1400\n"
        )

        response = upload_file("patients.csv", csv_content, "text/csv")
        body = response.json()
        response_text = json.dumps(body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            body["anonymization_status"],
            "completed_with_warnings",
        )
        self.assertEqual(
            body["tabular_summary"]["precise_geography_columns_removed"],
            ["LAT", "LON"],
        )
        self.assertNotIn("LAT", body["tabular_summary"]["output_columns"])
        self.assertNotIn("LON", body["tabular_summary"]["output_columns"])
        self.assertNotIn("ZIP", body["tabular_summary"]["output_columns"])
        for raw_value in (
            "uuid-a",
            "111-22-3333",
            "DL-A",
            "P-A",
            "Alice",
            "1 Fake St",
            "1980-01-01",
            "42.3601",
            "-71.0589",
            "1000",
        ):
            self.assertNotIn(raw_value, response_text)

    def test_csv_ingestion_does_not_report_completed_when_k_fails(self):
        csv_content = b"Id,age,gender\na,30,F\nb,31,M\n"

        response = upload_file("small.csv", csv_content, "text/csv")
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            body["anonymization_status"],
            "failed_privacy_validation",
        )
        self.assertFalse(body["tabular_summary"]["k_anonymity_satisfied"])
        self.assertEqual(body["tabular_summary"]["min_group_size"], 2)

    def test_csv_download_endpoint_reports_analysis_without_releasing_rows(self):
        # Phase 9: /anonymize_csv no longer streams rows. It shares the single
        # release-decision function with /api/v1/ingest, which holds CSV at
        # manual review, so the analysis comes back but the rows do not.
        csv_content = (
            b"name,email,phone,mrn,age,gender,diagnosis\n"
            b"Alice Adams,alice@example.com,555-111-2222,MRN-001,31,F,flu\n"
            b"Bob Baker,bob@example.com,555-111-3333,MRN-002,32,F,cold\n"
            b"Carol Chen,carol@example.com,555-111-4444,MRN-003,33,M,flu\n"
            b"Dan Diaz,dan@example.com,555-111-5555,MRN-004,34,M,cold\n"
        )

        with patch("main.audit_logger.log_operation", lambda *args, **kwargs: None):
            response = client.post(
                "/anonymize_csv",
                files={"file": ("sample.csv", BytesIO(csv_content), "text/csv")},
                data={"k": "2", "l": "2"},
            )

        self.assertEqual(response.status_code, 422)
        self.assertTrue(
            response.headers["content-type"].startswith("application/json")
        )
        body = response.json()
        self.assertEqual(body["anonymization_status"], "completed_with_warnings")
        self.assertFalse(body["release_decision"]["releasable"])
        self.assertIsNone(body["release_decision"]["artifact_sha256"])
        self.assertEqual(body["serialized_output_validation"], "passed")
        self.assertTrue(body["k_anonymity_satisfied"])
        self.assertTrue(body["l_diversity_satisfied"])

        # The analysis is still useful: the removal plan is reported.
        summary = body["tabular_summary"]
        self.assertEqual(summary["rows_in"], 4)
        self.assertIn("name", summary["columns_removed"])
        self.assertIn("email", summary["columns_removed"])
        self.assertEqual(summary["output_columns"], ["age", "gender", "diagnosis"])

        # No row content, generalized or otherwise, and no raw identifier.
        self.assertNotIn("31-32", response.text)
        for raw_identifier in (
            "Alice Adams",
            "alice@example.com",
            "555-111-2222",
            "MRN-001",
        ):
            self.assertNotIn(raw_identifier, response.text)

    def test_csv_download_endpoint_ignores_swagger_placeholder_strings(self):
        csv_content = (
            b"name,email,phone,mrn,age,gender,diagnosis\n"
            b"Alice Adams,alice@example.com,555-111-2222,MRN-001,31,F,flu\n"
            b"Bob Baker,bob@example.com,555-111-3333,MRN-002,32,F,cold\n"
            b"Carol Chen,carol@example.com,555-111-4444,MRN-003,33,M,flu\n"
            b"Dan Diaz,dan@example.com,555-111-5555,MRN-004,34,M,cold\n"
        )

        with patch("main.audit_logger.log_operation", lambda *args, **kwargs: None):
            response = client.post(
                "/anonymize_csv",
                files={"file": ("sample.csv", BytesIO(csv_content), "text/csv")},
                data={
                    "k": "2",
                    "l": "2",
                    "direct_identifiers": "string",
                    "quasi_identifiers": "string",
                    "sensitive_column": "string",
                },
            )

        # The placeholder strings are still ignored rather than treated as
        # column names: the default classification runs and the analysis
        # succeeds. Only the row release changed.
        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["anonymization_status"], "completed_with_warnings")
        self.assertEqual(
            body["tabular_summary"]["output_columns"],
            ["age", "gender", "diagnosis"],
        )
        self.assertNotIn("Alice Adams", response.text)

    def test_openapi_includes_csv_download_endpoint(self):
        response = client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertIn("/anonymize_csv", response.json()["paths"])

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
        self.assertEqual(body["date_strategy"], "redact")
        self.assertEqual(body["text_identifier_strategy"], "redact")
        self.assertRegex(body["anonymized_text"], r"RECORD_\d{3,}")
        self.assertIn("<REDACTED_EMAIL>", body["anonymized_text"])
        self.assertIn("diabetes", body["anonymized_text"])
        self.assertEqual(body["detected_entities"]["MEDICAL_RECORD_NUMBER"], 1)
        self.assertEqual(body["detected_entities"]["EMAIL_ADDRESS"], 1)
        self.assertEqual(body["entity_count"], 2)
        self.assertEqual(body["detection_sources"], {"structured_pattern": 2})
        self.assertEqual(body["ner_model"], "en_core_web_sm")
        self.assertTrue(body["trained_ner_active"])
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
        self.assertRegex(body["anonymized_text"], r"PATIENTID_\d{3,}")

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

        self.assertEqual(fake_file.seek_offsets, [0, 0])
        self.assertEqual(fake_file.position, 0)
        self.assertEqual(result["detected_modality"], "csv")
        self.assertEqual(result["handler"], "anonymize_csv")
        self.assertEqual(
            result["anonymization_status"],
            "failed_privacy_validation",
        )
        self.assertFalse(result["tabular_summary"]["k_anonymity_satisfied"])
        self.assertIn("tabular_summary", result)

    def test_text_upload_returns_only_redacted_content_and_safe_counts(self):
        raw_text = (
            b"Patient Rahul Sharma has MRN-458921 and was examined by "
            b"Dr. Amit Verma."
        )

        response = upload_file("synthetic-note.txt", raw_text, "text/plain")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        serialized = response.text
        self.assertEqual(body["detected_modality"], "text")
        self.assertEqual(body["anonymization_status"], "completed")
        self.assertNotIn("Rahul Sharma", serialized)
        self.assertNotIn("Amit Verma", serialized)
        self.assertNotIn("458921", serialized)
        self.assertRegex(body["anonymized_text"], r"(?:PATIENT|PROVIDER)_\d{3,}")
        self.assertRegex(body["anonymized_text"], r"RECORD_\d{3,}")
        self.assertEqual(body["detected_entities"]["PERSON"], 2)
        self.assertEqual(body["detected_entities"]["MEDICAL_RECORD_NUMBER"], 1)
        for raw_span_key in (
            '"start"',
            '"end"',
            '"score"',
            '"original_label"',
            '"entity_text"',
        ):
            self.assertNotIn(raw_span_key, serialized)

    def test_text_upload_model_failure_cannot_return_original_content(self):
        raw_text = b"Patient Synthetic Person has MRN-458921."
        previous_model = os.environ.get("PHI_NER_MODEL")
        os.environ["PHI_NER_MODEL"] = "missing_configured_phi_model"
        try:
            response = upload_file("synthetic-note.txt", raw_text, "text/plain")
        finally:
            if previous_model is None:
                os.environ.pop("PHI_NER_MODEL", None)
            else:
                os.environ["PHI_NER_MODEL"] = previous_model

        self.assertEqual(response.status_code, 503)
        serialized = response.text
        self.assertEqual(response.json()["detail"], "ner_model_unavailable")
        self.assertNotIn("Synthetic Person", serialized)
        self.assertNotIn("458921", serialized)
        self.assertNotIn("anonymized_text", serialized)
        self.assertNotIn("traceback", serialized.lower())

    def test_strict_text_upload_redacts_arbitrary_proper_noun(self):
        response = upload_file(
            "synthetic-note.txt",
            b"Kartik went home after lunch.",
            "text/plain",
            "strict",
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotIn("Kartik", response.text)
        self.assertRegex(body["anonymized_text"], r"(?:PATIENT|PROVIDER)_\d{3,}")
        self.assertEqual(body["anonymization_status"], "completed")
        self.assertEqual(body["detected_entities"], {"PERSON": 1})
        self.assertEqual(body["detection_sources"], {"strict_proper_noun": 1})

    def test_research_text_upload_requires_expert_determination_without_content(self):
        response = upload_file(
            "synthetic-note.txt",
            b"Patient Synthetic Person has MRN-458921.",
            "text/plain",
            "research",
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["anonymization_status"],
            "expert_determination_required",
        )
        self.assertFalse(body["release_decision"]["releasable"])
        self.assertNotIn("anonymized_text", body)
        self.assertNotIn("Synthetic Person", response.text)
        self.assertNotIn("458921", response.text)
        self.assertTrue(all(value == "blocked" for value in body["downstream"].values()))

    def test_safe_harbor_v1_profile_is_accepted_as_canonical_policy(self):
        response = upload_file(
            "synthetic-note.txt",
            b"Patient Synthetic Person was admitted.",
            "text/plain",
            "safe_harbor_v1",
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["privacy_profile"], "safe_harbor_v1")
        self.assertEqual(body["privacy_policy"], "safe_harbor_v1")
        self.assertTrue(body["release_decision"]["releasable"])
        self.assertNotIn("Synthetic Person", response.text)


if __name__ == "__main__":
    unittest.main()


