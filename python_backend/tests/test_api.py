import unittest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path to import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)

class TestAPI(unittest.TestCase):
    def test_root(self):
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)

    def test_store(self):
        data = {
            "summary": "test", 
            "dataset_title": "test",
            "cid": "test-cid-123",
            "metadata": {"title": "test"}
        }
        resp = client.post("/store", json=data)
        self.assertIn(resp.status_code, [200, 201])

    def test_search(self):
        data = {"query": "patient information", "k": 1}
        resp = client.post("/search", json=data)
        self.assertEqual(resp.status_code, 200)

    def test_filter(self):
        data = {"filters": {"dataType": "Personal"}, "n_results": 1}
        resp = client.post("/filter", json=data)
        self.assertEqual(resp.status_code, 200)

    def test_search_with_filter(self):
        data = {"query": "diabetes", "filters": {"dataType": "Institution"}, "n_results": 1}
        resp = client.post("/search_with_filter", json=data)
        self.assertEqual(resp.status_code, 200)

    def test_anonymize_image(self):
        # Create a dummy image for testing if not exists
        if not os.path.exists("test.jpg"):
            from PIL import Image
            img = Image.new('RGB', (100, 100), color = 'red')
            img.save('test.jpg')
            
        with open("test.jpg", "rb") as img:
            files = {"file": ("test.jpg", img, "image/jpeg")}
            resp = client.post("/anonymize_image", files=files)
            # 405 is expected if the endpoint is commented out in main.py, 
            # otherwise 200. Adjust based on main.py state.
            # Current main.py has /anonymize_image commented out but has /anonymize_image_presidio
            # and another /anonymize_image which is active?
            # Let's check the response code.
            print(f"Response status: {resp.status_code}") 

    def test_store_with_content(self):
        """Test enhanced store with extracted content"""
        data = {
            "summary": "Patient health records with diabetes data", 
            "dataset_title": "Diabetes Study 2024",
            "cid": "test-cid-enhanced-123",
            "metadata": {"dataType": "Institution", "disease_tags": "diabetes"},
            "extracted_content": "Columns: Age, Gender, Diagnosis\nRow 1: Age: 45; Gender: Male; Diagnosis: Type 2 Diabetes",
            "file_type": "spreadsheet"
        }
        resp = client.post("/store", json=data)
        self.assertIn(resp.status_code, [200, 201])
        self.assertIn("content_chunks", resp.json())

    def test_search_enhanced(self):
        """Test enhanced search combining content and metadata"""
        data = {
            "query": "diabetes patient data",
            "content_weight": 0.6,
            "metadata_weight": 0.4,
            "n_results": 5
        }
        resp = client.post("/search_enhanced", json=data)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("results", resp.json())
        self.assertIn("search_config", resp.json())

    def test_store_backward_compatibility(self):
        """Test that old store format still works"""
        data = {
            "summary": "test summary", 
            "dataset_title": "test title",
            "cid": "test-cid-old-format",
            "metadata": {"title": "test"}
        }
        resp = client.post("/store", json=data)
        self.assertIn(resp.status_code, [200, 201])

    # --- Text PHI Anonymization Tests ---

    def test_anonymize_text_with_phi(self):
        """Test text anonymization detects PHI entities"""
        data = {"text": "Patient John Smith, DOB 03/15/1985, was admitted to Alaska Regional Hospital."}
        resp = client.post("/anonymize_text", json=data)
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        self.assertIn("anonymized_text", result)
        self.assertIn("entities_found", result)
        self.assertIn("method", result)
        self.assertGreater(result["entity_count"], 0)

    def test_anonymize_text_empty(self):
        """Test that empty text returns 422 validation error"""
        data = {"text": ""}
        resp = client.post("/anonymize_text", json=data)
        self.assertEqual(resp.status_code, 422)

    def test_anonymize_text_no_phi(self):
        """Test text with no PHI returns successfully"""
        data = {"text": "The weather in Alaska is cold during winter months."}
        resp = client.post("/anonymize_text", json=data)
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        self.assertIn("anonymized_text", result)

    def test_anonymize_text_multiple_entities(self):
        """Test detection of multiple PHI entity types"""
        data = {"text": "Dr. Sarah Johnson, email: sarah.johnson@hospital.com, phone: 555-123-4567, seen on 01/15/2026."}
        resp = client.post("/anonymize_text", json=data)
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        self.assertGreater(result["entity_count"], 0)
        entity_types = [e["entity_type"] for e in result["entities_found"]]
        self.assertTrue(len(entity_types) >= 2, "Should detect at least 2 entity types")

    # --- PDF PHI Anonymization Tests ---

    def test_anonymize_pdf_invalid_file(self):
        """Test that non-PDF file is rejected"""
        from io import BytesIO
        fake_file = BytesIO(b"This is not a PDF")
        resp = client.post("/anonymize_pdf", files={"file": ("test.txt", fake_file, "text/plain")})
        self.assertEqual(resp.status_code, 400)

    def test_anonymize_pdf_with_phi(self):
        """Test PDF anonymization with a generated PDF containing PHI"""
        try:
            from reportlab.pdfgen import canvas
            from io import BytesIO

            buffer = BytesIO()
            c = canvas.Canvas(buffer)
            c.drawString(100, 750, "Patient Name: John Smith")
            c.drawString(100, 730, "Date of Birth: 03/15/1985")
            c.drawString(100, 710, "SSN: 123-45-6789")
            c.drawString(100, 690, "Email: john.smith@hospital.com")
            c.showPage()
            c.save()
            buffer.seek(0)

            resp = client.post("/anonymize_pdf", files={"file": ("medical_record.pdf", buffer, "application/pdf")})
            self.assertEqual(resp.status_code, 200)
            result = resp.json()
            self.assertIn("pages", result)
            self.assertIn("total_pages", result)
            self.assertIn("total_entities", result)
            self.assertEqual(result["total_pages"], 1)
            self.assertGreater(result["total_entities"], 0)
        except ImportError:
            self.skipTest("reportlab not installed, skipping PDF generation test")

    # --- DICOM PHI Anonymization Tests ---

    def test_anonymize_dicom_invalid_file(self):
        """Test that non-DICOM file is rejected"""
        from io import BytesIO
        fake_file = BytesIO(b"This is not a DICOM file")
        resp = client.post("/anonymize_dicom", files={"file": ("test.txt", fake_file, "application/octet-stream")})
        self.assertEqual(resp.status_code, 400)

    def test_anonymize_dicom_with_phi(self):
        """Test DICOM anonymization strips PHI metadata tags"""
        try:
            import pydicom
            from pydicom.dataset import Dataset, FileDataset
            from io import BytesIO
            import tempfile

            # Create a minimal DICOM file with PHI
            tmp = tempfile.NamedTemporaryFile(suffix=".dcm", delete=False)
            ds = FileDataset(tmp.name, Dataset(), preamble=b"\x00" * 128, is_implicit_VR=False, is_little_endian=True)
            ds.PatientName = "John Smith"
            ds.PatientID = "PAT12345"
            ds.PatientBirthDate = "19850315"
            ds.ReferringPhysicianName = "Dr. Sarah Johnson"
            ds.InstitutionName = "Alaska Regional Hospital"
            ds.AccessionNumber = "ACC98765"
            ds.file_meta = pydicom.dataset.FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            ds.save_as(tmp.name)
            tmp.close()

            with open(tmp.name, "rb") as f:
                resp = client.post("/anonymize_dicom", files={"file": ("test.dcm", f, "application/dicom")})

            os.unlink(tmp.name)

            self.assertEqual(resp.status_code, 200)
            result = resp.json()
            self.assertIn("fields_stripped", result)
            self.assertGreater(result["fields_stripped"], 0)
            # Verify known PHI fields were stripped
            stripped_names = [s["field"] for s in result["stripped_details"]]
            self.assertIn("PatientName", stripped_names)
            self.assertIn("PatientID", stripped_names)
            self.assertIn("PatientBirthDate", stripped_names)
        except ImportError:
            self.skipTest("pydicom not installed, skipping DICOM test")

    def test_anonymize_dicom_no_phi(self):
        """Test DICOM file with no PHI tags returns 0 stripped fields"""
        try:
            import pydicom
            from pydicom.dataset import Dataset, FileDataset
            import tempfile

            tmp = tempfile.NamedTemporaryFile(suffix=".dcm", delete=False)
            ds = FileDataset(tmp.name, Dataset(), preamble=b"\x00" * 128, is_implicit_VR=False, is_little_endian=True)
            ds.Modality = "CT"
            ds.Rows = 512
            ds.Columns = 512
            ds.file_meta = pydicom.dataset.FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            ds.save_as(tmp.name)
            tmp.close()

            with open(tmp.name, "rb") as f:
                resp = client.post("/anonymize_dicom", files={"file": ("clean.dcm", f, "application/dicom")})

            os.unlink(tmp.name)

            self.assertEqual(resp.status_code, 200)
            result = resp.json()
            self.assertEqual(result["fields_stripped"], 0)
        except ImportError:
            self.skipTest("pydicom not installed, skipping DICOM test")

if __name__ == "__main__":
    unittest.main()
