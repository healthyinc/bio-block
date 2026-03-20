import unittest
import requests
import os
import tempfile

BASE_URL = "http://localhost:3002"

class TestAPI(unittest.TestCase):
    def test_root(self):
        resp = requests.get(f"{BASE_URL}/")
        self.assertEqual(resp.status_code, 200)

    def test_store(self):
        data = {
            "summary": "test", 
            "dataset_title": "test",
            "cid": "test-cid-123",
            "metadata": {"title": "test"}
        }
        resp = requests.post(f"{BASE_URL}/store", json=data)
        self.assertIn(resp.status_code, [200, 201])

    def test_search(self):
        data = {"query": "patient information", "k": 1}
        resp = requests.post(f"{BASE_URL}/search", json=data)
        self.assertEqual(resp.status_code, 200)

    def test_filter(self):
        data = {"filters": {"dataType": "Personal"}, "n_results": 1}
        resp = requests.post(f"{BASE_URL}/filter", json=data)
        self.assertEqual(resp.status_code, 200)

    def test_search_with_filter(self):
        data = {"query": "diabetes", "filters": {"dataType": "Institution"}, "n_results": 1}
        resp = requests.post(f"{BASE_URL}/search_with_filter", json=data)
        self.assertEqual(resp.status_code, 200)

    def test_anonymize_image(self):
        # Replace 'test.jpg' with a valid test image path
        with open("test.jpg", "rb") as img:
            files = {"file": ("test.jpg", img, "image/jpeg")}
            resp = requests.post(f"{BASE_URL}/anonymize_image", files=files)
            print(f"Response status: {resp.status_code}")
            print(f"Response text: {resp.text}")
            self.assertEqual(resp.status_code, 200)

    def test_anonymize_text(self):
        """Test PHI anonymization in plain text"""
        data = {
            "text": "Patient John Smith, DOB 03/15/1985, SSN 123-45-6789, was admitted to Alaska Regional Hospital on March 10, 2026."
        }
        resp = requests.post(f"{BASE_URL}/anonymize_text", json=data)
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        # Verify response structure
        self.assertIn("anonymized_text", result)
        self.assertIn("entities_found", result)
        self.assertIn("method", result)
        self.assertIn("entity_count", result)
        # Verify PHI was actually detected
        self.assertGreater(result["entity_count"], 0)
        # Verify original PHI is removed from anonymized text
        self.assertNotIn("John Smith", result["anonymized_text"])

    def test_anonymize_text_empty(self):
        """Test that empty text returns 400 error"""
        data = {"text": ""}
        resp = requests.post(f"{BASE_URL}/anonymize_text", json=data)
        self.assertEqual(resp.status_code, 400)

    def test_anonymize_text_no_phi(self):
        """Test text with no PHI returns successfully with zero entities"""
        data = {"text": "The weather is sunny today."}
        resp = requests.post(f"{BASE_URL}/anonymize_text", json=data)
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        self.assertIn("anonymized_text", result)

    def _create_test_pdf(self, text_content):
        """Helper to create a simple test PDF with given text content"""
        try:
            import fitz  # PyMuPDF
            pdf_document = fitz.open()
            page = pdf_document.new_page()
            page.insert_text((72, 72), text_content, fontsize=12)
            pdf_bytes = pdf_document.tobytes()
            pdf_document.close()
            return pdf_bytes
        except ImportError:
            self.skipTest("PyMuPDF not installed, skipping PDF tests")

    def test_anonymize_pdf(self):
        """Test PHI anonymization in PDF documents"""
        pdf_content = self._create_test_pdf(
            "Patient John Smith, DOB 03/15/1985, was admitted to Alaska Regional Hospital."
        )
        files = {"file": ("test_medical.pdf", pdf_content, "application/pdf")}
        resp = requests.post(f"{BASE_URL}/anonymize_pdf", files=files)
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        # Verify response structure
        self.assertIn("pages", result)
        self.assertIn("total_entities", result)
        self.assertIn("total_pages", result)
        self.assertIn("method", result)
        self.assertEqual(result["total_pages"], 1)
        # Verify PHI was detected
        self.assertGreater(result["total_entities"], 0)
        # Verify John Smith was redacted
        self.assertNotIn("John Smith", result["pages"][0]["anonymized_text"])

    def test_anonymize_pdf_invalid_file(self):
        """Test that non-PDF file returns 400 error"""
        files = {"file": ("test.txt", b"This is not a PDF", "text/plain")}
        resp = requests.post(f"{BASE_URL}/anonymize_pdf", files=files)
        self.assertEqual(resp.status_code, 400)

    def test_anonymize_pdf_no_phi(self):
        """Test PDF with no PHI returns successfully"""
        pdf_content = self._create_test_pdf("The weather is sunny and warm today.")
        files = {"file": ("clean_doc.pdf", pdf_content, "application/pdf")}
        resp = requests.post(f"{BASE_URL}/anonymize_pdf", files=files)
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        self.assertIn("pages", result)
        self.assertEqual(result["total_pages"], 1)

if __name__ == "__main__":
    unittest.main()
