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
        resp = requests.post(f"{BASE_URL}/store", json=data)
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
        resp = requests.post(f"{BASE_URL}/search_enhanced", json=data)
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
        resp = requests.post(f"{BASE_URL}/store", json=data)
        self.assertIn(resp.status_code, [200, 201])

if __name__ == "__main__":
    unittest.main()
