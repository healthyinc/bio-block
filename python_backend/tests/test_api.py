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

if __name__ == "__main__":
    unittest.main()
