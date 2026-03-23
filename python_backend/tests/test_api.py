import unittest
import requests

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
        with open("tests/test.jpg", "rb") as img:
            files = {"file": ("test.jpg", img, "image/jpeg")}
            resp = requests.post(f"{BASE_URL}/anonymize_image", files=files)
            print(f"Response status: {resp.status_code}")
            print(f"Response text: {resp.text}")
            self.assertEqual(resp.status_code, 200)

if __name__ == "__main__":
    unittest.main()
