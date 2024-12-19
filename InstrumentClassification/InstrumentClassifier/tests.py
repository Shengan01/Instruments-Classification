import os

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from django.urls import reverse


class UploadTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_index_page(self):
        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "index.html")

    def test_upload_valid_wav_file(self):
        with open("test.wav", "wb") as f:
            f.write(os.urandom(1024))

        with open("test.wav", "rb") as f:
            response = self.client.post(reverse("upload"), {"file": f})
            self.assertEqual(response.status_code, 200)
            self.assertIn("predictions", response.json())

        os.remove("test.wav")

    def test_upload_valid_mp3_file(self):
        with open("test.mp3", "wb") as f:
            f.write(os.urandom(1024))

        with open("test.mp3", "rb") as f:
            response = self.client.post(reverse("upload"), {"file": f})
            self.assertEqual(response.status_code, 200)
            self.assertIn("predictions", response.json())

        os.remove("test.mp3")

    def test_upload_invalid_file_format(self):
        with open("test.txt", "wb") as f:
            f.write(b"This is a test file.")

        with open("test.txt", "rb") as f:
            response = self.client.post(reverse("upload"), {"file": f})
            self.assertEqual(response.status_code, 400)
            self.assertIn("error", response.json())

        os.remove("test.txt")

    def test_upload_file_too_large(self):
        large_file_size = 800 * 1024 * 1024 + 1
        with open("large_test.wav", "wb") as f:
            f.write(os.urandom(large_file_size))

        with open("large_test.wav", "rb") as f:
            response = self.client.post(reverse("upload"), {"file": f})
            self.assertEqual(response.status_code, 400)
            self.assertIn("error", response.json())
            self.assertEqual(response.json()["error"], "File too large")

        os.remove("large_test.wav")
