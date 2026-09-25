"""
Integration Tests for Standalone FastAPI Inference Endpoint.
"""

import io
import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2

from backend.app.main import app


class TestInferenceAPI:
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def sample_png_bytes(self):
        """Generates a valid encoded PNG image in bytes."""
        img = np.full((640, 640, 3), 128, dtype=np.uint8)
        # Draw synthetic feature
        cv2.rectangle(img, (200, 200), (350, 300), (255, 255, 255), -1)
        _, encoded = cv2.imencode(".png", img)
        return encoded.tobytes()

    def test_detect_endpoint_valid_image(self, client, sample_png_bytes):
        response = client.post(
            "/api/inference/detect",
            files={"file": ("test_sonar.png", io.BytesIO(sample_png_bytes), "image/png")}
        )
        assert response.status_code == 200
        data = response.json()

        assert "model_name" in data and data["model_name"] == "DRISHTI-YOLOv8s"
        assert "model_version" in data and data["model_version"] == "baseline-v1"
        assert "image" in data
        assert data["image"]["width"] == 640
        assert data["image"]["height"] == 640
        assert "detections" in data
        assert isinstance(data["detections"], list)
        assert "filtered_detections_count" in data

        for det in data["detections"]:
            assert "class_name" in det
            assert "confidence" in det
            assert "bbox" in det
            assert len(det["bbox"]) == 4
            assert det["review_status"] == "AI_CANDIDATE"

    def test_detect_endpoint_rejects_non_image(self, client):
        text_content = b"This is a text file, not sonar imagery."
        response = client.post(
            "/api/inference/detect",
            files={"file": ("notes.txt", io.BytesIO(text_content), "text/plain")}
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_detect_endpoint_rejects_empty_file(self, client):
        empty_content = b""
        response = client.post(
            "/api/inference/detect",
            files={"file": ("empty.png", io.BytesIO(empty_content), "image/png")}
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
