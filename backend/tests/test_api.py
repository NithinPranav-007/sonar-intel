"""
FastAPI End-to-End API Route Tests.
"""

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_survey_upload_and_pipeline():
    # 1. Create in-memory synthetic test image
    img = np.ones((500, 500, 3), dtype=np.uint8) * 110
    img[150:200, 150:200] = 245  # highlight target
    img[150:200, 100:150] = 15   # shadow
    _, encoded = cv2.imencode(".png", img)
    image_bytes = encoded.tobytes()

    # Create in-memory synthetic navigation CSV
    nav_content = (
        "ping_id,timestamp,latitude,longitude,heading,altitude,range\n"
        "1,2026-08-31T14:00:00Z,11.2345,76.5432,35.0,10.0,50.0\n"
        "2,2026-08-31T14:00:01Z,11.2346,76.5433,35.0,10.0,50.0\n"
    ).encode("utf-8")

    # 2. Upload survey
    files = {
        "sonar_file": ("test_sonar.png", image_bytes, "image/png"),
        "nav_file": ("test_nav.csv", nav_content, "text/csv")
    }
    data = {"survey_id_override": "TEST_SURV_01"}

    upload_res = client.post("/api/surveys/upload", files=files, data=data)
    assert upload_res.status_code == 201
    survey_info = upload_res.json()
    assert survey_info["survey_id"] == "TEST_SURV_01"
    assert survey_info["has_navigation"] is True

    # 3. Trigger Analysis
    analyze_res = client.post("/api/surveys/TEST_SURV_01/analyze", json={"confidence_threshold": 0.20})
    assert analyze_res.status_code == 200
    analysis = analyze_res.json()
    assert "contacts" in analysis
    assert len(analysis["contacts"]) > 0

    first_contact = analysis["contacts"][0]
    contact_id = first_contact["contact_id"]
    assert "confidence" in first_contact
    assert "priority" in first_contact
    assert "bbox" in first_contact

    # 4. Get Contacts
    contacts_res = client.get("/api/surveys/TEST_SURV_01/contacts")
    assert contacts_res.status_code == 200
    assert len(contacts_res.json()) >= 1

    # 5. Submit Human Review
    review_res = client.post(f"/api/contacts/{contact_id}/review", json={
        "review_status": "CONFIRMED",
        "review_note": "Acoustic return confirms anthropogenic target."
    })
    assert review_res.status_code == 200
    reviewed_contact = review_res.json()
    assert reviewed_contact["review_status"] == "CONFIRMED"
    assert reviewed_contact["review_note"] == "Acoustic return confirms anthropogenic target."

    # 6. Check GeoJSON Export
    geojson_res = client.get("/api/surveys/TEST_SURV_01/geojson")
    assert geojson_res.status_code == 200
    geojson_data = geojson_res.json()
    assert geojson_data["type"] == "FeatureCollection"
    assert len(geojson_data["features"]) >= 1

    # 7. Check Summary
    summary_res = client.get("/api/surveys/TEST_SURV_01/summary")
    assert summary_res.status_code == 200
    summary_data = summary_res.json()
    assert summary_data["total_contacts"] >= 1
    assert summary_data["reviewed_count"] >= 1

    # 8. Check CSV Export
    csv_res = client.get("/api/surveys/TEST_SURV_01/csv")
    assert csv_res.status_code == 200
    assert "contact_id,survey_id" in csv_res.text
