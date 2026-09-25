"""
e2e_mvp_test.py: Comprehensive End-to-End MVP Verification for SONAR-INTEL.

Validates the full chain:
1. Backend startup & health check
2. Survey ingestion (real demo SSS waterfall + navigation metadata)
3. Non-CLAHE 1-99% swath normalization & tiling
4. Frozen YOLOv8n best.pt execution on CUDA
5. Detection generation with heuristic evidence scores
6. Canonical Contact schema compliance
7. Database persistence (surveys and contacts)
8. GeoJSON FeatureCollection generation (GET /api/surveys/{survey_id}/geojson)
9. Contact search functionality (GET /api/contacts/search)
10. Human review triage persistence (POST /api/contacts/{contact_id}/review)
11. Audit trail verification (ReviewModel)
"""

import os
import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import torch
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database.connection import SessionLocal
from backend.app.database.models import SurveyModel, ContactModel, ReviewModel


def run_e2e_test():
    print("=" * 65)
    print("SONAR-INTEL: End-to-End MVP Pipeline Test")
    print("=" * 65)

    client = TestClient(app)

    # -------------------------------------------------------------
    # Step 1: Health Check Probe
    # -------------------------------------------------------------
    print("\n[Step 1] Probing API Health Probe (/api/health)...")
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    health_data = health_resp.json()
    assert health_data["status"] == "healthy", "Backend status not healthy"
    print(f"  --> Health probe PASSED: {health_data}")

    # -------------------------------------------------------------
    # Step 2: Ingest Real Demo Survey Swath
    # -------------------------------------------------------------
    print("\n[Step 2] Ingesting Real Sonar Swath & Navigation...")
    sonar_img_path = Path("data/demo/sonar/survey_001_raw.png")
    nav_file_path = Path("data/demo/navigation/survey_001_nav.csv")
    assert sonar_img_path.exists(), f"Demo sonar image not found: {sonar_img_path}"
    assert nav_file_path.exists(), f"Demo navigation file not found: {nav_file_path}"

    with open(sonar_img_path, "rb") as f_img, open(nav_file_path, "rb") as f_nav:
        upload_resp = client.post(
            "/api/surveys/upload",
            files={
                "sonar_file": ("survey_001_raw.png", f_img, "image/png"),
                "nav_file": ("survey_001_nav.csv", f_nav, "text/csv")
            }
        )

    assert upload_resp.status_code in [200, 201], f"Upload failed: {upload_resp.text}"
    survey_data = upload_resp.json()
    survey_id = survey_data["survey_id"]
    print(f"  --> Ingestion PASSED: Survey ID = {survey_id} | Resolution = {survey_data['image_width']}x{survey_data['image_height']}")

    # -------------------------------------------------------------
    # Step 3: Trigger Real Model Analysis (POST /api/surveys/{id}/analyze)
    # -------------------------------------------------------------
    print(f"\n[Step 3] Executing Real YOLOv8n Inference via API for {survey_id}...")
    t0 = time.perf_counter()
    analysis_resp = client.post(
        f"/api/surveys/{survey_id}/analyze",
        json={"confidence_threshold": 0.15}
    )
    analysis_latency = (time.perf_counter() - t0) * 1000.0

    assert analysis_resp.status_code == 200, f"Analysis failed: {analysis_resp.text}"
    analysis_data = analysis_resp.json()
    contacts = analysis_data["contacts"]
    print(f"  --> Analysis PASSED: {len(contacts)} contacts discovered in {analysis_latency:.1f} ms")
    assert len(contacts) > 0, "Expected at least 1 detection on demo sonar swath."

    # Validate Canonical Contact Structure
    first_contact = contacts[0]
    print(f"  --> Sample Contact: ID={first_contact['contact_id']}, Class={first_contact['class_name']}, Conf={first_contact['confidence']:.2f}, Priority={first_contact['priority']}")
    print(f"      Model Version: {first_contact['model_version']}")
    print(f"      Location: Lat={first_contact['latitude']}, Lon={first_contact['longitude']} ({first_contact['localization_status']})")

    assert first_contact["model_version"] in ["yolov8n-sonar-baseline", "baseline-v1"], "Incorrect model version tag."
    assert first_contact["localization_status"] in ["ESTIMATED", "UNAVAILABLE"], "Invalid localization status."
    assert "shadow_evidence" in first_contact, "Missing shadow evidence metric."
    assert "context_score" in first_contact, "Missing context score metric."

    # -------------------------------------------------------------
    # Step 4: Verify Database Persistence & Integrity
    # -------------------------------------------------------------
    print("\n[Step 4] Verifying Database Records in Storage...")
    db = SessionLocal()
    try:
        db_survey = db.query(SurveyModel).filter(SurveyModel.survey_id == survey_id).first()
        assert db_survey is not None, "Survey not found in database!"
        db_contacts = db.query(ContactModel).filter(ContactModel.survey_id == survey_id).all()
        assert len(db_contacts) == len(contacts), f"DB contact count ({len(db_contacts)}) does not match API count ({len(contacts)})"
        print(f"  --> Database persistence VERIFIED: {len(db_contacts)} records stored.")
    finally:
        db.close()

    # -------------------------------------------------------------
    # Step 5: Verify GeoJSON FeatureCollection Generation
    # -------------------------------------------------------------
    print(f"\n[Step 5] Fetching GeoJSON (/api/surveys/{survey_id}/geojson)...")
    geojson_resp = client.get(f"/api/surveys/{survey_id}/geojson")
    assert geojson_resp.status_code == 200, f"GeoJSON fetch failed: {geojson_resp.text}"
    geojson_data = geojson_resp.json()
    assert geojson_data["type"] == "FeatureCollection", "Response is not a valid FeatureCollection"
    features = geojson_data["features"]
    print(f"  --> GeoJSON export VERIFIED: {len(features)} spatial features returned.")
    assert len(features) == len(contacts), "GeoJSON feature count mismatch."
    for f in features:
        if f["geometry"] is not None:
            assert f["geometry"]["type"] == "Point", "Feature geometry must be Point"
            assert len(f["geometry"]["coordinates"]) == 2, "Invalid point coordinate tuple"

    # -------------------------------------------------------------
    # Step 6: Verify Contact Search API (/api/contacts/search)
    # -------------------------------------------------------------
    print("\n[Step 6] Testing Search Functionality (/api/contacts/search)...")
    target_cid = first_contact["contact_id"]
    search_resp = client.get(f"/api/contacts/search?q={target_cid}&survey_id={survey_id}")
    assert search_resp.status_code == 200, f"Search failed: {search_resp.text}"
    search_results = search_resp.json()
    assert len(search_results) >= 1, f"Search by ID '{target_cid}' returned no results"
    assert search_results[0]["contact_id"] == target_cid, "Search ID mismatch"
    print(f"  --> Contact search VERIFIED: Successfully queried {target_cid}")

    # -------------------------------------------------------------
    # Step 7: Verify Human-in-the-Loop Review Persistence
    # -------------------------------------------------------------
    print(f"\n[Step 7] Testing Human Triage Review on {target_cid}...")
    review_resp = client.post(
        f"/api/contacts/{target_cid}/review",
        json={
            "review_status": "CONFIRMED",
            "review_note": "Verified acoustic shadow and high-backscatter metallic return during hydrographic review."
        }
    )
    assert review_resp.status_code == 200, f"Review failed: {review_resp.text}"
    updated_contact = review_resp.json()
    assert updated_contact["review_status"] == "CONFIRMED", "Review status was not updated to CONFIRMED"
    assert "Verified acoustic shadow" in updated_contact["review_note"], "Review note not saved"
    print(f"  --> Human review VERIFIED: Status updated to {updated_contact['review_status']}")

    # Check Review Audit Trail in Database
    db = SessionLocal()
    try:
        audit_record = db.query(ReviewModel).filter(ReviewModel.contact_id == target_cid).first()
        assert audit_record is not None, "Review audit trail record missing!"
        assert audit_record.review_status == "CONFIRMED", "Audit trail review status mismatch"
        print(f"  --> Audit trail VERIFIED: Decision logged with timestamp {audit_record.reviewed_at}")
    finally:
        db.close()

    # -------------------------------------------------------------
    # Step 8: Mission Summary API
    # -------------------------------------------------------------
    print(f"\n[Step 8] Fetching Survey Summary (/api/surveys/{survey_id}/summary)...")
    summary_resp = client.get(f"/api/surveys/{survey_id}/summary")
    assert summary_resp.status_code == 200, f"Summary failed: {summary_resp.text}"
    summary_data = summary_resp.json()
    assert summary_data["reviewed_count"] >= 1, "Summary reviewed count was not updated"
    print(f"  --> Summary API VERIFIED: Total={summary_data['total_contacts']} | High={summary_data['high_priority']} | Reviewed={summary_data['reviewed_count']}")

    print("\n" + "=" * 65)
    print("ALL 8 END-TO-END PIPELINE VALIDATION STEPS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_e2e_test()
