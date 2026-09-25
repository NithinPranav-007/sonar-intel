# SONAR-INTEL: Inference Smoke Test Report

**Model:** Frozen YOLOv8n (`outputs/models/yolov8n_sonar_baseline/best.pt`)  
**Execution Timestamp:** 2026-09-01 18:58:56  
**Hardware Device:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.6)  
**Test Image Source:** `data\demo\sonar\survey_001_raw.png` (1280 &times; 1800 px)  
**Navigation Metadata:** `data\demo\navigation\survey_001_nav.csv`  
**FastAPI Status:** HTTP 200 OK  

---

## 1. Latency & Resource Benchmarks

| Stage | Execution Time | Notes |
| :--- | :--- | :--- |
| **Model Weight Loading** | `114.89 ms` | Loaded into GPU VRAM (`best.pt`) |
| **Preprocessing & Tiling** | `131.88 ms` | 1–99% swath normalization (CLAHE=False), 12 640x640 tiles |
| **YOLO Batched Inference** | `4787.19 ms` | PyTorch AMP FP16, 12 tiles |
| **Postprocessing & Context** | `47.16 ms` | Spatial deduplication, shadow/context scores, geolocation |
| **Total Inference Latency** | **4966.23 ms** | End-to-end swath processing |
| **Peak GPU VRAM Usage** | **41 MB** | Under 1.2 GB dedicated VRAM footprint |

---

## 2. API Contract & Schema Verification

- **Endpoint:** `POST /api/surveys/SURVEY-SMOKE-TEST/analyze`
- **Output Schema:** Canonical `Contact` data model
- **Model Version Provenance:** `yolov8n-sonar-baseline`
- **Total Anomaly Contacts Returned:** `11`
- **Localization Handling:**
  - Latitude / Longitude: `11.234695, 76.543647`
  - Localization Status: `ESTIMATED` (Honest estimation label)

---

## 3. Acoustic Heuristics Clarification

- **Confidence:** Raw detector output confidence from YOLOv8n.
- **Shadow Evidence Score:** Heuristic ratio measuring down-range backscatter intensity depression behind the target highlight. Not a probability.
- **Context Score:** Geometric and physics consistency score combining highlight elevation and shadow geometry. Not a probability.
- **Data Quality Score:** Global swath signal-to-noise and intensity dynamic range ratio.

---

## 4. Verification Conclusion

**SMOKE TEST: PASSED**  
The frozen model, preprocessing pipeline, coordinate estimator, and FastAPI REST endpoint executed seamlessly without synthetic coordinates, CLAHE artifacts, or schema violations.
