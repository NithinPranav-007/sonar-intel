# DRISHTI YOLOv8s Detector Integration — Walkthrough

## Summary of Completed Work

The pretrained **DRISHTI YOLOv8s** side-scan sonar detector has been integrated into the SONAR-INTEL project architecture as the primary reproducible baseline. The integration satisfies all requirements:
- Model weights (`best_detector.pt`) are used without modification.
- Model-specific preprocessing (`drishti-prep-v1`: vectorized Lee speckle filtering + CLAHE) is applied during inference.
- Downstream product class policy filters `crab_pot` while eligible classes (`submarine_pipeline`, `shipwreck`, `ghost_net`, `mine_cylinder`) generate canonical contacts.
- Strict confidence separation is enforced (`model_score`, `calibrated_confidence`, `data_quality`, `acoustic_context`, `priority`).
- Standalone FastAPI inference endpoint `POST /api/inference/detect` is available.
- 4 comprehensive test suites (19 test cases) pass with 100% success.
- End-to-end 8-step MVP pipeline test passes with 100% success.

---

## 1. Key Components Implemented

### Configuration Layer
- **[config.py](backend/app/core/config.py)**:
  Centralizes model path (`ml/models/dristri/best_detector.pt`), architecture parameters, preprocessing hyperparameters, and product class policies (`FILTERED_CLASSES = ["crab_pot"]`).

### Preprocessing Layer
- **[filters.py](ml/preprocessing/filters.py)**:
  Vectorized OpenCV box-filter implementation of the **Lee speckle noise filter**:
  $$\text{weight} = \max\left(0, \frac{\text{Var}(I) - \sigma_{\text{noise}}^2}{\text{Var}(I) + \epsilon}\right)$$
- **[drishti_preprocess.py](ml/preprocessing/drishti_preprocess.py)**:
  `drishti-prep-v1`: Grayscale percentile stretch (1%–99%) $\rightarrow$ Lee Filter ($5 \times 5$ window) $\rightarrow$ CLAHE ($2.0$ clip limit, $8 \times 8$ grid) $\rightarrow$ 3-channel BGR. Never mutates the raw source image.

### Decoupled Detector & Internal Schema
- **[drishti_detector.py](ml/inference/drishti_detector.py)**:
  - `DrishtiDetector`: Singleton process-level model caching (`_model_cache`), automated CUDA/CPU selection, preprocessing, inference execution, bounding box clamping, and class policy filtering.
  - `DrishtiDetection`: Clean model-agnostic dataclass schema.

### Canonical Contact Transformation & Confidence Separation
- **[contact.py](backend/app/schemas/contact.py)**:
  Updated canonical `Contact` schema with explicit `model_score` (raw YOLO confidence), `calibrated_confidence` (None until calibrated), `detection_timestamp`, and `review_status="AI_CANDIDATE"`.
- **[transformer.py](backend/app/services/transformer.py)**:
  Adapts `DrishtiDetection` $\rightarrow$ `Contact`. Enforces product filtering so `crab_pot` detections never generate production contacts, attaches acoustic context scores, operational priority, and WGS-84 coordinates without synthetic fabrication.

### Backend Orchestration & FastAPI Endpoint
- **[inference_service.py](backend/app/services/inference_service.py)**:
  Orchestrates swath tiling ($640 \times 640$ with 20% overlap), tile-based DRISHTI inference, NMS deduplication, acoustic context analysis, geolocation, and contact transformation.
- **[inference.py](backend/app/api/inference.py)**:
  `POST /api/inference/detect`: Accepts image uploads via `multipart/form-data`, executes DRISHTI detector, and returns standardized detection JSON.
- **[main.py](backend/app/main.py)**:
  Mounted `inference_router` under `/api/inference`.

### Baseline Documentation
- **[DRISHTI_BASELINE.md](docs/models/DRISHTI_BASELINE.md)**:
  Documents SHA256 (`2f55eec5d8fe6b4737706392e259c02660a8542cddbcbd603f96d606c54cb927`), HuggingFace provenance, classes, product filtering policy, preprocessing specifications, and failure analysis taxonomy.

---

## 2. Verification Results

### Automated Pytest Suite (19/19 Passing)
Command:
```powershell
.\.venv\Scripts\pytest.exe tests/ -v
```

Output:
```
tests/test_contact_transformation.py::TestContactTransformation::test_filtered_classes_are_excluded_from_contacts PASSED
tests/test_contact_transformation.py::TestContactTransformation::test_no_coordinate_fabrication PASSED
tests/test_contact_transformation.py::TestContactTransformation::test_coordinate_attachment_when_navigation_valid PASSED
tests/test_contact_transformation.py::TestContactTransformation::test_priority_assignment_rules PASSED
tests/test_drishti_detector.py::TestDrishtiDetector::test_model_loads_successfully PASSED
tests/test_drishti_detector.py::TestDrishtiDetector::test_expected_class_mapping PASSED
tests/test_drishti_detector.py::TestDrishtiDetector::test_model_cached_per_process PASSED
tests/test_drishti_detector.py::TestDrishtiDetector::test_inference_on_real_sonar_imagery PASSED
tests/test_drishti_detector.py::TestDrishtiDetector::test_empty_detection_handling PASSED
tests/test_drishti_detector.py::TestDrishtiDetector::test_crab_pot_is_tagged_as_filtered PASSED
tests/test_drishti_preprocessing.py::TestDrishtiPreprocessing::test_lee_filter_preserves_dimensions_and_dtype PASSED
tests/test_drishti_preprocessing.py::TestDrishtiPreprocessing::test_lee_filter_noise_suppression PASSED
tests/test_drishti_preprocessing.py::TestDrishtiPreprocessing::test_lee_filter_invalid_window_size PASSED
tests/test_drishti_preprocessing.py::TestDrishtiPreprocessing::test_drishti_preprocess_immutability PASSED
tests/test_drishti_preprocessing.py::TestDrishtiPreprocessing::test_drishti_preprocess_deterministic PASSED
tests/test_drishti_preprocessing.py::TestDrishtiPreprocessing::test_drishti_preprocess_handles_empty_image PASSED
tests/test_inference_api.py::TestInferenceAPI::test_detect_endpoint_valid_image PASSED
tests/test_inference_api.py::TestInferenceAPI::test_detect_endpoint_rejects_non_image PASSED
tests/test_inference_api.py::TestInferenceAPI::test_detect_endpoint_rejects_empty_file PASSED

======================= 19 passed in 11.21s =======================
```

### End-to-End Pipeline Test (8/8 Steps Passing)
Command:
```powershell
.\.venv\Scripts\python.exe scripts/e2e_mvp_test.py
```

Output:
```
[Step 1] Probing API Health Probe (/api/health)...
  --> Health probe PASSED
[Step 2] Ingesting Real Sonar Swath & Navigation...
  --> Ingestion PASSED: Survey ID = SURV_20260902_214759 | Resolution = 1280x1800
[Step 3] Executing Real Inference via API...
  --> Analysis PASSED: 3 contacts discovered in 5520.2 ms
  --> Sample Contact: ID=C001, Class=ghost_net, Conf=0.69, Priority=HIGH
      Model Version: baseline-v1
      Location: Lat=11.235591, Lon=76.543735 (ESTIMATED)
[Step 4] Verifying Database Records in Storage...
  --> Database persistence VERIFIED: 3 records stored.
[Step 5] Fetching GeoJSON (/api/surveys/.../geojson)...
  --> GeoJSON export VERIFIED: 3 spatial features returned.
[Step 6] Testing Search Functionality (/api/contacts/search)...
  --> Contact search VERIFIED: Successfully queried C001
[Step 7] Testing Human Triage Review on C001...
  --> Human review VERIFIED: Status updated to CONFIRMED
[Step 8] Fetching Survey Summary (/api/surveys/.../summary)...
  --> Summary API VERIFIED: Total=3 | High=3 | Reviewed=1

=================================================================
ALL 8 END-TO-END PIPELINE VALIDATION STEPS PASSED SUCCESSFULLY!
=================================================================
```
