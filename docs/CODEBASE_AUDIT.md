# Pre-Training Codebase Audit & Repository Map

**Project:** SONAR-INTEL  
**Audit Date:** 2026-09-01  
**Scope:** Pre-Training Codebase Inventory, Dataset Verification, Cleanup, and Hardware Readiness  
**Hardware Environment:** NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM, CC 8.6, CUDA 12.6, PyTorch 2.13.0+cu126)  

---

## 1. Current Architecture

SONAR-INTEL operates on a modular 3-tier architecture:

1. **ML Pipeline & Training Tier (`ml/`):**
   - **Offline Preprocessing:** Swath-level 1–99% percentile normalization, 640×640 tiling with 20% overlap, hybrid mask-to-YOLO conversion (Class 0), and site-aware zero-leakage dataset partitioning (70/15/15).
   - **Training Subsystem:** Low-VRAM YOLOv8n single-class baseline detector (`ml/training/train_yolov8n.py`) strictly targeting `data/interim/yolo_split/` via `ml/training/dataset.yaml`.
   - **Inference Modules (`ml/inference/`):** Modular detector (`detector.py`), deduplication (`postprocess.py`), acoustic context extraction (`context.py`), and scoring (`scoring.py`).
2. **Backend API & Service Tier (`backend/`):**
   - FastAPI REST API (`backend/app/main.py`) providing survey upload, automated inference triggering, contact management, human verification workflows, and GeoJSON / CSV export.
   - Dual-persistence architecture: PostGIS with GeoAlchemy2 + SQLite fallback (`sonar_intel_fallback.db`).
3. **Frontend Presentation Tier (`frontend/`):**
   - React 18 + Vite + TypeScript single-page application with MapLibre GL for side-scan sonar contact visualization and reviewer workflows.

---

## 2. Current Directory Structure

```
SONAR-INTEL/
├── .venv/                         # CUDA-enabled Python virtual environment (PyTorch 2.13+cu126, Ultralytics 8.4.137)
├── archive/                       # Archived legacy & unreferenced artifacts
│   ├── README.md
│   └── weights/yolo26n.pt        # Archived 80-class COCO checkpoint (5.5 MB)
├── backend/                       # FastAPI backend service
│   ├── app/
│   │   ├── api/                   # REST endpoints (upload, analysis, contacts, review, reports)
│   │   ├── database/              # SQLAlchemy / GeoAlchemy2 models and repositories
│   │   ├── schemas/               # Pydantic data schemas
│   │   ├── services/              # Geolocation, inference orchestration, sonar services
│   │   └── utils/                 # GeoJSON formatting utilities
│   ├── tests/                     # API and pipeline integration tests
│   ├── requirements.txt           # Backend dependencies
│   └── sonar_intel_fallback.db    # SQLite fallback database for local MVP
├── data/
│   ├── demo/                      # Demonstration sonar swaths and navigation CSVs
│   ├── interim/                   # Preprocessed and partitioned dataset
│   │   ├── clahe/                 # Research ablation: CLAHE enhanced swaths
│   │   ├── denoised/              # Research ablation: 2D-FFT filtered swaths
│   │   ├── inspection/            # Swath inspection artifacts
│   │   ├── normalized/            # 1-99% swath-level percentile normalized images
│   │   ├── quality_checked/       # Data quality audit outputs
│   │   ├── tiled/                 # 640x640 image and mask tiles (20% overlap)
│   │   ├── yolo/                  # Full interim YOLO images and annotations (8,356)
│   │   └── yolo_split/            # Site-separated partitions (train: 5844, val: 1256, test: 1256)
│   ├── processed/                 # Runtime generated test outputs (.gitkeep)
│   └── raw/                       # Immutable raw AI4Shipwrecks dataset
├── database/                      # PostGIS schema and seed scripts
├── docs/                          # Architecture, contract, dataset, and preprocessing documentation
│   ├── dataset/                   # Annotation strategy, site split, target size distribution reports
│   └── preprocessing/             # Pipeline status and experimental reports
├── ml/
│   ├── inference/                 # Pure ML inference routines (detector, context, scoring)
│   ├── models/                    # Canonical base weights (ml/models/yolov8n.pt)
│   ├── preprocessing/             # Dataset generation scripts (01-09) and runtime modules
│   └── training/                  # Training script (train_yolov8n.py) and dataset.yaml
├── outputs/                       # Audited research reports, metrics, and smoke test artifacts
│   ├── training/smoke_test/       # Validated 1-epoch GPU micro-training run & samples
│   └── *.json, *.csv              # Experiment reports (quality, tiling, yolo conversion, split)
├── scratch/                       # Temporary workspace scratch directory (.gitkeep)
├── scripts/                       # Analytical scripts (analyze_annotations.py, gpu_smoke_test.py)
├── yolov8n.pt                     # Base COCO YOLOv8n weights for root execution (6.5 MB)
└── docker-compose.yml             # Local PostGIS container configuration
```

---

## 3. Files Kept (Validated Assets)

- **Dataset Assets (`data/interim/yolo_split/`):** All 8,356 tiles (5,844 train, 1,256 val, 1,256 test) and matching label files.
- **Dataset Configuration:** `ml/training/dataset.yaml`.
- **Training Scripts:** `ml/training/train_yolov8n.py` (authoritative baseline trainer) and `ml/training/train.py` (CLI wrapper).
- **Evaluation Script:** `ml/training/evaluate.py`.
- **Base Model Weights:** `yolov8n.pt` (root) and `ml/models/yolov8n.pt` (referenced by `inference_service.py`).
- **Research Artifacts:** `data/interim/clahe/`, `data/interim/denoised/`, and all reports in `outputs/` (retained as documented ablation evidence).
- **Backend Fallback Database:** `backend/sonar_intel_fallback.db` (retained for MVP standalone operation).
- **Preprocessing Modules:** All numbered offline pipeline scripts and modular runtime helpers (`normalize.py`, `quality.py`, `tiling.py`, `pipeline.py`).

---

## 4. Files Archived

- `weights/yolo26n.pt`: 80-class COCO checkpoint (5.5 MB) that was unreferenced across the codebase and irrelevant to the single-class sonar anomaly detector. Moved to `archive/weights/yolo26n.pt` accompanied by `archive/README.md`.

---

## 5. Files Deleted (Safe Cleanup)

- `data/interim/yolo_split/train/labels.cache`: Stale Ultralytics label cache containing only 58 items from the 1% micro-train smoke test. Deleted to guarantee full 5,844 image indexing on full training run.
- `data/interim/yolo_split/val/labels.cache`: Stale label cache containing only 5 items from the smoke test. Deleted to guarantee full 1,256 image validation indexing.
- `runs/` (`runs/detect/outputs/training/smoke_test/micro_train/...`): Accidental duplicate run directory created during the smoke test when relative output paths were passed to Ultralytics. True duplicate of `outputs/training/smoke_test/micro_train`. Directory removed.

---

## 6. Duplicate Implementations Found

1. **Preprocessing Script vs Runtime Library:**
   - *Observation:* `03_normalize.py` vs `normalize.py`; `07_tile.py` vs `tiling.py`; `02_quality_check.py` vs `quality.py`.
   - *Finding:* NOT redundant duplicates. The numbered scripts are offline batch dataset generators for AI4Shipwrecks swaths; the unnumbered files are runtime library modules imported by `inference_service.py` and `sonar_service.py`. Both are necessary.
2. **Model Checkpoints:**
   - *Observation:* Root `yolov8n.pt` (6,549,796 bytes) and `ml/models/yolov8n.pt` (6,549,796 bytes).
   - *Finding:* Byte-for-byte identical. Root is invoked by root CLI / Ultralytics default resolution; `ml/models/yolov8n.pt` is referenced by backend inference service. Both retained to avoid breaking either execution context.
3. **Training Scripts:**
   - *Observation:* `ml/training/train.py` vs `ml/training/train_yolov8n.py`.
   - *Finding:* `train.py` is a basic CLI script. `train_yolov8n.py` is the comprehensive baseline pipeline script with strict CUDA checks, dynamic VRAM batch allocation, automated validation, test set evaluation, and metric logging. Classified `train_yolov8n.py` as authoritative.

---

## 7. Broken References Found

- No broken imports or missing module errors were detected in any of the 42 Python files across the repository.
- `README.md` and `ml/training/README.md` referenced `train.py`; documented that `train_yolov8n.py` is the production baseline trainer.
- Stale Ultralytics caches (`labels.cache`) in `yolo_split/` that could have prevented loading the full dataset were detected and cleaned.

---

## 8. Dataset Verification

Audit performed on `data/interim/yolo_split/` and `ml/training/dataset.yaml`:

| Check Parameter | Train Split | Validation Split | Test Split | Total / Overall | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Images** | 5,844 | 1,256 | 1,256 | 8,356 | **PASS** |
| **Total Labels** | 5,844 | 1,256 | 1,256 | 8,356 | **PASS** |
| **Positive Tiles** | 612 (10.47%) | 130 (10.35%) | 132 (10.51%) | 874 (10.46%) | **PASS** |
| **Negative Tiles (Empty Labels)** | 5,232 (89.53%) | 1,126 (89.65%) | 1,124 (89.49%) | 7,482 (89.54%) | **PASS** |
| **Total Bounding Boxes** | 1,034 | 195 | 271 | 1,500 | **PASS** |
| **Class ID Verification** | All `0` | All `0` | All `0` | Strictly `0` | **PASS** |
| **Normalized Coordinates [0, 1]** | 100% Valid | 100% Valid | 100% Valid | Zero Out-of-Bounds | **PASS** |
| **Image-to-Label Match** | 100% 1:1 Match | 100% 1:1 Match | 100% 1:1 Match | Zero Missing Pairs | **PASS** |
| **Duplicate Tile IDs** | 0 | 0 | 0 | 0 across all splits | **PASS** |
| **Survey Sites** | 185 unique sites | 55 unique sites | 46 unique sites | 286 total sites | **PASS** |
| **Cross-Split Site Leakage** | 0 | 0 | 0 | **ZERO LEAKAGE** | **PASS** |

---

## 9. Training Readiness

- **Hardware & CUDA:** NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) verified with CUDA 12.6 and PyTorch 2.13.0+cu126.
- **VRAM Utilization:** Smoke test with batch=16 peaked at 1,843.22 MB (45% VRAM capacity), leaving >2.25 GB headroom with zero OOM risk.
- **Dataset Path Integrity:** `dataset.yaml` correctly points to `data/interim/yolo_split`, resolving `train/images`, `val/images`, and `test/images`.
- **Ablation Isolation:** CLAHE and FFT denoised datasets are strictly excluded from `dataset.yaml` and training configurations.

---

## 10. Remaining Risks & Recommendations

1. **Path Resolution in Ultralytics:**
   - *Risk:* Passing relative paths to `project=` in Ultralytics may cause it to create outputs inside `runs/detect/`.
   - *Mitigation:* `train_yolov8n.py` already includes fallback resolution logic to locate weights if saved in `results.save_dir`.
2. **Label Cache Regeneration:**
   - *Note:* Because stale `labels.cache` files were deleted, the first epoch will spend ~15–25 seconds scanning and compiling a fresh, complete label cache for all 8,356 tiles. This is expected and desirable.
