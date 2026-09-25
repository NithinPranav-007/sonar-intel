# Preprocessing Pipeline Status & Authoritative Path

**Project:** SONAR-INTEL  
**Document:** Authoritative Preprocessing Pipeline & Script Audit  
**Date:** 2026-09-01  
**Status:** VALIDATED PRE-TRAINING BASELINE  

---

## 1. Executive Summary & Authoritative Preprocessing Path

The authoritative pre-training dataset generation pipeline for YOLOv8n artificial anomaly detection is strictly linear, deterministic, and site-aware. 

```
[RAW AI4Shipwrecks] (Immutable, read-only: data/raw/AI4Shipwrecks/)
         │
         ▼ (Stage 01: Inspection & Stage 02: Quality Check)
[Quality-Controlled Swaths] (Flagged / Filtered: outputs/data_quality_report.csv)
         │
         ▼ (Stage 03: Normalization & Stage 07: Tiling)
[1–99% Swath Percentile Normalized 640×640 Tiles] (data/interim/tiled/)
         │
         ▼ (Stage 08: Mask-to-YOLO Hybrid Conversion)
[YOLO Format Bounding Boxes & Empty Labels] (data/interim/yolo/)
         │
         ▼ (Stage 09: Site-Aware Split, 70/15/15)
[Site-Separated Partition: Train (5,844), Val (1,256), Test (1,256)] (data/interim/yolo_split/)
         │
         ▼
[YOLOv8n Training Input via ml/training/dataset.yaml]
```

> [!IMPORTANT]
> **Ablation Safeguard:** CLAHE (Stage 04) and FFT Denoising (Stage 05) are **RESEARCH/EXPERIMENT ARTIFACTS ONLY**. They are strictly isolated in `data/interim/clahe/` and `data/interim/denoised/` and do **NOT** enter the training set.

---

## 2. Pipeline Script Classification & Status Table

| Stage | Script / Module | Classification | Status | Used for Training? | Reason / Technical Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01 Dataset Inspection** | `ml/preprocessing/01_inspect.py` | **B. RESEARCH / EXPERIMENT** | COMPLETE | **No** | Offline scan of raw swath dimensions, channel statistics, and site metadata. Read-only. |
| **02 Quality Control** | `ml/preprocessing/02_quality_check.py` | **A. REQUIRED FOR PIPELINE** | COMPLETE | **Yes (Filter)** | Evaluates along-track height, missing acoustic pings, zero-data bands, and snr. Emits `outputs/data_quality_report.csv`. |
| **03 Normalization** | `ml/preprocessing/03_normalize.py` | **A. REQUIRED FOR PIPELINE** | COMPLETE | **Yes (Methodology)** | Evaluates 1–99% swath-level percentile normalization against min-max and z-score. Approved baseline methodology. |
| **04 CLAHE** | `ml/preprocessing/04_clahe.py` | **B. RESEARCH / EXPERIMENT** | COMPLETE | **NO (Ablation Only)** | Local contrast enhancement evaluation. Amplified acoustic reverberation noise; excluded from default training baseline. |
| **05 Denoising** | `ml/preprocessing/05_denoise.py` | **B. RESEARCH / EXPERIMENT** | COMPLETE | **NO (Ablation Only)** | 2D-FFT periodic striping filter. Suppressed high-frequency boundary acoustic shadows; excluded from default training baseline. |
| **06 Geometry** | *None* | **N/A** | UNAVAILABLE | **No** | Slant-range altitude and sensor beam geometry unavailable in AI4Shipwrecks dataset. |
| **07 Tiling** | `ml/preprocessing/07_tile.py` | **A. REQUIRED FOR PIPELINE** | COMPLETE | **Yes** | Slices swaths into 640×640 tiles with 20% overlap (512px stride) with 1–99% percentile stretch and deterministic padding. |
| **08 Mask → YOLO** | `ml/preprocessing/08_mask_to_yolo.py` | **A. REQUIRED FOR PIPELINE** | COMPLETE | **Yes** | Converts binary masks to single-class (`0: artificial_anomaly`) normalized YOLO coordinates. Emits empty `.txt` for negative tiles. |
| **09 Site Split** | `ml/preprocessing/09_site_split.py` | **A. REQUIRED FOR PIPELINE** | COMPLETE | **Yes** | Group-partitions data by physical survey site (70% train, 15% val, 15% test) to prevent spatial autocorrelation leakage. |
| **Runtime: Quality** | `ml/preprocessing/quality.py` | **A. REQUIRED FOR PIPELINE** | ACTIVE | **Yes (Service)** | Reusable quality metric module imported by `pipeline.py` and `sonar_service.py` during inference and API upload. |
| **Runtime: Normalize** | `ml/preprocessing/normalize.py` | **A. REQUIRED FOR PIPELINE** | ACTIVE | **Yes (Service)** | Reusable normalization utilities (percentile stretch, water column blanking) for inference and backend testing. |
| **Runtime: Tiling** | `ml/preprocessing/tiling.py` | **A. REQUIRED FOR PIPELINE** | ACTIVE | **Yes (Service)** | Reusable sliding-window generator and global-to-tile coordinate mapper used in the inference pipeline. |
| **Runtime: Pipeline** | `ml/preprocessing/pipeline.py` | **A. REQUIRED FOR PIPELINE** | ACTIVE | **Yes (Service)** | Unified `SonarPreprocessingPipeline` orchestration class used by `inference_service.py`. |
---

## 2b. SCTD Dataset — Pipeline Script Classification & Status Table

| Stage | Script / Module | Classification | Status | Used for Training? | Reason / Technical Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01 Dataset Inspection** | `ml/preprocessing/01_inspect_sctd.py` | **B. RESEARCH / EXPERIMENT** | COMPLETE | No | Read-only scan: 357 images, 3 raw classes (ship/aircraft/human), no site/nav/mask metadata. |
| **02 Quality Control** | `ml/preprocessing/02_quality_check_sctd.py` | **A. REQUIRED FOR PIPELINE** | COMPLETE | Yes (Filter) | 356 VALID, 1 SUSPICIOUS, 0 INVALID. |
| **03 Normalization** | `ml/preprocessing/03_normalize_sctd.py` | **A. REQUIRED FOR PIPELINE** | COMPLETE | Yes (Methodology) | 1–99% percentile stretch, same validated method as AI4Shipwrecks. Run as a separate step (not baked into tiling) for this adaptation. |
| **04 CLAHE** | *Not run for SCTD* | **B. RESEARCH / EXPERIMENT** | SKIPPED | No | Reused AI4Shipwrecks ablation finding — not re-tested. |
| **05 Denoising** | *Not run for SCTD* | **B. RESEARCH / EXPERIMENT** | SKIPPED | No | Reused AI4Shipwrecks ablation finding — not re-tested. |
| **06 Geometry / Nadir** | *None* | **N/A** | SKIPPED | No | SCTD images are small pre-cropped chips, not continuous swaths — no nadir strip present. |
| **07 Tiling** | `ml/preprocessing/07_tile_sctd.py` | **A. REQUIRED FOR PIPELINE** | COMPLETE | Yes | 640×640, 20% overlap/512px stride for images ≥640px; zero-padding for smaller (majority of SCTD). 398 tiles produced. |
| **08 Box → YOLO** | *(combined into 07_tile_sctd.py)* | **A. REQUIRED FOR PIPELINE** | COMPLETE | Yes | SCTD provides real bounding boxes (VOC XML), not masks — boxes re-clipped and converted directly per tile; no separate mask step needed. |
| **09 Split** | `ml/preprocessing/09_site_split_sctd.py` | **A. REQUIRED FOR PIPELINE** | COMPLETE | Yes | No site metadata available — split by source image instead (249/54/54 images → 282/60/56 tiles), same leakage-prevention goal. |

**Known limitations (SCTD):** no hard-negative examples present; no navigation metadata; "ship" label not distinguished as wreck vs. active vessel.
---

## 2c. KLSG Dataset — Pipeline Script Classification & Status Table

| Stage | Script / Module | Classification | Status | Used for Training? | Reason / Technical Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01 Dataset Inspection** | `ml/preprocessing/01_inspect_klsg.py` | **B. RESEARCH / EXPERIMENT** | COMPLETE | No | 428 images (380 ship, 48 plane), zero annotation files of any kind. |
| **02 Quality Control** | `ml/preprocessing/02_quality_check_klsg.py` | **A. REQUIRED FOR PIPELINE** | COMPLETE | Yes (Filter) | 425 VALID, 3 SUSPICIOUS, 0 INVALID. |
| **03 Normalization** | `ml/preprocessing/03_normalize_klsg.py` | **A. REQUIRED FOR PIPELINE** | COMPLETE | Yes (Methodology) | 1-99% percentile stretch, same validated method. |
| **04 CLAHE / 05 Denoising** | *Not run* | **B. RESEARCH / EXPERIMENT** | SKIPPED | No | Reused AI4Shipwrecks ablation finding. |
| **06 Geometry / Nadir** | *None* | **N/A** | SKIPPED | No | Pre-cropped chips, no nadir strip present. |
| **07 Tiling + Labeling** | `ml/preprocessing/07_tile_klsg.py` | **A. REQUIRED FOR PIPELINE** | COMPLETE, **WEAK LABELS** | Yes (flagged) | 474 tiles. **No real annotations exist in KLSG** — every box is an approximated central-80% region, flagged `weak_label=True`. |
| **09 Split** | `ml/preprocessing/09_site_split_klsg.py` | **A. REQUIRED FOR PIPELINE** | COMPLETE | Yes | No site metadata — split by source image (299/64/65 images -> 340/65/69 tiles). |

**Critical flag:** KLSG contributes weak/approximated labels only. Any reported accuracy figures must separate KLSG's contribution from SCTD/AI4Shipwrecks' real-annotation contributions — do not present a single blended accuracy number across all three without disclosing this difference.

---


## 3. Script Classification Definitions

- **A. REQUIRED FOR FINAL PIPELINE:** Core deterministic modules and dataset preparation scripts essential for the training data pipeline and real-time backend inference service.
- **B. RESEARCH / EXPERIMENT ONLY:** Exploratory, diagnostic, or ablation scripts. Their outputs provide comparative benchmarks (e.g. CLAHE vs baseline, FFT denoising vs baseline), but their transforms must never leak into training data.
- **C. OBSOLETE / DUPLICATE:** Redundant or replaced scripts. None present in `ml/preprocessing/`; modular runtime files (`normalize.py`, `quality.py`, `tiling.py`, `pipeline.py`) serve live services, while numbered scripts serve offline dataset preparation.
- **D. BROKEN:** Scripts with syntax errors, failed imports, or corrupt logic. None detected (all 42 Python files compile cleanly).

---

## 4. Production Pipeline Repreducibility Guide

To reproduce the dataset from raw files from scratch:

```bash
# 1. Inspect and check raw data quality (read-only)
python ml/preprocessing/01_inspect.py
python ml/preprocessing/02_quality_check.py

# 2. Tile normalized images and masks (640x640, 20% overlap, 1-99% percentile stretch)
python ml/preprocessing/07_tile.py

# 3. Convert segmentation masks to YOLO bounding boxes (Class 0, empty negatives)
python ml/preprocessing/08_mask_to_yolo.py

# 4. Partition by survey site into zero-leakage splits (70/15/15)
python ml/preprocessing/09_site_split.py
```
