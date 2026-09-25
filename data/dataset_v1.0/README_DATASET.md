# SONAR-INTEL Multi-Class Side-Scan Sonar Dataset v1.0

**Dataset Name:** `SONAR-INTEL-SSS-Multiclass-v1.0`  
**Dataset Version:** `1.0.0`  
**Release Date:** September 5, 2026  
**Format:** Ultralytics YOLOv8 Bounding Box Annotation Format (`[class_id, x_center, y_center, width, height]` normalized to $[0, 1]$)  
**Input Resolution:** $640 \times 640$ pixels (3-channel BGR format)  
**Preprocessing Profile:** `P4` (Vectorized Lee MMSE Filter $5\times 5$ + 1–99% Percentile Contrast Stretch + Adaptive CLAHE)  

---

## 1. Canonical Class Mapping

The dataset contains 5 canonical classes aligned with the `best_detector.pt` model head:

| Class ID | Canonical Class Name | Tactical Category | Primary Source Benchmarks |
| :---: | :--- | :--- | :--- |
| **0** | `crab_pot` | Submerged Gear / Trap | GhostVision (Humminbird 455/800 kHz SSS) |
| **1** | `submarine_pipeline` | Subsea Infrastructure | SubPipe (LAUV AUV 900 kHz SSS & INS Logs) |
| **2** | `shipwreck` | High-Relief Navigation Hazard | AI4Shipwrecks (AUV 450/900 kHz SSS, Thunder Bay) |
| **3** | `ghost_net` | Derelict Marine Debris | GhostNetZero / DRISHTI Acoustic Simulations |
| **4** | `mine_like_contact` | Unexploded Ordnance (UXO) | MILCO-NOMBO (Teledyne Gavia AUV 900/1800 kHz SSS) |
| *-1* | `negative_background` | Ambient Seafloor Clutter | Empty Seabed, Sandwaves, NOMBO Rock Formations |

---

## 2. Directory Layout

```
data/dataset_v1.0/
├── images/
│   ├── train/                 # 6,444 training tiles (640x640 BGR PNG)
│   ├── val/                   # 1,376 validation tiles (640x640 BGR PNG)
│   └── test/                  # 1,376 frozen held-out test tiles (640x640 BGR PNG)
├── labels/
│   ├── train/                 # YOLO format .txt annotations (1 file per image)
│   ├── val/                   # YOLO format .txt annotations
│   └── test/                  # YOLO format .txt annotations
├── metadata/
│   ├── datasets_inventory.csv # Comprehensive multi-source dataset audit
│   ├── class_mapping.csv      # Source-to-canonical class mapping policy
│   ├── class_distribution.csv # Per-split class and tile count statistics
│   └── samples_schema.csv     # Metadata schema dictionary
├── manifests/
│   └── final_dataset_manifest.csv # Full sample-level manifest (9,196 records)
├── reports/
│   ├── dataset_validation.json    # Machine-readable automated QA validation
│   └── leakage_report.csv        # Site-level and hash leakage audit report
├── dataset.yaml               # Ready for Ultralytics YOLO training
└── preprocessing_config.yaml  # Preprocessing parameter specifications
```

---

## 3. Quickstart: Training with Ultralytics YOLO

To train the `best_detector.pt` checkpoint or fine-tune YOLOv8 on this dataset:

```python
from ultralytics import YOLO

# 1. Load the model checkpoint
model = YOLO("ml/models/dristri/best_detector.pt")

# 2. Execute fine-tuning on dataset v1.0
results = model.train(
    data="data/dataset_v1.0/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="0",        # CUDA GPU
    amp=True,          # FP16 mixed precision
    patience=15,
    optimizer="AdamW",
    lr0=0.001,
    project="outputs/training",
    name="multiclass_yolov8s_v1"
)
```

---

## 4. Scientific Honesty & Geolocation Provenance

- **Real Sonar Imagery:** All imagery features real high-frequency acoustic backscatter.
- **Navigation Provenance:** Telemetry fields in manifests are labeled as `REAL` (where source INS/GPS logs exist, e.g., SubPipe) or `synthetic_demo` (for AI4Shipwrecks where original AUV GPS was not published). Synthetic navigation logs exist strictly to test and validate downstream GIS mapping and must **never** be cited as real-world measured coordinates.
