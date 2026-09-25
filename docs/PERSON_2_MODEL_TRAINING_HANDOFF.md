# PERSON 2: MODEL TRAINING HANDOFF & OPERATIONAL RUNBOOK

**Document ID**: `DOC-P2-HANDOFF-v1.0`  
**Target Engineer**: Person 2 (ML / Model Training Engineer)  
**Upstream Author**: Person 1 (Data Engineering & Preprocessing Lead)  
**Dataset Name**: `SONAR-INTEL-SSS-Multiclass-v1.0`  
**Dataset Version**: `v1.0.0`  
**Dataset Status**: **TRAINER-READY**  
**Audit Timestamp**: `2026-09-05T02:30:00Z`  

---

## 1. Dataset Location & Directory Layout

The complete, validated, trainer-ready dataset is located at:
```
c:\Users\Asus\Desktop\SONAR-INTEL\data\dataset_v1.0\
```

### Directory Structure:
```
data/dataset_v1.0/
├── dataset.yaml                  # Ultralytics YOLOv8 training config
├── preprocessing_config.yaml     # P4 Preprocessing configuration profile
├── README_DATASET.md             # Dataset documentation
├── images/
│   ├── train/                    # 6,444 verified 640x640 PNG tiles
│   ├── val/                      # 1,376 verified 640x640 PNG tiles
│   └── test/                     # 1,376 verified 640x640 PNG tiles
├── labels/
│   ├── train/                    # 6,444 normalized YOLO .txt label files
│   ├── val/                      # 1,376 normalized YOLO .txt label files
│   └── test/                     # 1,376 normalized YOLO .txt label files
├── manifests/
│   └── final_dataset_manifest.csv # Full sample-level telemetry & metadata (9,196 rows)
├── metadata/
│   ├── datasets_inventory.csv     # Source inventory and license provenance
│   ├── class_mapping.csv          # Canonical class mapping (0-4)
│   ├── class_distribution.csv     # Per-split tile and object counts
│   └── samples_schema.csv         # Schema definition for telemetry & reviews
└── reports/
    ├── dataset_validation.json    # Machine-readable forensic audit results
    └── leakage_report.csv         # Site and hash isolation audit log
```

---

## 2. Dataset Core Statistics

| Metric | Train | Val | Test | Total |
| :--- | :--- | :--- | :--- | :--- |
| **Total 640×640 Tiles** | **6,444** | **1,376** | **1,376** | **9,196** |
| **Positive Tiles (Foreground)** | 1,212 | 250 | 252 | **1,714** (18.64%) |
| **Negative Tiles (Background)** | 5,232 | 1,126 | 1,124 | **7,482** (81.36%) |
| **Total Annotated Objects** | 1,692 | 324 | 324 | **2,340** |
| **Shipwreck Objects (Class 2)** | 1,062 | 219 | 219 | **1,500** |
| **Crab Pot Objects (Class 0)** | 157 | 26 | 27 | **210** |
| **Pipeline Objects (Class 1)** | 158 | 26 | 26 | **210** |
| **Ghost Net Objects (Class 3)** | 158 | 26 | 26 | **210** |
| **Mine Contact Objects (Class 4)**| 157 | 27 | 26 | **210** |

### Negative Tile Breakdown (7,482 Tiles):
- **`hard_negative_clutter`** (7,472 tiles / 99.87%): High-backscatter seafloor ripples, rocky ledges, acoustic shadows, glacial drop-offs.
- **`easy_background`** (10 tiles / 0.13%): Homogeneous flat sand/mud.
- **`confusing_acoustic_artifact`** (0 tiles): Filtered or suppressed during P4 dynamic range clipping.

---

## 3. Canonical Class Ontology (5 Classes)

The dataset and model head use the following zero-indexed classes:

```yaml
nc: 5
names:
  0: crab_pot
  1: submarine_pipeline
  2: shipwreck
  3: ghost_net
  4: mine_like_contact
```

> [!NOTE]
> **MILCO Rule**: Class `4` (`mine_like_contact`) represents acoustic candidate anomalies ("Mine-Like Contacts"). Under no circumstances should this class be termed "confirmed mine". Non-mine bottom objects (NOMBO) are mapped to background / hard-negative tiles.

---

## 4. Preprocessing Profile: Active Baseline P4

Every image in `data/dataset_v1.0/images/` was processed under the **P4** chain:
$$\text{Raw Sonar} \xrightarrow{\text{Grayscale}} \text{1--99\% Normalization} \xrightarrow{\text{Lee Speckle Filter (5}\times\text{5, }\sigma^2=0.04\text{)}} \xrightarrow{\text{CLAHE (clip}=2.0\text{)}} \xrightarrow{\text{3-channel BGR}}$$

- **Lee Filter**: Additive MMSE speckle reduction ($5 \times 5$ window, $\sigma^2_{\text{noise}} = 0.04$, reflect padding).
- **Normalization**: 1st to 99th percentile stretching to $[0, 255]$ with NaN/inf handling.
- **CLAHE**: Tile grid $(8, 8)$, contrast clipping limit $2.0$.
- **Image Input Size**: Fixed $640 \times 640 \times 3$ uint8.

---

## 5. Model Provenance & Compatibility

- **Checkpoint Path**: `ml/models/dristri/best_detector.pt`
- **Architecture**: Ultralytics YOLOv8s (Side-Scan Sonar Debris Detector)
- **Compatibility Status**: **DIRECT COMPATIBILITY VERIFIED**
  - Model loads cleanly via `from ultralytics import YOLO; model = YOLO("ml/models/dristri/best_detector.pt")`.
  - Output head contains 5 classes: `{0: 'crab_pot', 1: 'submarine_pipeline', 2: 'shipwreck', 3: 'ghost_net', 4: 'mine_cylinder'}`.
  - Class `4` in checkpoint (`mine_cylinder`) matches canonical `mine_like_contact` without requiring head surgery or weight resection.

---

## 6. Exact Training Command & Python Script

### Recommended First Controlled Experiment:
Person 2 should launch the baseline fine-tuning using the script below:

```python
"""
SONAR-INTEL: Person 2 Baseline Training Script
Experiment: EXP-01-BASELINE-P4
"""
import os
from ultralytics import YOLO

def train_sonar_intel_baseline():
    dataset_yaml = "data/dataset_v1.0/dataset.yaml"
    checkpoint_path = "ml/models/dristri/best_detector.pt"
    
    # Load pretrained DRISHTI YOLOv8s detector
    model = YOLO(checkpoint_path)
    
    # Controlled Training Run
    results = model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,                   # CUDA GPU (falls back to 'cpu' if unavailable)
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        patience=15,
        save=True,
        project="outputs/train_runs",
        name="exp01_p4_baseline",
        seed=42,
        deterministic=True,
        verbose=True
    )
    print("Baseline Training Complete. Metrics logged to outputs/train_runs/exp01_p4_baseline/")

if __name__ == "__main__":
    train_sonar_intel_baseline()
```

---

## 7. Metrics to Record During Experiment 1

Person 2 must record the following metrics on the held-out `test` set (`1,376` tiles):

1. **Overall Detection**:
   - $\text{mAP}_{50}$
   - $\text{mAP}_{50\text{--}95}$
   - Precision ($P$) & Recall ($R$)
   - $F_1\text{-score}$
2. **Per-Class Metrics**:
   - Per-class Average Precision ($\text{AP}_{50}$ and $\text{AP}_{50\text{--}95}$) for all 5 classes.
   - Per-class Recall ($R_{50}$) for `shipwreck` vs minor classes.
3. **Hard-Negative Specificity**:
   - False Positive Rate ($FPR$) on the 1,124 negative test tiles.
4. **Diagnostic Artifacts**:
   - Confusion Matrix (normalized).
   - $F_1\text{-Confidence Curve}$.
   - Precision-Recall Curve.

---

## 8. What NOT to Change During Experiment 1

To maintain strict scientific control:
- ❌ **DO NOT change the split** (train/val/test must remain as defined in `dataset.yaml`).
- ❌ **DO NOT resize images** (keep `imgsz=640`).
- ❌ **DO NOT delete shipwreck tiles** to balance classes (use weighted loss or focal loss if needed).
- ❌ **DO NOT alter the canonical class IDs** (0: crab_pot, 1: submarine_pipeline, 2: shipwreck, 3: ghost_net, 4: mine_like_contact).
- ❌ **DO NOT modify the P4 preprocessing** for Experiment 1.

---

## 9. Preprocessing Ablation Matrix (Subsequent Experiments)

After Experiment 1 establishes the P4 baseline, Person 2 is requested to execute the 4-way preprocessing ablation under identical seeds, batch sizes, and splits:

| Experiment | Profile | Preprocessing Chain | Objective |
| :--- | :--- | :--- | :--- |
| **EXP-01** | **P4 (Active)** | 1-99% Norm + Lee ($5\times5$) + CLAHE | Full baseline |
| **EXP-02** | **P1** | 1-99% Norm only | Gauge raw sonar contrast without filtering |
| **EXP-03** | **P2** | 1-99% Norm + Lee ($5\times5$) | Isolate Lee speckle reduction effect |
| **EXP-04** | **P3** | 1-99% Norm + CLAHE | Isolate histogram equalization effect |

---

## 10. Downstream SW-Net & Continuous Learning Handoff

Each tile's provenance is traceable via `data/dataset_v1.0/manifests/final_dataset_manifest.csv`.

When Person 2 outputs bounding boxes, the candidate format connects to future downstream components:
1. **Candidate Box**: $(x_1, y_1, x_2, y_2, \text{conf}, \text{class\_id})$
2. **Global Geospatial Projection**: Using `coordinate_source`, `latitude`, `longitude`, `depth`, `heading` from manifest.
3. **SW-Net Feature Extractor**: Will extract acoustic highlight/shadow pairs from raw crops indexed by `parent_image_id`.
4. **Active Learning Feedback**: Verified annotations from human review will feed into `data/dataset_v2.0/` via the schema fields documented in `metadata/samples_schema.csv`.

---

**Sign-off**:  
Person 1 (Data Engineering Lead): *Validated & Approved*  
Dataset State: **TRAINER-READY: YES**
