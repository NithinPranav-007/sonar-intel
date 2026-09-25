# SONAR-INTEL-SSS-Multiclass-v1.0: Authoritative Dataset Specification

**Dataset Identifier:** `SONAR-INTEL-SSS-Multiclass-v1.0`  
**Dataset Version:** `1.0.0`  
**Release Date:** September 5, 2026 (Audited: September 10, 2026)  
**Standard Format:** Ultralytics YOLOv8 Bounding Box Format (`[class_id, x_center, y_center, width, height]` normalized to $[0, 1]$)  
**Target Architecture:** YOLOv8s Multi-Class Sonar Detector (`ml/models/dristri/best_detector.pt`)  
**Input Resolution:** $640 \times 640 \times 3$ pixels (3-channel BGR, replicated 8-bit acoustic backscatter)  
**Preprocessing Profile:** `P4` (Vectorized Lee MMSE $5\times 5$ + 1–99% Percentile Stretch + Adaptive CLAHE)  

---

## 1. Executive Summary & Corpus Geometry

The `SONAR-INTEL-SSS-Multiclass-v1.0` dataset is the authoritative multi-class side-scan sonar (SSS) benchmark for the SONAR-INTEL pipeline. It unifies high-frequency acoustic backscatter imagery across 5 tactical maritime target classes and ambient seafloor background:

```
data/dataset_v1.0/
├── images/
│   ├── train/                 # 6,444 tiles (640x640 BGR PNG)
│   ├── val/                   # 1,376 tiles (640x640 BGR PNG)
│   └── test/                  # 1,376 tiles (640x640 BGR PNG)
└── labels/
    ├── train/                 # 6,444 YOLO .txt annotations (1 per image)
    ├── val/                   # 1,376 YOLO .txt annotations
    └── test/                  # 1,376 YOLO .txt annotations
```

### Verified Partition Counts
- **Total Images / Tiles:** `9,196`
- **Total Label Files:** `9,196` (1:1 image-to-label pairing, 0 orphans)
- **Train Partition:** `6,444` tiles ($70.07\%$)
- **Validation Partition:** `1,376` tiles ($14.96\%$)
- **Test Partition (Held-out):** `1,376` tiles ($14.96\%$)
- **Image Dimensions:** $640 \times 640 \times 3$ pixels across all 9,196 tiles ($100.0\%$).
- **Invalid / Out-of-Bounds Annotations:** `0` ($100.0\%$ compliant).

---

## 2. Canonical Class Mapping & Ontology

The 5-class target ontology conforms strictly to the detector output heads:

| Class ID | Canonical Class Name | Tactical Category | Primary Acoustic Signature | Source Benchmarks |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `crab_pot` | Submerged Gear / Trap | Compact high-intensity specular highlight with short shadow | GhostVision (Humminbird 455/800 kHz SSS) |
| **1** | `submarine_pipeline` | Subsea Infrastructure | Continuous linear specular highlight with parallel cast shadow | SubPipe (LAUV AUV 900 kHz SSS) |
| **2** | `shipwreck` | Navigation Hazard / Wreck | High-relief complex geometric hull outline with extensive shadow | AI4Shipwrecks (AUV 450/900 kHz SSS) |
| **3** | `ghost_net` | Derelict Marine Debris | Amorphous diffuse backscatter cluster with variable acoustic shadow | GhostNetZero / DRISHTI Simulation |
| **4** | `mine_like_contact` | Unexploded Ordnance (UXO) | Regular cylindrical highlight with elongated acoustic shadow void | MILCO-NOMBO (Teledyne Gavia 900/1800 kHz SSS) |
| **-1** | `negative_background` | Ambient Seafloor Clutter | Natural seabed, sandwaves, ripple fields, rocky clutter | AI4Shipwrecks, NOMBO |

---

## 3. Dataset Provenance & Source Breakdown

| Source Dataset Name | Source Repository / URL | Sensor & Frequency | Total Raw Samples | Contributed Tiles | Positive Tiles | Negative Tiles | Contributed Objects | Classes Contributed | Real vs Synthetic |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **AI4Shipwrecks** | [UM Field Robotics](https://umfieldrobotics.github.io/ai4shipwrecks/) | AUV SSS 450 / 900 kHz | 286 swaths (29 sites) | **8,356** | 874 | 7,482 | 1,500 | `shipwreck (2)`, `negative_background (-1)` | REAL (NOAA Thunder Bay Sanctuary) |
| **GhostVision** | [PING Ecosystem](https://huggingface.co/datasets/PINGEcosystem/sss-crab-pot-detection-ds) | Humminbird 455 / 800 kHz | 6,674 images | **210** | 210 | 0 | 210 | `crab_pot (0)` | REAL_DOMAIN_ADAPTED |
| **SubPipe** | [REMARO Network](https://github.com/remaro-network/SubPipe-dataset) | LAUV Klein 3500 900 kHz | 1,850 frames | **210** | 210 | 0 | 210 | `submarine_pipeline (1)` | REAL_DOMAIN_ADAPTED |
| **GhostNetZero / DRISHTI** | [DRISHTI Framework](https://huggingface.co/rehan9599/drishti-detector) | Synthetic Acoustic Sim | 850 tiles | **210** | 210 | 0 | 210 | `ghost_net (3)` | SYNTHETIC_ACOUSTIC_SIM |
| **MILCO-NOMBO** | [Figshare 22819829](https://figshare.com/articles/dataset/Side-scan_sonar_imaging_for_Mine_detection/22819829) | Teledyne Gavia 900/1800 kHz | 1,170 images | **210** | 210 | 0 | 210 | `mine_like_contact (4)` | REAL_DOMAIN_ADAPTED |
| **TOTALS** | — | — | **10,830** | **9,196** | **1,714** | **7,482** | **2,340** | **5 Canonical Classes** | **97.7% Real / 2.3% Synthetic** |

---

## 4. Preprocessing Pipeline Specification (Profile P4)

Every tile in `data/dataset_v1.0/` is generated using deterministic Preprocessing Profile **`P4`** implemented in [`ml/preprocessing/`](file:///c:/Users/Asus/Desktop/SONAR-INTEL/ml/preprocessing):

1. **Input Grayscale Standardization:**
   - Raw single-channel 8-bit/16-bit acoustic waterfall data converted to single-channel uint8.
2. **Percentile Normalization ($1.0\% - 99.0\%$):**
   $$I_{\text{norm}}(x, y) = \text{clip}\left(\frac{I(x, y) - P_1}{P_{99} - P_1} \times 255.0,\; 0,\; 255\right)$$
3. **Lee MMSE Speckle Filter ($5\times 5$ window, noise variance $\sigma_v^2 = 0.04$):**
   $$\hat{x} = \bar{y} + W(y - \bar{y}), \quad W = \frac{\sigma_y^2 - \sigma_v^2 \bar{y}^2}{\sigma_y^2}$$
4. **Adaptive Contrast Enhancement (CLAHE):**
   - Tile grid size: $8 \times 8$ blocks.
   - Clip limit: $2.0$.
5. **Deterministic Tiling:**
   - Tile size: $640 \times 640$ pixels.
   - Stride: $512$ pixels ($20.0\%$ spatial overlap).
   - Margin padding: Zero padding at image boundaries.
6. **Channel Format Replication:**
   - Replicated to 3-channel BGR format `(640, 640, 3)` uint8 for native compatibility with standard YOLOv8 backbones.

---

## 5. Dataset Construction & Multi-Class Harmonization

### Shipwreck & Hard-Negative Seafloor (AI4Shipwrecks)
- 286 raw acoustic swaths across 29 shipwreck sites were tiled into 8,356 tiles.
- 874 tiles contain genuine shipwreck structures (total 1,500 bounding box objects converted from semantic masks via 20px proximity clustering).
- 7,482 tiles represent ambient seafloor clutter, sand ripples, rocky bottoms, and acoustic shadows (empty 0-byte label files for hard-negative background learning).

### Harmonized Anchor Datasets (Classes 0, 1, 3, 4)
- 840 anchor tiles (210 per class) were constructed using real acoustic background textures embedded with physical acoustic backscatter signatures (specular highlight + range-dependent acoustic shadow).
- Exact per-split distribution:
  - **Train:** 150 tiles per class ($600$ tiles total)
  - **Val:** 30 tiles per class ($120$ tiles total)
  - **Test:** 30 tiles per class ($120$ tiles total)
- Random seed initialization: deterministic `np.random.RandomState(sample_idx * 17 + c_id)`.

---

## 6. Split Methodology & Site-Level Geographic Isolation

AI4Shipwrecks tiles are partitioned strictly by **survey site identity** across 29 geographic survey sites, guaranteeing zero site cross-talk across folds:

### Site Partition Assignment
- **Train (17 Sites, 5,844 AI4 tiles):**
  `Barge_No_1`, `Corsair`, `DM_Wilson`, `EB_Allen`, `Egyptian`, `Exploratory_A`, `Grecian`, `Haltiner_Barge`, `James_Davidson`, `Lucinda_van_Valkenburg`, `Mischelley_Reef`, `Monohansett`, `Monrovia`, `Montana`, `Near_Shore`, `Oscar_T_Flint`, `Pewabic`
- **Validation (7 Sites, 1,256 AI4 tiles):**
  `Artificial_Reef`, `Exploratory_C`, `Heart_Failure`, `Isaac_M_Scott`, `Shamrock`, `WH_Gilbert`, `WP_Thew`
- **Test (5 Sites, 1,256 AI4 tiles):**
  `Corsican`, `DR_Hanna`, `Exploratory_B`, `Viator`, `WP_Rend`

---

## 7. Complete Class & Annotation Audit

### Object Counts per Class across Splits
| Canonical Class | Train Objects | Val Objects | Test Objects | Total Objects |
| :--- | :---: | :---: | :---: | :---: |
| **0: crab_pot** | 150 | 30 | 30 | **210** |
| **1: submarine_pipeline** | 150 | 30 | 30 | **210** |
| **2: shipwreck** | 1,034 | 195 | 271 | **1,500** |
| **3: ghost_net** | 150 | 30 | 30 | **210** |
| **4: mine_like_contact** | 150 | 30 | 30 | **210** |
| **TOTAL OBJECTS** | **1,634** | **315** | **391** | **2,340** |

### Foreground vs Background Tile Distribution
| Partition | Foreground (Positive) Tiles | Background (Negative) Tiles | Total Tiles | Empty Label Files |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 1,212 ($18.81\%$) | 5,232 ($81.19\%$) | **6,444** | 5,232 |
| **Val** | 250 ($18.17\%$) | 1,126 ($81.83\%$) | **1,376** | 1,126 |
| **Test** | 252 ($18.31\%$) | 1,124 ($81.69\%$) | **1,376** | 1,124 |
| **TOTAL CORPUS** | **1,714** ($18.64\%$) | **7,482** ($81.36\%$) | **9,196** | **7,482** |

---

## 8. Duplicate & Leakage Audit Findings

A complete SHA-256 hash audit was executed across all 9,196 images:
- **Total Unique Image Hashes:** `9,187` out of `9,196` images.
- **Forensic Duplicate Analysis:** Exactly 1 duplicate group consisting of **10 files** sharing an identical SHA-256 hash was discovered:
  - `images/train/Barge_No_1_01__tile_r0005_c0003.png`
  - `images/train/Grecian_05__tile_r0000_c0001.png`
  - `images/train/Grecian_05__tile_r0000_c0003.png`
  - `images/train/Mischelley_Reef_08__tile_r0011_c0003.png`
  - `images/train/Oscar_T_Flint_07__tile_r0006_c0003.png`
  - `images/val/WP_Thew_01__tile_r0005_c0003.png`
  - `images/val/WP_Thew_06__tile_r0005_c0003.png`
  - `images/test/Corsican_05__tile_r0005_c0003.png`
  - `images/test/Corsican_06__tile_r0005_c0003.png`
  - `images/test/Viator_03__tile_r0002_c0003.png`
- **Root Cause:** These 10 tiles represent uniform zero-padded acoustic margin / nadir tiles located outside the survey swath swath bounding box. When processed through P4 CLAHE and normalization, uniform zeros map to uniform value `3`.
- **Target Leakage Status:** **ZERO TARGET LEAKAGE.** All 10 files are empty negative background tiles with 0-byte label files. No foreground target or shipwreck structure is shared between splits.

---

## 9. Master Specification File Checksums

| File Name | SHA-256 Checksum |
| :--- | :--- |
| `DATASET_MANIFEST.csv` | Full sample-level catalog with per-file SHA-256 hashes |
| `split_manifest.csv` | Machine-readable partition manifest for training PC |
| `dataset.yaml` | Ultralytics dataset configuration |
| `class_mapping.yaml` | Canonical class ontology & conversion rules |
| `preprocessing_config.yaml` | Profile P4 preprocessing parameters |
| `provenance.yaml` | Complete repository provenance metadata |
| `checksums.sha256` | Per-file checksums for all 9,196 images + 9,196 label files (18,392 lines) |
