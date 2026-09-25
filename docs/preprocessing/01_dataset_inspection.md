# AI4Shipwrecks Dataset Inspection Report

**Date of Inspection:** 2026-08-31  
**Dataset Reference:** [UM Field Robotics AI4Shipwrecks](https://umfieldrobotics.github.io/ai4shipwrecks/)  
**Dataset Root:** `data/raw/ai4shipwrecks`  

---

## 1. Executive Summary

The **AI4Shipwrecks** benchmark dataset consists of 286 high-resolution side-scan sonar (SSS) waterfall recordings collected by Autonomous Underwater Vehicles (AUVs) across Thunder Bay National Marine Sanctuary in Lake Huron, Michigan. 

Every image is accompanied by an exact corresponding pixel-level binary segmentation mask labeled with maritime archaeological consultation.

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Total Sonar Images** | **286** | Single-channel 8-bit grayscale PNG |
| **Total Semantic Masks** | **286** | Single-channel 8-bit binary PNG |
| **Matched Pairs** | **286 (100%)** | Exact 1-to-1 filename match |
| **Unmatched Images** | **0** | No missing masks |
| **Unmatched Masks** | **0** | No orphan masks |
| **Masks With Target (1)** | **161 (56.3%)** | Swaths intersecting the wreck structure |
| **Masks Background Only (0)** | **125 (43.7%)** | Ambient seabed / run-in swaths |
| **Across-Track Width** | **1728 px** | Constant across 100% of images |
| **Along-Track Height** | **13 – 18,745 px** | Median: **2,218.5 px** |
| **Intensity Range** | **[0, 255]** | Uncalibrated acoustic backscatter |
| **Mask Unique Values** | **[0, 1]** | `0` = Seabed/Water Column, `1` = Shipwreck |
| **Unique Sites** | **29 sites** | 28 shipwreck sites + 1 terrain exploration site |

---

## 2. Directory Structure & Partition

The raw dataset adheres to a strict site-separated train/test partition to prevent spatial leakage across contiguous acoustic swaths:

```
data/raw/ai4shipwrecks/
├── train/
│   ├── images/  (141 PNG swaths, 12 unique shipwreck sites)
│   └── labels/  (141 PNG masks)
├── test/
│   ├── images/  (120 PNG swaths, 13 unique shipwreck sites)
│   └── labels/  (120 PNG masks)
└── extras/
    └── terrain/
        ├── images/ (25 PNG swaths, 4 terrain exploration sites)
        └── labels/ (25 PNG masks)
```

### Site Separation Verification
- **Train Sites (12):** DM_Wilson, DR_Hanna, EB_Allen, Egyptian, Grecian, Heart_Failure, Lucinda_van_Valkenburg, Monohansett, Montana, Oscar_T_Flint, Viator, WP_Thew.
- **Test Sites (13):** Artificial_Reef, Barge_No_1, Corsair, Corsican, Haltiner_Barge, Isaac_M_Scott, James_Davidson, Monrovia, Near_Shore, Pewabic, Shamrock, WH_Gilbert, WP_Rend.
- **Extras/Terrain (4):** Exploratory_A, Exploratory_B, Exploratory_C, Mischelley_Reef.
- **Train/Test Site Overlap:** **0 sites (0.0%)** — strict geographic isolation verified.

---

## 3. Dimensional and Radiometric Characteristics

### Image & Mask Geometry
- **Width:** Exactly **1,728 pixels** across every survey frame. In typical SSS configurations, this corresponds to ~50m port and ~50m starboard swath coverage with central nadir.
- **Height:** Highly variable along-track length, ranging from **13 pixels** up to **18,745 pixels** (median: **2,218.5 pixels**).
- **Channels:** Single-channel 2D grayscale (`uint8`).
- **Dynamic Range:** Minimum pixel value **0**, maximum pixel value **255**.

---

## 4. Ground Truth Annotations & Labels

- **Label Format:** Single-channel 8-bit PNG binary segmentation mask.
- **Class Mapping:**
  - `0`: Background (Ambient seabed, water column nadir, acoustic shadow, or open water).
  - `1`: Shipwreck / Anthropogenic structure highlight.
- **Target Coverage:**
  - **161 images** contain shipwreck labels (values `[0, 1]`).
  - **125 images** contain purely background labels (value `[0]`). These represent essential negative samples (seabed textures, ripples, exploratory passes).

---

## 5. Domain & Operational Suitability Analysis

### A. Suitability for the SONAR-INTEL MVP
1. **Strong Fit for Anomaly Candidate Detection:**
   Shipwrecks represent large-scale man-made artificial structures exhibiting intense acoustic returns accompanied by elongated acoustic shadows. This serves as an ideal training ground for the candidate triage pipeline.
2. **Real Acoustic Physics:**
   Unlike synthetic or cleaned toy images, the dataset contains authentic acoustic phenomena: beam spread, grazing angle attenuation, central water-column nadir zones, and speckle noise.

### B. Supported vs. Unsupported Classes
- **SUPPORTED:** Binary `shipwreck` structure (mapped to `artificial_anomaly`).
- **NOT SUPPORTED:** The labels do **NOT** distinguish ghost nets, abandoned fishing gear, marine plastics, or shipping containers.
- **Scientific Discipline:** Outputs trained on this dataset must be labeled `"AI Candidate / Possible Artificial Anomaly"` or `"Shipwreck Candidate"`, and must **never** claim to confirm ghost nets or lost fishing gear without ground-truth gear annotations.

### C. Missing Information (Explicit Gap Analysis)
The following metadata fields are **NOT AVAILABLE** in the original dataset:

| Field | Availability | Mitigation in SONAR-INTEL |
| :--- | :--- | :--- |
| **GPS Coordinates (Lat/Lon)** | **NOT AVAILABLE** | System stores `latitude=null, longitude=null, localization_status='UNAVAILABLE'`. |
| **Navigation Logs (CSV/XTF)** | **NOT AVAILABLE** | Geolocation estimation bypassed; analyst notified navigation unavailable. |
| **Vessel Heading & Speed** | **NOT AVAILABLE** | Directional slant-range projection disabled. |
| **Towfish Altitude & Range** | **NOT AVAILABLE** | Altitude estimated via empirical nadir width (~8% swath width). |
| **Bounding Boxes (YOLO)** | **NOT AVAILABLE** | YOLO candidate bounding boxes must be derived from mask contour envelopes during dataset preparation. |

---

## 6. Visual Inspection Samples

Side-by-side inspection visualizations (LEFT: raw SSS image, RIGHT: ground-truth binary mask) have been generated in `data/interim/inspection/`:

- `inspect_DM_Wilson_02_pair.png`
- `inspect_EB_Allen_03_pair.png`
- `inspect_Egyptian_04_pair.png`
- `inspect_Lucinda_van_Valkenburg_05_pair.png`
- `inspect_Oscar_T_Flint_01_pair.png`
- `inspect_Artificial_Reef_06_pair.png`
- `inspect_Barge_No_1_02_pair.png`
- `inspect_Corsair_01_pair.png`

---

## 7. Next Stage Constraints
As mandated by project protocol:
- Raw dataset in `data/raw/ai4shipwrecks/` remains completely unmodified.
- No normalization, CLAHE, tiling, YOLO fine-tuning, or database modification has been executed during this inspection stage.
