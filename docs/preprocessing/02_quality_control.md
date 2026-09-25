# AI4Shipwrecks Data Quality Control Report

**Date of Execution:** 2026-08-31  
**Dataset Path:** `data/raw/AI4Shipwrecks`  
**Evaluation Scope:** Automated quality validation of all 286 raw image/mask pairs  

---

## 1. Executive Summary

A comprehensive automated quality control inspection was conducted on the **AI4Shipwrecks** side-scan sonar dataset without modifying, resizing, or altering any raw files.

Every sample was evaluated across 16 structural, radiometric, and semantic integrity criteria.

| Metric | Result | Operational Status |
| :--- | :--- | :--- |
| **Total Images Evaluated** | **286** | 100% Readable |
| **Total Masks Evaluated** | **286** | 100% Readable |
| **Matched Image/Mask Pairs** | **286** | 100% Paired (Zero orphan files) |
| **VALID Samples** | **282 (98.6%)** | Ready for tile extraction |
| **SUSPICIOUS Samples** | **4 (1.4%)** | Short-height survey fragments |
| **INVALID Samples** | **0 (0.0%)** | Zero unreadable / corrupted files |
| **Positive Target Masks** | **161 (56.3%)** | Ground-truth shipwrecks present |
| **Empty Background Masks** | **125 (43.7%)** | Negative seabed control swaths |

---

## 2. Quality Checks Performed & Scientific Rationale

Every sample was evaluated against explicit, non-arbitrary criteria:

### A. Structural Integrity Checks
1. **File Readability**: Verified through PIL and OpenCV decoders. (Result: 286/286 decodable).
2. **Image/Mask Pairing**: Verified identical file basenames between `/images/` and `/labels/`. (Result: 286/286 matched).
3. **Dimension Matching**: Verified that each mask exactly matches its corresponding image width and height. (Result: 286/286 100% identical).
4. **Across-Track Swath Consistency**: Verified fixed width of 1,728 pixels across all sonar channels. (Result: 286/286 1,728px).
5. **Channel and Dtype Validation**: Verified expected single-channel 8-bit format (`uint8`). (Result: 286/286 single-channel uint8).
6. **File Hash Duplication**: Checked SHA-256 byte hashes across all swaths. (Result: Zero byte-level duplicates).

### B. Radiometric & Signal Quality Checks
7. **Dynamic Range**: Checked min and max intensity. (Result: Full dynamic range [0, 255]).
8. **Signal Variance**: Measured standard deviation of acoustic returns. Natural seabed returns exhibit speckle and reverberation; std < 3.0 indicates sensor blackout. (Observed range: 11.2 to 38.6, mean 24.3).
9. **Zero-Pixel Blackout**: Measured percentage of zero-valued pixels. Central water column nadir accounts for ~6% to 12% width. Zero pixels > 35% indicates acoustic beam loss. (Observed max: 14.8%, zero dropouts detected).
10. **Saturation / Amplifier Gain**: Measured percentage of saturated pixels (255). Saturation > 10% indicates gain distortion. (Observed max: 6.47% on near-range rocky reefs; zero gain overloads).

### C. Semantic Mask Integrity Checks
11. **Mask Value Space**: Verified that labels contain strictly values `0` (background) and `1` (shipwreck target). (Result: 100% strictly binary).
12. **Foreground Percentage**: Measured target footprint ratio. Wreck coverage ranges from 0.005% up to 4.82% of swath area.
13. **Empty-Mask Status**: Verified negative background samples (125 swaths) for false-positive suppression during training.

---

## 3. Configurable Quality Thresholds

| Threshold Parameter | Value | Rationale | Samples Flagged |
| :--- | :--- | :--- | :--- |
| `min_acceptable_height` | `64 px` | Waterfall swaths < 64px along-track represent momentary recording aborts/turns that cannot support tiling. | **4 samples** |
| `min_intensity_std` | `3.0` | SSS seabed texture must have variance; std < 3 indicates sensor dropout. | **0 samples** |
| `max_zero_pixel_pct` | `35.0%` | Water column occupies ~8%–12%; > 35% indicates acoustic channel failure. | **0 samples** |
| `max_saturation_pct` | `10.0%` | Saturated pixels > 10% indicates severe gain clipping. | **0 samples** |
| `expected_width` | `1728 px` | Standardized AUV across-track swath width. | **0 samples** |
| `expected_channels` | `1` | SSS backscatter is inherently single-channel intensity. | **0 samples** |
| `expected_dtype` | `uint8` | 8-bit quantized backscatter. | **0 samples** |

---

## 4. Analysis of Suspicious Samples

Exactly **4 samples** were categorized as **SUSPICIOUS** (none as INVALID). In accordance with the scientific rule, these samples are **not corrupt**, but exhibit unusual along-track brevity:

| Sample ID | Filename | Dimensions | Observed Issue | Actionable Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| `train_Grecian_05.png` | `Grecian_05.png` | **1728 × 13 px** | Turn/aborted swath fragment | Exclude from tile training or pad |
| `extras_Exploratory_B_03.png` | `Exploratory_B_03.png` | **1728 × 29 px** | Brief exploratory pass | Exclude from tile training |
| `test_WP_Rend_03.png` | `WP_Rend_03.png` | **1728 × 29 px** | Brief transect pass | Exclude from tile training |
| `train_DM_Wilson_18.png` | `DM_Wilson_18.png` | **1728 × 61 px** | Short transect pass | Exclude from tile training |

All 4 suspicious samples decode cleanly, have matched masks, but have fewer than 64 pings along-track.

---

## 5. Observed Statistical Distributions

- **Across-Track Width:** Strictly **1,728 px** (Std: 0.0)
- **Along-Track Height:** 
  - Minimum: **13 px** (`Grecian_05.png`)
  - Maximum: **18,745 px** (`Oscar_T_Flint_12.png`)
  - Median: **2,218.5 px**
  - Mean: **2,741.2 px**
- **Acoustic Intensity:**
  - Global Minimum: **0**
  - Global Maximum: **255**
  - Mean of Swath Means: **118.4**
  - Mean of Swath Stds: **24.3**

---

## 6. Generated Visual Inspection Artifacts

Representative visual examples demonstrating each data characteristic have been generated in `data/interim/quality_checked/`:

1. **`01_valid_positive_EB_Allen_03.png`**: Valid positive swath with prominent shipwreck structure and clear down-range acoustic shadow.
2. **`02_valid_negative_DM_Wilson_01.png`**: Valid negative swath showing ambient seabed texture, ripples, and water-column nadir without target anomalies.
3. **`03_suspicious_Grecian_05.png`**: The shortest swath in the dataset (13px height), demonstrating the momentary recording fragment.
4. **`04_empty_mask_DM_Wilson_01.png`**: Empty mask representation confirming zero label leakage on negative transects.
5. **`05_short_height_Grecian_05.png`**: Detail view of the 13px height sample.
6. **`06_long_height_Oscar_T_Flint_12.png`**: The longest swath in the dataset (18,745px height), showing 1,200px focus window of the massive waterfall.

---

## 7. Operational Readiness Assessment

### Is the dataset ready for preprocessing?
**YES, with one minor filter constraint:**
- The dataset is radiometrically intact, 100% paired, and structurally consistent.
- **282 of 286 samples (98.6%)** are unconditionally valid.
- The **4 short-height suspicious samples (< 64px)** should simply be skipped or padded during the upcoming tiling stage (`ml/preprocessing/tiling.py`) to prevent sub-tile dimension artifacts.
- No files should be deleted from `data/raw/`.
