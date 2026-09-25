# AI4Shipwrecks Target Spatial Size & Bounding-Box Distribution Report

**Document:** `docs/dataset/target_size_distribution.md`  
**Dataset:** AI4Shipwrecks (`data/raw/AI4Shipwrecks/`)  
**Scope:** Read-Only Geometric & Spatial Dimension Analysis of Ground-Truth Segmentation Masks  

---

## 1. Executive Summary

A comprehensive, read-only spatial analysis was conducted across all **286 binary segmentation masks** in the AI4Shipwrecks dataset to determine the exact bounding-box size distribution of annotated targets.

| Metric | Value | Domain Interpretation |
| :--- | :--- | :--- |
| **Total Survey Swaths** | **286** | 100% Inspected |
| **Positive Images (with Target)** | **161 (56.3%)** | Ground-truth shipwrecks present |
| **Empty Images (Background Only)** | **125 (43.7%)** | Ambient seafloor control swaths |
| **Total Foreground Components** | **994** | Across all 161 positive masks |
| **Average Components per Swath** | **6.17** | Multi-part structural fragmentation |
| **Median Target Width** | **42.0 px** | Highly concentrated in small-to-medium scale |
| **Median Target Height** | **41.0 px** | Compact along-track profile |
| **Maximum Target Width** | **822 px** | Broad acoustic returns across-track |
| **Maximum Target Height** | **1450 px** | Massive along-track wreck structures |

---

## 2. Statistical Dimension Breakdown

### 2.1 Bounding-Box Width, Height & Area

| Metric | Minimum | Median (P50) | Mean | Maximum | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Target Width (w_bbox)** | **1 px** | **42.0 px** | **100.56 px** | **822 px** | 140.21 px |
| **Target Height (h_bbox)** | **1 px** | **41.0 px** | **114.93 px** | **1450 px** | 200.37 px |
| **BBox Area (w * h)** | **1 px²** | **1628.0 px²** | **31500.28 px²** | **817699 px²** | — |

### 2.2 Percentile Distribution

| Percentile | Target Width (px) | Target Height (px) | BBox Area (px²) | Aspect Ratio ($w/h$) |
| :--- | :--- | :--- | :--- | :--- |
| **P10** | 10.0 | 12.0 | 137.2 | 0.37 |
| **P25** | 19.0 | 22.0 | 493.0 | 0.61 |
| **P50 (Median)** | **42.0** | **41.0** | **1628.0** | **0.95** |
| **P75** | 109.0 | 96.0 | 9924.0 | 1.47 |
| **P90** | 299.0 | 316.2 | 78282.8 | 2.19 |
| **P95** | **458.7** | **547.4** | **205562.7** | **3.01** |

---

## 3. Critical Domain Limitation: Component vs. Physical Object

> [!WARNING]
> **Connected components in binary side-scan sonar masks do NOT directly correspond to discrete physical objects.**
>
> In side-scan sonar imagery, a single physical shipwreck frequently fragments into multiple disjoint connected components due to:
> 1. **Internal acoustic shadows**: Acoustic occlusions cast by gunwales or deck structures create interior zero-pixel gaps that split a hull mask.
> 2. **Structural collapse**: Dispersed debris fields, dislodged boilers, engines, and masts detached from the main hull.
> 3. **Grazing-angle dropouts**: Areas of low acoustic backscatter in the central hull interior.
>
> Therefore, both individual connected component dimensions and macro-envelope footprints must be considered when selecting sliding-window tile sizes.

---

## 4. Candidate Tile Size Evaluation (512, 640, 768, 1024)

To prevent targets from being bisected or truncated across tile seams during downstream inference, we measure what percentage of targets fit **entirely within a single candidate window**:

| Candidate Tile Size | Component Enclosure Rate | Macro-Envelope Enclosure Rate | Structural & Compute Trade-off |
| :--- | :--- | :--- | :--- |
| **512 × 512 px** | **92.25%** (917/994) | **27.33%** (44/161) | Fits most small sub-components, but truncates 72.7% of full shipwreck macro-structures. |
| **640 × 640 px** | **95.47%** (949/994) | **37.89%** (61/161) | Standard YOLO native resolution. Encloses 95.5% of components and 37.9% of full wrecks. |
| **768 × 768 px** | **96.58%** (960/994) | **46.58%** (75/161) | Encloses 96.6% of components and 46.6% of whole wrecks; modest compute footprint. |
| **1024 × 1024 px**| **99.09%** (985/994) | **61.49%** (99/161) | Captures 99.1% of components and 61.5% of macro-envelopes intact with acoustic shadows. |

---

## 5. Tile Size Recommendation for Experimental Evaluation

Based on the observed spatial dimensions:

1. **Candidate 640 × 640 px (Recommended Default Baseline)**:
   - Evaluates standard YOLOv8 native receptive fields.
   - Encloses **95.47% of all individual target components** (P90 width = 299px, P90 height = 316px).
   - High training throughput and low VRAM footprint on consumer GPUs (e.g. RTX 3050 4GB).

2. **Candidate 1024 × 1024 px (Recommended Extended Context Baseline)**:
   - Essential for large maritime wrecks and down-range acoustic shadows extending > 500 px.
   - Encloses **99.09% of components** and **61.49% of full shipwreck envelopes**.
   - Preserves complete vessel geometry (e.g. 800+ px long freighters and barges) in a single window.

3. **Candidate 512 × 512 px & 768 × 768 px (Ablation Comparison)**:
   - 512px tests extreme lightweight mobile/edge inference.
   - 768px provides a balanced intermediate stepping point.

4. **Sliding-Window Overlap Requirement**:
   - Because 38.5% of macro-envelopes exceed 1024px and 62.1% exceed 640px, a **20% to 25% sliding-window overlap** (e.g. stride = 480px for 640px tiles) is mandatory to prevent boundary truncation.
