# Side-Scan Sonar Mask-to-Detection Annotation Strategy Analysis

**Document:** `docs/dataset/annotation_strategy_analysis.md`  
**Project:** SONAR-INTEL  
**Dataset:** AI4Shipwrecks ($640 \times 640$ Tiled Dataset)  
**Input Scope:** 8,356 Total Tiles (887 Positive Tiles, 7,469 Negative Control Tiles)  
**Target Model:** YOLOv8n (Single-Stage Object Detector)  

---

## 1. Executive Summary & Core Domain Thesis

A comprehensive analytical evaluation was performed across all **887 positive image/mask tiles** to establish a principled mapping from pixel-level binary segmentation masks to 2D bounding boxes for YOLOv8n training.

> [!CRITICAL]
> **Core Hydroacoustic Constraint**:
> **A connected component in a side-scan sonar mask does NOT directly represent one discrete physical object.**
>
> An individual physical vessel (e.g. an 80-meter Great Lakes steam freighter) produces multiple disjoint acoustic highlight regions separated by:
> 1. **Internal acoustic shadows**: Bulkheads, engine casings, and gunwales occlude acoustic beams, creating interior zero-pixel voids that bisect the continuous hull.
> 2. **Grazing-angle dropouts**: Low backscatter within open cargo holds.
> 3. **Scattered structural debris**: Boilers, engines, and masts broken off and resting within 5–20 meters of the hull.
>
> Assuming 1 connected component = 1 physical detection results in catastrophic over-fragmentation during object detector training.

---

## 2. Quantitative Component Distribution

From the 887 positive tiles, **2237 individual connected components** were extracted (8-connectivity):

| Metric | Minimum | P10 | P25 | P50 (Median) | P75 | P90 | P95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Component Width (w)** | 1 px | 8.6 px | 18.0 px | **42.0 px** | 116.0 px | 225.4 px | **339.6 px** |
| **Component Height (h)** | 1 px | 10.0 px | 19.0 px | **41.0 px** | 104.0 px | 259.0 px | **401.0 px** |
| **BBox Area (w * h)** | 1 px² | 99.6 px² | 400.0 px² | **1624.0 px²** | 11616.0 px² | 53824.0 px² | **99543.6 px²** |
| **Foreground Density (A_px/A_box)** | 0.003 | 0.2809 | 0.3946 | **0.54** | 0.6577 | 0.7557 | **0.8095** |

### 2.1 Prevalence of Tiny Components

| Dimension Filter | Component Count | Percentage of Total | Hydroacoustic Interpretation |
| :--- | :--- | :--- | :--- |
| **Dimension < 5 px** | **86** | **3.84%** | Acoustic speckle spikes & single-pixel boundary noise. |
| **Dimension < 10 px** | **327** | **14.62%** | Sub-resolution debris artifacts; below YOLOv8 P3 anchor resolution (8x8 stride). |
| **Dimension < 20 px** | **766** | **34.24%** | Small debris fragments; vulnerable to false-positive clutter. |
| **Dimension < 32 px** | **1148** | **51.32%** | Standard small-object cutoff (32x32 px MS COCO definition). |

---

## 3. Boundary-Touching Analysis

When swaths are sliced into 640x640 windows, targets crossing tile seams are clipped:

- **Components Touching Tile Seams**: **1127 components (50.38%)**
- **Positive Tiles Containing Boundary Components**: **728 tiles (82.07%)**
- **Distribution of Seam Intersections**:
  - Left edge (x=0): 380
  - Right edge (x=valid_w): 397
  - Top edge (y=0): 344
  - Bottom edge (y=valid_h): 355

> [!NOTE]
> **Boundary Preservation Rule**:
> Boundary-crossing targets must **NOT** be deleted. Because a 20% spatial overlap (128 px) exists between adjacent tiles, the matching adjacent tile captures the complementary section with substantial context. Truncating boundary annotations would train the network to ignore edge-proximity targets.

---

## 4. Evaluation of Candidate Annotation Strategies

Three distinct conversion strategies were evaluated across all 887 positive tiles:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STRATEGY COMPARISON                                                                                    │
├─────────────────────────┬──────────────────────┬──────────────────────┬────────────────────────────────┤
│ Metric                  │ Strategy A           │ Strategy B           │ Strategy C                     │
│                         │ (Pure Components)    │ (Proximity Grouped)  │ (Foreground Envelope)          │
├─────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────┤
│ Box Count Produced      │ 2237 boxes          │ 1675 boxes          │ 887 boxes            │
│ Boxes / Positive Tile   │ 2.52 boxes/tile        │ 1.89 boxes/tile        │ 1.00 box/tile          │
│ Box Count Reduction     │ Baseline (0.0%)      │ -25.12%                │ -60.35%                 │
│ Risk of Fragmentation   │ CRITICAL / SEVERE    │ MINIMAL / CONTROLLED │ ZERO                           │
│ Risk of Bloated Boxes   │ ZERO                 │ LOW                  │ HIGH (Includes vast mud seabed)│
│ Shadow Representation   │ Shadows excluded     │ Preserves shadow gap │ Voids included in box          │
└─────────────────────────┴──────────────────────┴──────────────────────┴────────────────────────────────┤
```

### Strategy A: One Bounding Box per Connected Component
- **Mechanics**: Every 8-connected foreground island produces an independent bounding box ($2,572$ boxes total).
- **Major Flaw**: A single shipwreck is split into 5 to 15 tiny separate boxes. During YOLO training, the network is penalized for detecting the vessel as a unified object. Furthermore, tiny 3-pixel speckle fragments receive equal loss weight as a 300-pixel hull.

### Strategy B: One Bounding Box per Grouped Proximity Region
- **Mechanics**: Components within 20 px (~1.5 m acoustic scale) are merged into a macro-region. Gaps wider than 20 px remain distinct objects (1675 boxes total).
- **Benefits**: Bridges internal acoustic shadow gaps and deck bulkheads while preserving the distinction between a main hull and a separate debris pile 50 meters away.

### Strategy C: Single Foreground Envelope Box per Tile
- **Mechanics**: Exactly 1 bounding box per positive tile encompassing all foreground pixels (887 boxes total).
- **Major Flaw**: If a tile contains a small wreck in the upper-left and an isolated timber in the lower-right, Strategy C draws a massive 600x600 box that is 98% empty seabed. YOLO struggles to learn feature localization when boxes have near-zero foreground density.

---

## 5. Visual Artifacts Generated

12 representative three-panel visual comparisons (`[SSS Image | Connected Components | Candidate Boxes A vs B vs C]`) have been saved to `outputs/annotation_samples/`:
- `sample_01_*.png`: Normal coherent target (Single component)
- `sample_02_*.png`: Fragmented wreck site (Disjoint highlight patches)
- `sample_03_*.png`: Boundary-crossing target
- `sample_04_*.png`: Large target footprint (Major hull span)
- `sample_05_*.png`: Tiny component target (< 20px speckle)
- `sample_06_*.png`: Aircraft debris structure (Corsair site)
- `sample_07_*.png`: Barge deck structure with shadow separation
- `sample_08_*.png`: Wooden schooner hull fragmentation (EB_Allen)
- `sample_09_*.png`: Artificial reef dispersed targets
- `sample_10_*.png`: Extreme debris field (High component count)
- `sample_11_*.png`: Right-padded tile target seam
- `sample_12_*.png`: Negative seabed control tile (0 detections)

---

## 6. RECOMMENDED ANNOTATION STRATEGY

### 6.1 Recommended Architecture: Hybrid Proximity Grouping (Strategy B with Size Thresholding)

1. **Definition of a Detection Instance**:
   - A detection instance represents a **coherent acoustic target structure or localized debris cluster**.
   - Components separated by < 20 px are merged into a single detection instance to bridge internal acoustic shadows and structural ribs.
   - Disconnected debris clusters separated by > 20 px remain distinct detection instances.

2. **Handling of Tiny Components (< 10 px)**:
   - Components with both w < 10 px and h < 10 px (and area < 50 px²) that are isolated from any larger structure are **filtered out as acoustic speckle noise**.
   - If a small component is within 20 px of a larger hull, it is absorbed into the parent instance rather than discarded.

3. **Handling of Boundary-Crossing Targets**:
   - Clipped boundary targets are **retained as valid training instances** provided their clipped area is >= 30 px² or dimension >= 8 px.
   - Because of the 20% tile overlap, the model learns to identify partial hull targets near edges without boundary penalty.

4. **Why This Is Optimal for YOLOv8n**:
   - YOLOv8's smallest feature stride is 8 px (P3 feature pyramid level). Merging tightly clustered components prevents contradictory gradient signals across adjacent anchor cells.
   - It eliminates significant redundant, fragmented labels while avoiding the severe background-noise contamination of single-envelope boxing (Strategy C).

5. **Information Lost When Converting Masks to Bounding Boxes**:
   - **Complex Shape Geometry**: Slanted hulls and crescent-shaped debris fields lose their angular orientation.
   - **Acoustic Shadow Depth**: Bounding boxes encompass both high-relief acoustic highlights and the acoustic shadow void behind them, treating the void as target space.
