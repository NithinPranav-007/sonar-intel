# Leakage-Safe Site-Level YOLO Dataset Splitting

**Document:** `docs/dataset/site_level_split.md`  
**Project:** SONAR-INTEL  
**Dataset:** AI4Shipwrecks ($640 \times 640\text{ px}$ Tiled Dataset)  
**Total Samples:** 8,356 Tiles (874 Positive Tiles, 7,482 Negative Control Tiles, 1,500 Bounding Boxes)  
**Split Partitioning:** Source Survey Site Identity ($\text{site\_id}$)  
**Target Proportions:** $70\%\text{ Train} \;/ \;15\%\text{ Validation} \;/ \;15\%\text{ Test}$  

---

## 1. Executive Summary & Why Random Tile Splitting Is Forbidden

In computer vision benchmarks, random sample shuffling (e.g. `train_test_split(shuffle=True)`) is common practice. In side-scan sonar (SSS) machine learning, **random tile-level splitting causes catastrophic data leakage and invalidates scientific evaluation**.

### 1.1 Overlapping Tiles and Spatial Correlation
The tiling stage slices continuous along-track swaths into $640 \times 640$ windows with a **20% spatial overlap (128 px)**.
- If Tile A ($y=0..640$) is placed in Train and Tile B ($y=512..1152$) is placed in Test, the test set contains a $128$-pixel duplicate slice of the exact acoustic highlights, seabed texture, and target geometry trained by the network.
- The model memorizes local acoustic speckle patterns rather than learning generalizable hydroacoustic target representations.

### 1.2 Multi-Pass Survey Correlation
AUV survey lines across a given shipwreck site (e.g. `DM_Wilson_01` through `DM_Wilson_22`) repeatedly image the same seabed geographic location from parallel swaths or reciprocal headings.
- If `DM_Wilson_05` is in Train while `DM_Wilson_06` is in Test, the model is evaluated on the exact same physical wreck hull under slightly altered grazing angles.
- Performance metrics (mAP50, mAP50-95) would be artificially inflated, failing completely when deployed to unmapped maritime areas.

> [!CRITICAL]
> **Strict Site-Level Partitioning Policy**:
> All tiles derived from any survey pass at a given site (e.g. all 22 swaths and 1,372 tiles of `DM_Wilson`) must reside **exclusively in a single split partition**.
> $$\text{Site}(\text{tile}_i) \in \{\text{Train}\} \implies \text{All tiles from Site} \in \{\text{Train}\}$$

---

## 2. Site-Level Partitioning Strategy & Optimization

The AI4Shipwrecks dataset comprises **29 distinct survey sites** (25 shipwreck/reef anomaly sites and 4 pure negative seabed control sites: `Exploratory_A`, `Exploratory_B`, `Exploratory_C`, and `Mischelley_Reef`).

To achieve the desired $70\% / 15\% / 15\%$ balance without fragmenting sites, a multi-objective combinatorial search optimized:
1. **Total Tile Balance**: Target $70\% / 15\% / 15\%$ ($\sim 5,849 / 1,253 / 1,253$ tiles).
2. **Positive Tile Balance**: Target $70\% / 15\% / 15\%$ ($\sim 612 / 131 / 131$ positive tiles).
3. **Bounding Box Balance**: Target $70\% / 15\% / 15\%$ ($\sim 1,050 / 225 / 225$ boxes).
4. **Negative Control Retention**: Ensuring Train, Validation, and Test each receive representative pure negative control survey sites.

---

## 3. Exact Split Partition & Distribution

### 3.1 Split Statistics

| Split | Sites | Tiles | Tile Pct | Positive Tiles | Pos Pct | Negative Tiles | Bounding Boxes | Box Pct |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | **17** | **5,844** | **69.94%** | **612** | **70.02%** | 5,232 | **1,034** | **68.93%** |
| **VALIDATION** | **7** | **1,256** | **15.03%** | **130** | **14.87%** | 1,126 | **195** | **13.00%** |
| **TEST** | **5** | **1,256** | **15.03%** | **132** | **15.10%** | 1,124 | **271** | **18.07%** |
| **TOTAL** | **29** | **8,356** | **100.0%** | **874** | **100.0%** | **7,482** | **1,500** | **100.0%** |

### 3.2 Site Assignments

#### TRAIN Split (17 Sites, 5,844 Tiles)
1. `Barge_No_1` (15 swaths, 388 tiles, 40 pos, 45 boxes)
2. `Corsair` (4 swaths, 64 tiles, 3 pos, 3 boxes)
3. `DM_Wilson` (22 swaths, 1,372 tiles, 89 pos, 141 boxes)
4. `EB_Allen` (24 swaths, 384 tiles, 56 pos, 67 boxes)
5. `Egyptian` (15 swaths, 236 tiles, 50 pos, 154 boxes)
6. `Exploratory_A` (2 swaths, 216 tiles, 0 pos, 0 boxes) — *Negative Control*
7. `Grecian` (5 swaths, 48 tiles, 19 pos, 27 boxes)
8. `Haltiner_Barge` (13 swaths, 512 tiles, 18 pos, 31 boxes)
9. `James_Davidson` (4 swaths, 216 tiles, 14 pos, 16 boxes)
10. `Lucinda_van_Valkenburg` (19 swaths, 372 tiles, 76 pos, 125 boxes)
11. `Mischelley_Reef` (8 swaths, 428 tiles, 0 pos, 0 boxes) — *Negative Control*
12. `Monohansett` (5 swaths, 296 tiles, 27 pos, 45 boxes)
13. `Monrovia` (8 swaths, 128 tiles, 61 pos, 118 boxes)
14. `Montana` (8 swaths, 188 tiles, 75 pos, 102 boxes)
15. `Near_Shore` (6 swaths, 448 tiles, 29 pos, 50 boxes)
16. `Oscar_T_Flint` (22 swaths, 468 tiles, 31 pos, 67 boxes)
17. `Pewabic` (5 swaths, 80 tiles, 24 pos, 43 boxes)

#### VALIDATION Split (7 Sites, 1,256 Tiles)
1. `Artificial_Reef` (6 swaths, 120 tiles, 2 pos, 2 boxes)
2. `Exploratory_C` (2 swaths, 296 tiles, 0 pos, 0 boxes) — *Negative Control*
3. `Heart_Failure` (12 swaths, 152 tiles, 21 pos, 38 boxes)
4. `Isaac_M_Scott` (6 swaths, 104 tiles, 23 pos, 25 boxes)
5. `Shamrock` (6 swaths, 112 tiles, 12 pos, 12 boxes)
6. `WH_Gilbert` (6 swaths, 64 tiles, 16 pos, 38 boxes)
7. `WP_Thew` (17 swaths, 408 tiles, 56 pos, 80 boxes)

#### TEST Split (5 Sites, 1,256 Tiles)
1. `Corsican` (6 swaths, 96 tiles, 7 pos, 8 boxes)
2. `DR_Hanna` (5 swaths, 100 tiles, 11 pos, 12 boxes)
3. `Exploratory_B` (13 swaths, 544 tiles, 0 pos, 0 boxes) — *Negative Control*
4. `Viator` (11 swaths, 236 tiles, 44 pos, 51 boxes)
5. `WP_Rend` (11 swaths, 280 tiles, 70 pos, 200 boxes)

---

## 4. Rigorous Leakage Verification

Programmatic set intersection tests verify zero spatial or identifier leakage across splits:

- **Site Intersections**:
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 common sites)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 common sites)
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 common sites)
- **Source Swath Image Intersections**:
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 common swaths)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 common swaths)
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 common swaths)
- **Tile ID Intersections**:
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 common tiles)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 common tiles)
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 common tiles)

**Leakage Test Status:** **PASS (Zero Leakage)**

---

## 5. Known Limitations

- **Coarse Site Granularity**: Because large sites like `DM_Wilson` account for 1,372 tiles (16.4% of the entire dataset alone), exact mathematical percentages ($70.000\% / 15.000\% / 15.000\%$) are impossible without splitting swaths. Achieving $69.94\% / 15.03\% / 15.03\%$ is optimal under the constraint of site integrity.
- **Geographic Representation**: Test set performance specifically reflects model generalization to unseen shipwreck architectures (`WP_Rend`, `Viator`, `DR_Hanna`, `Corsican`) in new acoustic bottom conditions.
