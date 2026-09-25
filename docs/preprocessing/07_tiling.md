# SSS Image & Mask Tiling Specification & Methodology

**Document:** `docs/preprocessing/07_tiling.md`  
**Project:** SONAR-INTEL  
**Dataset:** AI4Shipwrecks (`data/raw/AI4Shipwrecks/`)  
**Pipeline Stage:** Isolated Data Preparation / Tiling Stage  
**Tile Configuration:** $640 \times 640\text{ px}$, $20\%\text{ overlap}$ ($\text{stride} = 512\text{ px}$)  

---

## 1. Executive Summary & Rationale

Side-scan sonar (SSS) waterfall imagery is characterized by an extreme aspect ratio: a fixed across-track width of **1,728 pixels** paired with variable along-track lengths ranging from **13 to 18,745 pixels**. 

Standard convolutional neural networks and object detectors (e.g. YOLOv8) require square, fixed-dimension inputs. Resizing an 18,745-pixel swath directly to $640 \times 640$ would introduce catastrophic spatial distortion ($>29:1$ aspect ratio compression), obliterating fine acoustic highlight-and-shadow structures.

Therefore, an overlapping spatial sliding-window tiling pipeline is implemented to transform continuous swaths into standardized $640 \times 640$ training patches while rigorously maintaining pixel-level alignment with the ground-truth segmentation masks.

> [!NOTE]
> **Engineering Baseline Starting Point**:
> The $640 \times 640$ tile size is an engineering starting point selected from empirical target-size distribution analysis (where $95.5\%$ of target components measure $\le 640\text{ px}$) and consumer GPU VRAM constraints (RTX 3050 4GB). It is **not claimed to be globally optimal**, but provides an efficient, standard receptive field for YOLO model exploration.

---

## 2. Tiling Methodology & Geometry

### 2.1 Across-Track Tiling (Width = 1,728 px)
With tile width $W = 640\text{ px}$ and stride $S = 512\text{ px}$:
- **Column 0**: $x \in [0, 640)$ — Port far-field to near-nadir.
- **Column 1**: $x \in [512, 1152)$ — Port nadir, water column, and starboard nadir.
- **Column 2**: $x \in [1024, 1664)$ — Starboard near-field to mid-range.
- **Column 3**: $x \in [1536, 1728)$ — Starboard far-field ($192\text{ px}$ valid content) padded deterministically on the right with $448\text{ px}$ of zero-padding.
- **Total Columns**: Exactly 4 columns per along-track row across the entire swath width.

### 2.2 Along-Track Tiling (Variable Height $H$)
With tile height $H = 640\text{ px}$ and stride $S = 512\text{ px}$:
- Rows are indexed $r = 0, 1, 2, \dots$ at $y = r \times 512$.
- If $y + 640 \le H$, a full $640\text{ px}$ slice is extracted with zero vertical padding.
- If $y + 640 > H$, the remaining $H - y$ lines are extracted and padded at the bottom with $640 - (H - y)$ pixels of zero-padding.

---

## 3. Boundary & Short-Swath Handling

### 3.1 Deterministic Padding Strategy
- **Padding Values**: Padded regions are filled with **0** in both image and mask tiles. In sonar acoustics, 0 represents an acoustic void (identical to acoustic shadow or water column), ensuring models do not mistake padding for high-backscatter seabed.
- **Metadata Logging**: For every tile, four explicit padding coordinates are logged: `padding_left`, `padding_right`, `padding_top`, `padding_bottom`.
- **Zero Truncation**: No boundary pixels are silently dropped. Every pixel of the original source swath is covered by at least one tile.

### 3.2 Short Swaths ($H < 640\text{ px}$)
- Swaths shorter than 640 pixels (including the 4 anomalous survey fragments under 64 pixels identified during quality control: `Grecian_05` [13px], `Exploratory_B_03` [29px], `WP_Rend_03` [29px], `DM_Wilson_18` [61px]) are **padded to $640 \times 640$ rather than deleted**.
- Swaths with $H < 64\text{ px}$ are flagged in metadata as `is_training_suitable = False` (`TRAINING_UNSUITABLE`), ensuring researchers can filter them out during training while maintaining 100% data auditability.

---

## 4. Mask Handling & Target Fragment Preservation

- **Exact Value Space**: Masks are processed strictly with nearest-neighbor deterministic slicing. Mask values remain strictly binary: $\{0 = \text{background}, 1 = \text{shipwreck target}\}$.
- **Boundary-Crossing Targets**: When a target structure spans across a tile seam, the mask slice is **never deleted or discarded**. The clipped target region is preserved intact in both adjacent tiles.
- **Boundary Flags**: Each tile logs `has_boundary_target = True` if any positive target pixel intersects the tile boundary ($x=0, y=0, x=639, y=639$, or the valid slice edge before padding).
- **Acoustic Regions vs. Objects**: Connected components in masks are logged purely as acoustic regions, not independent physical vessels.

---

## 5. Provenance & Anti-Leakage Architecture

### 5.1 Deterministic Naming Convention
Every tile receives an immutable, human-readable identifier encoding its source swath, along-track row, and across-track column:
```
{source_image_basename}__tile_r{row_index:04d}_c{column_index:04d}
Example: Corsair_02__tile_r0003_c0002.png
```

### 5.2 Preservation of Site Identity
Every tile's metadata record stores:
- `source_image`: e.g. `Corsair_02.png`
- `site_id`: e.g. `Corsair`
- `source_split`: `train`, `test`, or `extras`

> [!IMPORTANT]
> **Zero Spatial Data Leakage**:
> Random assignment of tiles into train/val/test splits is strictly forbidden. Because adjacent overlapping tiles share acoustic backscatter features, splitting must be performed at the **site level** (`site_id`), ensuring all tiles from a given shipwreck site reside exclusively in either train or test.

---

## 6. Reconstruction & Coverage Verification

For full auditability, a reconstruction test verifies that the complete original source swath can be reconstructed from the generated tiles without gaps:
- **Coverage**: **100.0%** of all source pixels are represented.
- **Uncovered Pixels**: **0 pixels**.
- **Overlap**: A 20% along-track and across-track overlap ensures targets near tile seams are represented with full spatial context in adjacent windows.
