# SSS Mask-to-YOLO Detection Annotation Conversion

**Document:** `docs/dataset/yolo_annotation_conversion.md`  
**Project:** SONAR-INTEL  
**Dataset:** AI4Shipwrecks ($640 \times 640\text{ px}$ Tiled Dataset)  
**Input Scope:** 8,356 Total Tiles (887 Positive Tiles, 7,469 Negative Control Tiles)  
**Target Detection Framework:** Ultralytics YOLOv8n (Normalized $[0, 1]$ Single-Class Coordinates)  
**Annotation Algorithm:** Hybrid Local Proximity Grouping with Tiny-Component Filtering  

---

## 1. Executive Summary & Rationale

Side-scan sonar (SSS) imagery presents unique acoustic characteristics that make direct conversion from binary segmentation masks to bounding-box object detection challenging.

In side-scan sonar, an individual physical shipwreck (such as an 80-meter Great Lakes steam freighter) frequently presents as a collection of disjoint acoustic highlight patches separated by:
1. **Internal acoustic shadows**: Gunwales, bulkheads, and engine blocks occlude sonar pings, casting zero-backscatter shadow occlusions across the hull.
2. **Grazing-angle dropouts**: Areas of low reflectivity inside open cargo holds.
3. **Scattered structural debris**: Dislodged boilers, masts, winches, and hull plates scattered within 5–25 meters of the vessel.

### 1.1 Why Strategy A (Pure Connected Components) Was Rejected
Assuming $1\text{ connected component} = 1\text{ detection}$ splits a single contiguous vessel into 5 to 15 tiny separate boxes. During YOLO training:
- The detector's loss function is penalized when it correctly predicts the entire vessel as a single cohesive object.
- Tiny 3-pixel speckle fragments receive identical loss weight as a 300-pixel hull, destabilizing gradient descent.

### 1.2 Why Strategy C (Single Foreground Envelope per Tile) Was Rejected
Enclosing all foreground pixels in a positive tile within a single bounding box generates bloated boxes whenever dispersed debris pieces reside far apart. A tile with a target in the top-left and an isolated fragment in the bottom-right produces a massive $600 \times 600$ box containing $>95\%$ empty mud seabed, degrading feature localization.

### 1.3 Why Hybrid Local Proximity Grouping Was Selected
Local proximity grouping merges components whose bounding boundaries are within an empirical threshold of **20 pixels**, while maintaining independent boxes for distant debris fields. This bridges internal acoustic shadows and broken bulkheads without bloating bounding boxes.

---

## 2. Mathematical Definition of the Annotation Algorithm

### 2.1 Component Extraction
For each binary mask $M \in \{0, 1\}^{640 \times 640}$, connected components are extracted using standard 8-connectivity:
$$\mathcal{C} = \{c_1, c_2, \dots, c_N\}$$
Each component $c_i$ is characterized by bounding coordinates $(x_i, y_i, w_i, h_i)$, pixel area $A_{px}(c_i)$, and bounding box area $A_{box}(c_i) = w_i \times h_i$.

### 2.2 Boundary-to-Boundary Distance
For two rectangular components $c_i$ and $c_j$, the minimum boundary-to-boundary distance $D_{gap}(c_i, c_j)$ is computed as:
$$\Delta x = \max(0, \max(x_i, x_j) - \min(x_i + w_i, x_j + w_j))$$
$$\Delta y = \max(0, \max(y_i, y_j) - \min(y_i + h_i, y_j + h_j))$$
$$D_{gap}(c_i, c_j) = \sqrt{\Delta x^2 + \Delta y^2}$$

### 2.3 Empirical Merge Threshold ($\text{MERGE\_DISTANCE\_PX} = 20$)
> [!IMPORTANT]
> **Image-Space Heuristic**:
> The 20-pixel threshold is strictly an **empirical image-space threshold**. The AI4Shipwrecks dataset is image-only and lacks vehicle altitude, heading, and slant-range metadata; therefore, no claim is made that 20 pixels corresponds to a specific metric physical distance in meters.

### 2.4 Tiny-Component Filtering Heuristic
Components satisfying all three conditions:
$$w_i < 10\text{ px} \quad \text{AND} \quad h_i < 10\text{ px} \quad \text{AND} \quad A_{px}(c_i) < 50\text{ px}^2$$
are evaluated. If an isolated component does **not** reside within $20\text{ px}$ of any larger component, it is filtered out as acoustic speckle noise. If it is within $20\text{ px}$ of a larger component, it is absorbed into that component's group.

### 2.5 Group Bounding Box
For a clustered group $\mathcal{G} = \{c_k\}$, the unified bounding box is defined by:
$$x_{\min} = \min_{c \in \mathcal{G}} x_c, \quad y_{\min} = \min_{c \in \mathcal{G}} y_c$$
$$x_{\max} = \max_{c \in \mathcal{G}} (x_c + w_c), \quad y_{\max} = \max_{c \in \mathcal{G}} (y_c + h_c)$$

---

## 3. YOLOv8 Annotation Format & Normalization

Coordinates are normalized to $[0, 1]$ relative to the $640 \times 640$ tile dimensions:
$$x_{center} = \frac{x_{\min} + x_{\max}}{2 \times 640.0}, \quad y_{center} = \frac{y_{\min} + y_{\max}}{2 \times 640.0}$$
$$w_{norm} = \frac{x_{\max} - x_{\min}}{640.0}, \quad h_{norm} = \frac{y_{\max} - y_{\min}}{640.0}$$

Every detection entry in the `.txt` label file adheres to standard Ultralytics YOLO syntax:
```
<class_id> <x_center> <y_center> <width> <height>
0 0.500000 0.420000 0.180000 0.220000
```
- **Single Class**: `0 = artificial_anomaly`
- **Negative Tiles**: Contain an empty `0-byte` text file to train background false-positive suppression.

---

## 4. Boundary Targets & Overlap Alignment

When targets cross the tile boundary:
- Boundary-clipped targets are **retained as valid training instances**.
- Because adjacent tiles share a 20% spatial overlap ($128\text{ px}$ stride), the neighboring tile captures the complementary section with broader spatial context.
- Deleting boundary targets would train the network to ignore edge-proximity shipwrecks.

---

## 5. Information Inevitably Lost When Converting Segmentation to Bounding Boxes

1. **Orientation & Skew**: Diagonal shipwrecks lose their true heading angle within axis-aligned bounding boxes.
2. **Acoustic Shadow Distinction**: Bounding boxes encompass both high-relief acoustic highlights and the acoustic shadow void behind them, treating the void as target space.
3. **Internal Hole Details**: Open cargo hatches and interior hollows cannot be represented within 2D boxes.

---

## 6. Known Limitations & Constraints

- **No Semantic Subclasses**: All targets are categorized under `0 = artificial_anomaly`. Distinctions between steamships, schooners, aircraft, and artificial reef cages are not separated because source ground truth is binary.
- **No Real-World Metric Scaling**: Bounding boxes reflect image-space coordinates without slant-range or altitude correction.
