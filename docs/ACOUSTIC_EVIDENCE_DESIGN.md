# SONAR-INTEL: Multi-Modal Acoustic Evidence Extraction Layer

**Document ID:** `DOC-ACOUSTIC-EVIDENCE-2026.09`  
**Role:** Person 2 — ML / Model Architecture & Verification Lead  
**Component:** Evidence Extraction Layer (`ml/evidence/`)  
**Status:** **IMPLEMENTED & VERIFIED (40/40 TESTS PASSING)**  
**Downstream Consumer:** Stage 3 Multi-Evidence Fusion (Deferred / Future Work)  

---

## 1. Architectural Purpose & Pipeline Flow

The SONAR-INTEL candidate evaluation pipeline follows a modular cascade:

```
Full SSS Waterfall Tile (640x640)
           │
           ▼
Stage 1: YOLO Candidate Detector (SEARCH)
           │  └─ Outputs bounding boxes [x1, y1, x2, y2, conf, class_id]
           ▼
Candidate ROI Extraction (ROIExtractor)
           │  └─ Extracts localized crop with acoustic context margins
           ▼
Stage 2: SW-Net Semantic Verifier (VERIFICATION)
           │  └─ Directional physical priors & segmentation mask (Shipwreck only)
           ▼
Stage 2.5: Multi-Modal Evidence Extraction (EvidenceExtractor)
           │  └─ Measures & returns structured raw evidence features
           ▼
[Stage 3: Multi-Evidence Fusion & Scoring — DEFERRED]
           │  └─ (Combines evidence into final confidence score)
           ▼
Canonical Contact Transformation & Geolocation
```

> [!IMPORTANT]
> **Strict Separation of Concepts:**  
> 1. **MEASURED FEATURE:** A measurable, physics-based, or geometric image property (e.g. shadow deficit in intensity levels, compactness $C \in [0, 1]$, entropy $H \in [0, 8]$ bits).
> 2. **NORMALIZED FEATURE:** Mathematically scaled continuous feature in $[0.0, 1.0]$ for numerical stability.
> 3. **FINAL FUSION SCORE:** A learned or weighted combination yielding an operational decision (e.g. `final_confidence`, `priority`).  
> **This layer implements ONLY Measured & Normalized Features.** It does NOT make detection decisions or apply heuristic thresholding.

---

## 2. Why Detector Evidence Alone is Insufficient

Single-stage bounding-box detectors (e.g., YOLOv8s) produce rectangular candidate anchors based on RGB/grayscale texture, but lack acoustic physical grounding:
- **Clutter Susceptibility:** Rocky reefs, sand ripples, and bathymetric drop-offs generate rectangular backscatter that triggers high YOLO confidence.
- **No Acoustic Shadow Verification:** True 3D elevated targets cast down-range acoustic shadow zones away from the nadir line. Standard detectors cannot verify whether the shadow geometry matches the target height.
- **Rectangular Dilution:** Axis-aligned bounding boxes encompass large portions of background seafloor, diluting target backscatter.

---

## 3. Evidence Categories & Extraction Methodologies

The evidence layer (`ml/evidence/`) extracts four decoupled evidence categories:

### 3.1 Detector Evidence (`DetectorEvidence`)
Captures raw detector outputs without modification or normalization:
- `class_name` & `class_id`
- `detector_confidence` (raw model confidence $[0.0, 1.0]$)
- `bbox` $[x_1, y_1, x_2, y_2]$, `bbox_width`, `bbox_height`, `bbox_area`
- `aspect_ratio` ($\max(w, h) / \min(w, h)$)
- `class_probability_margin` (logit gap between top-1 and top-2 classes)

### 3.2 Segmentation Evidence (`SegmentationEvidence`)
Explicitly reflects the data reality of the underlying corpus:
- **Shipwreck Class (`class_id=2`):** May have `available=True`, `source="SW-Net"`, and mask metrics when a trained SW-Net checkpoint exists.
- **Other Classes (`pipeline`, `mine_contact`, `crab_pot`, `ghost_net`):** Explicitly flagged as `available=False` with `mask_status="UNAVAILABLE_CLASS_NOT_SUPPORTED"`.
- *No fake or pseudo-labeled ground truth is manufactured.*

### 3.3 Morphology Evidence (`MorphologyEvidence`)
Extracts raw geometric shape descriptors from the segmentation mask (when available) or bounding geometry:
- **Area ($A$):** Segmented pixel count.
- **Perimeter ($P$):** Contour arc length in pixels.
- **Compactness ($C$):** Isoperimetric quotient $4 \pi A / P^2 \in [0.0, 1.0]$ (1.0 for a perfect circle).
- **Principal Axis Orientation ($\theta$):** Computed via central moments:
  $$\theta = \frac{1}{2} \arctan\left( \frac{2\mu_{11}}{\mu_{20} - \mu_{02}} \right) \in [-90.0^\circ, +90.0^\circ]$$
- **Elongation:** Major axis / minor axis length ratio ($\ge 1.0$) from moment inertia tensor eigenvalues.
- **Eccentricity:** $\sqrt{1 - (b/a)^2} \in [0.0, 1.0]$.
- **Solidity:** $\text{Area} / \text{ConvexHullArea} \in [0.0, 1.0]$.

### 3.4 Acoustic Evidence (`AcousticEvidence`)
Physics-grounded sonar backscatter, shadow, and texture measurements:

| Sub-Category | Feature Name | Definition & Input | Physical Range | Failure / Edge Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Highlight** | `highlight_mean` | Mean 8-bit intensity inside candidate patch | $[0.0, 255.0]$ | $0.0$ if empty crop |
| | `local_background_mean` | Mean intensity of ambient context ring excluding target | $[0.0, 255.0]$ | Image mean if ring empty |
| | `local_contrast_ratio` | $\text{highlight\_mean} / \max(1.0, \text{background\_mean})$ | $[0.0, \infty)$ | $1.0$ (no contrast) |
| | `normalized_highlight_strength` | $(\text{highlight\_mean} - \text{background\_mean}) / 255.0$ | $[-1.0, 1.0]$ | $0.0$ if identical |
| **Shadow** | `shadow_status` | Classification of down-range deficit zone | `PRESENT`, `ABSENT`, `UNCERTAIN`, `OUT_OF_BOUNDS` | `OUT_OF_BOUNDS` at swath edge |
| | `shadow_deficit` | $\max(0.0, \text{background\_mean} - \text{shadow\_mean})$ | $[0.0, 255.0]$ | $0.0$ if no shadow |
| | `shadow_contrast_ratio` | $\text{background\_mean} / \max(1.0, \text{shadow\_mean})$ | $[1.0, \infty)$ | $1.0$ if no deficit |
| | `shadow_to_highlight_area_ratio` | $\text{Area}_{\text{shadow}} / \max(1, \text{Area}_{\text{target}})$ | $[0.0, \infty)$ | $0.0$ if absent |
| | `shadow_length_px` | Down-range pixel span of dark columns | $[0.0, W_{\text{shadow}}]$ | $0.0$ if absent |
| **Texture** | `local_variance` | Intensity variance inside target ROI | $[0.0, \infty)$ | $0.0$ if flat |
| | `entropy` | Shannon entropy of grayscale distribution: $-\sum p_i \log_2(p_i)$ | $[0.0, 8.0]$ bits | $0.0$ if constant |
| | `gradient_magnitude_mean` | Mean Sobel edge gradient $\sqrt{G_x^2 + G_y^2}$ | $[0.0, \infty)$ | $0.0$ if smooth |
| | `edge_density` | Fraction of Canny edge pixels in candidate | $[0.0, 1.0]$ | $0.0$ if no edges |
| **Direction** | `dominant_gradient_orientation_deg` | Mean gradient vector angle via double-angle projection | $[-90.0^\circ, +90.0^\circ]$ | $0.0$ if isotropic |
| | `directional_consistency` | Resultant vector length / total magnitude (coherence) | $[0.0, 1.0]$ | $0.0$ if random noise |

---

## 4. Acoustic Shadow & Nadir Geometry Rules

In side-scan sonar, acoustic shadows cast **strictly away from the nadir path**:
- **Port Swath ($x < \text{nadir\_x}$):** Acoustic wavefront travels left ($-\Delta x$), casting shadows to the **left** of the target highlight.
- **Starboard Swath ($x > \text{nadir\_x}$):** Acoustic wavefront travels right ($+\Delta x$), casting shadows to the **right** of the target highlight.

> [!NOTE]
> **Domain Principle — Shadow Absence $\neq$ False Alarm:**  
> Flat or low-profile targets (e.g., sunken cables, flat derelict fishing nets, buried pipelines, or targets at grazing incidence near nadir) legitimately cast minimal or zero shadow. The evidence layer records `shadow_status="ABSENT"` or `shadow_status="UNCERTAIN"` without forcing a negative detection decision.

---

## 5. Output Schema

The evidence extraction layer outputs a `CandidateEvidence` dataclass serializable to JSON:

```json
{
  "detector": {
    "class_name": "shipwreck",
    "class_id": 2,
    "detector_confidence": 0.88,
    "bbox": [350, 200, 400, 250],
    "bbox_width": 50,
    "bbox_height": 50,
    "bbox_area": 2500,
    "aspect_ratio": 1.0,
    "class_probability_margin": null
  },
  "segmentation": {
    "available": true,
    "source": "SW-Net",
    "confidence": 0.85,
    "mask_status": "AVAILABLE",
    "mask_area_px": 900,
    "mask_to_box_ratio": 0.36
  },
  "morphology": {
    "source": "segmentation_mask",
    "area_px": 900.0,
    "width_px": 30.0,
    "height_px": 30.0,
    "aspect_ratio": 1.0,
    "perimeter_px": 120.0,
    "compactness": 0.7854,
    "eccentricity": 0.0,
    "elongation": 1.0,
    "orientation_deg": 15.0,
    "solidity": 1.0,
    "status": "VALID"
  },
  "acoustic": {
    "highlight_mean": 210.5,
    "highlight_max": 245.0,
    "local_background_mean": 62.0,
    "local_background_std": 8.5,
    "local_contrast_ratio": 3.395,
    "normalized_highlight_strength": 0.582,
    "shadow_status": "PRESENT",
    "shadow_mean": 12.0,
    "shadow_contrast_ratio": 5.167,
    "shadow_deficit": 50.0,
    "shadow_candidate_area_px": 1800,
    "shadow_to_highlight_area_ratio": 0.72,
    "shadow_length_px": 36.0,
    "local_variance": 420.5,
    "entropy": 5.82,
    "gradient_magnitude_mean": 34.2,
    "edge_density": 0.145,
    "dominant_gradient_orientation_deg": 12.5,
    "directional_consistency": 0.742
  },
  "status": {
    "overall": "COMPLETE",
    "warnings": [],
    "is_touching_boundary": false,
    "is_low_contrast": false,
    "is_degenerate_box": false
  },
  "raw_metadata": {}
}
```

---

## 6. What is NOT Implemented Yet

1. ❌ **Evidence Fusion:** No weighted averaging, Bayesian updating, or Dempster-Shafer evidential reasoning is applied yet.
2. ❌ **Classification Thresholds:** No arbitrary cutoff logic (e.g. `if shadow > X`) is implemented.
3. ❌ **Supervised Non-Shipwreck Segmentation:** Bounding box targets are not converted into fake masks.
4. ❌ **LearnLoop / Retraining:** Active learning triggers are deferred to subsequent stages.
