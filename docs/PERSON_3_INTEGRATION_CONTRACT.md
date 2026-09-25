# PERSON 3: Contact Package & Evidence API Integration Contract

**Document ID:** `DOC-PERSON3-CONTRACT-2026.09`  
**Author:** Person 2 (ML / Evidence Fusion Lead)  
**Target Engineer:** Person 3 (Frontend / Fullstack / API Integration Lead)  
**Status:** **ACTIVE CONTRACT — BACKEND & ML ENGINE READY**  
**Schemas & Implementation:** [`ml/fusion/contact_package.py`](file:///c:/Users/Asus/Desktop/SONAR-INTEL/ml/fusion/contact_package.py)  

---

## 1. Executive Summary

This document establishes the **authoritative machine-readable Contact Package Contract** delivered by the ML inference and evidence fusion engine to **Person 3**.

The contact package decouples the user-facing web dashboard and API endpoints from the internal ML model mechanics. Person 3 can render all triage cards, evidence gauges, geospatial map layers, and detailed inspection panels directly against this JSON schema without knowing the internal YOLO or PyTorch architecture.

---

## 2. Complete Contact Package JSON Schema

```json
{
  "contact_id": "CNT-000001",
  "survey_id": "SURV_20260901_190201",
  "timestamp": "2026-09-07T14:25:00.000Z",
  "class_name": "mine_like_contact",
  "class_id": 4,
  "raw_detector_confidence": 0.91,
  "fused_score": 0.88,
  "final_confidence": 0.87,
  "priority": "HIGH",
  "review_status": "AI_CANDIDATE",

  "bbox": [350, 200, 400, 250],

  "location": {
    "latitude": 45.062123,
    "longitude": -83.312456,
    "depth_m": 18.5,
    "uncertainty_m": 1.5,
    "status": "ESTIMATED"
  },

  "modality_breakdown": {
    "detector": 0.91,
    "segmentation": null,
    "acoustic": 0.89,
    "morphology": 0.86
  },

  "indicators": {
    "acoustic": {
      "highlight_mean": 215.0,
      "local_contrast_ratio": 3.58,
      "shadow_status": "PRESENT",
      "shadow_deficit": 50.0,
      "texture_entropy": 5.60,
      "edge_density": 0.12
    },
    "morphological": {
      "aspect_ratio": 1.0,
      "elongation": 1.0,
      "orientation_deg": 10.0,
      "compactness": 0.7854,
      "solidity": 1.0,
      "source": "bbox_geometry"
    }
  },

  "evidence": {
    "detector": {
      "class_name": "mine_like_contact",
      "class_id": 4,
      "detector_confidence": 0.91,
      "bbox": [350, 200, 400, 250],
      "bbox_width": 50,
      "bbox_height": 50,
      "bbox_area": 2500,
      "aspect_ratio": 1.0,
      "class_probability_margin": 0.35,
      "class_probabilities": {
        "mine_like_contact": 0.91,
        "crab_pot": 0.04,
        "shipwreck": 0.03,
        "submarine_pipeline": 0.01,
        "ghost_net": 0.01
      },
      "tile_relative_coords": {
        "center_x": 0.5859,
        "center_y": 0.3516,
        "width": 0.0781,
        "height": 0.0781
      }
    },
    "segmentation": {
      "available": false,
      "source": null,
      "confidence": null,
      "mask_status": "UNAVAILABLE_CLASS_NOT_SUPPORTED",
      "mask_area_px": null,
      "mask_to_box_ratio": null,
      "mask_coverage": null,
      "mask_compactness": null,
      "mask_orientation_deg": null,
      "mask_elongation": null
    },
    "morphology": {
      "source": "bbox_geometry",
      "area_px": 2500.0,
      "width_px": 50.0,
      "height_px": 50.0,
      "aspect_ratio": 1.0,
      "perimeter_px": 200.0,
      "compactness": 0.7854,
      "eccentricity": 0.0,
      "elongation": 1.0,
      "orientation_deg": 10.0,
      "solidity": 1.0,
      "convexity": 1.0,
      "status": "ESTIMATED_FROM_BBOX"
    },
    "acoustic": {
      "highlight_mean": 215.0,
      "highlight_max": 245.0,
      "local_background_mean": 60.0,
      "local_background_std": 10.0,
      "local_contrast_ratio": 3.583,
      "normalized_highlight_strength": 0.608,
      "shadow_status": "PRESENT",
      "shadow_mean": 10.0,
      "shadow_contrast_ratio": 6.0,
      "shadow_deficit": 50.0,
      "shadow_candidate_area_px": 1500,
      "shadow_to_highlight_area_ratio": 0.60,
      "shadow_length_px": 30.0,
      "local_variance": 380.0,
      "entropy": 5.60,
      "gradient_magnitude_mean": 32.0,
      "edge_density": 0.12,
      "dominant_gradient_orientation_deg": 10.0,
      "directional_consistency": 0.72
    },
    "fused": {
      "fused_score": 0.88,
      "modality_scores": {
        "detector": 0.91,
        "segmentation": null,
        "acoustic": 0.89,
        "morphology": 0.86
      },
      "normalized_features": {
        "highlight_contrast": 1.0,
        "shadow_deficit": 0.833,
        "texture_entropy": 0.655,
        "edge_gradient": 0.60,
        "compactness": 0.785,
        "solidity": 1.0,
        "convexity": 1.0
      },
      "weights_applied": {
        "detector": 0.50,
        "acoustic": 0.30,
        "morphology": 0.20
      },
      "fusion_status": "DEGRADED_3_MODALITIES",
      "fusion_method": "baseline_adaptive_linear_v1"
    },
    "calibrated_confidence": 0.87,
    "status": {
      "overall": "PARTIAL",
      "warnings": [],
      "is_touching_boundary": false,
      "is_low_contrast": false,
      "is_degenerate_box": false
    }
  },

  "assets": {
    "roi_bounds": [350, 200, 400, 250],
    "crop_offset": [320, 180],
    "crop_size": [110, 90],
    "roi_ref": "/api/assets/roi/CNT-000001.png",
    "mask_ref": null
  },

  "model": {
    "detector_name": "DRISHTI-YOLOv8s",
    "detector_version": "baseline-v1",
    "swnet_version": "swnet-direction-aware-v1.0",
    "fusion_version": "baseline_adaptive_linear_v1",
    "calibration_version": "platt-mvp-v1.0"
  }
}
```

---

## 3. Field Semantics & Specification

### 3.1 Top-Level Contact Fields
| Field Name | Type | Required? | Semantics |
| :--- | :--- | :---: | :--- |
| `contact_id` | `string` | **YES** | Unique identifier (e.g. `"CNT-000001"`, `"C001"`). |
| `survey_id` | `string` | **YES** | Foreign key to parent acoustic survey. |
| `timestamp` | `string` | **YES** | ISO-8601 UTC timestamp of acquisition/detection. |
| `class_name` | `string` | **YES** | Canonical class name (`"mine_like_contact"`, `"shipwreck"`, `"submarine_pipeline"`, `"ghost_net"`, `"crab_pot"`). |
| `class_id` | `int` | **YES** | Zero-indexed canonical class ID (`0` to `4`). |
| `raw_detector_confidence` | `float` | **YES** | Unaltered YOLO confidence score in $[0.0, 1.0]$. |
| `fused_score` | `float` | **YES** | Composite multi-modal evidence score in $[0.0, 1.0]$. |
| `final_confidence` | `float` | **YES** | Calibrated probability in $[0.0, 1.0]$ displayed to operator. |
| `priority` | `string` | **YES** | Operational triage priority (`"HIGH"`, `"MEDIUM"`, `"LOW"`). |
| `review_status` | `string` | **YES** | Human review state (`"AI_CANDIDATE"`, `"CONFIRMED"`, `"FALSE_POSITIVE"`, `"UNCERTAIN"`). |
| `bbox` | `List[int]` | **YES** | Target bounding box in parent image $[x_1, y_1, x_2, y_2]$. |

---

### 3.2 Geospatial Location Semantics (`location`)
> [!IMPORTANT]
> **Zero Coordinate Fabrication:** If navigation logs were not recorded with the survey, `location.status` is set to `"UNAVAILABLE"`, and `latitude` / `longitude` are strictly `null`. Person 3 should show `"Navigation Unavailable"` instead of plotting at $(0, 0)$.

| Status Code | Latitude / Longitude | UI Rendering Guidance |
| :--- | :---: | :--- |
| `"ESTIMATED"` | Valid coordinates present | Render pin on MapLibre map layer. Uncertainty radius shown. |
| `"VERIFIED"` | Ground-truth coordinates | Render verified marker on MapLibre layer. |
| `"UNCERTAIN"` | Dead-reckoning degraded | Render yellow/warning pin with expanded uncertainty circle. |
| `"UNAVAILABLE"` | `null` | Hide map marker; display `"No GPS data for this swath"` badge. |

---

### 3.3 Operator Presentation Breakdown (`modality_breakdown` & `indicators`)

Person 3 can directly render operator triage cards:

```
┌────────────────────────────────────────────────────────┐
│  MINE-LIKE CONTACT                           HIGH (87%)│
├────────────────────────────────────────────────────────┤
│  Detector Confidence:                         91%      │
│  Segmentation Verifier:                  N/A (BBox)    │
│  Acoustic Evidence:                           89%      │
│  Morphology Priors:                           86%      │
├────────────────────────────────────────────────────────┤
│  Acoustic Indicators:                                  │
│  • Highlight Contrast: 3.58x relative to ambient      │
│  • Acoustic Shadow:    PRESENT (50 intensity deficit) │
│  • Texture Entropy:    5.60 bits                       │
│  • Edge Density:       12.0%                           │
├────────────────────────────────────────────────────────┤
│  Morphological Indicators:                             │
│  • Aspect Ratio:       1.00 (Square/Cylindrical)       │
│  • Compactness:        0.7854 (High)                   │
│  • Orientation:        10.0° relative to track         │
└────────────────────────────────────────────────────────┘
```

#### Modality Breakdown Fields:
- `modality_breakdown.detector`: Stage-1 detector confidence in $[0.0, 1.0]$.
- `modality_breakdown.segmentation`: SW-Net segmentation confidence in $[0.0, 1.0]$ for Shipwrecks, or `null` when unavailable.
- `modality_breakdown.acoustic`: Composite acoustic highlight/shadow/texture sub-score in $[0.0, 1.0]$.
- `modality_breakdown.morphology`: Composite geometric physical prior sub-score in $[0.0, 1.0]$.

---

## 4. Visual Assets & Cropped Patches (`assets`)

- `assets.roi_bounds`: Local bounding box in parent coordinates.
- `assets.crop_offset`: Top-left $(x, y)$ of cropped context patch.
- `assets.roi_ref`: Relative URL/path to cropped candidate image (e.g. `/api/assets/roi/CNT-000001.png`).
- `assets.mask_ref`: Relative URL/path to binary segmentation mask (or `null` when segmentation is unavailable).

---

## 5. Model Provenance & Versioning (`model`)

- `model.detector_name`: `"DRISHTI-YOLOv8s"`
- `model.detector_version`: `"baseline-v1"`
- `model.swnet_version`: `"swnet-direction-aware-v1.0"` (or `null`)
- `model.fusion_version`: `"baseline_adaptive_linear_v1"`
- `model.calibration_version`: `"platt-mvp-v1.0"`

---

## 6. Unavailable & Error States

| Error Condition | Pipeline Behavior | Person-3 UI Representation |
| :--- | :--- | :--- |
| **No GPS / Navigation** | `location.status = "UNAVAILABLE"`, `latitude = null` | Displays `"No Telemetry"` chip. Pin is not drawn. |
| **Non-Shipwreck Class** | `segmentation.available = false`, `modality_breakdown.segmentation = null` | Displays `"N/A (BBox Geometry)"` badge for segmentation gauge. |
| **Flat / Shadow-less Target** | `acoustic.shadow_status = "ABSENT"`, `shadow_deficit = 0.0` | Displays `"Shadow: ABSENT (Low Profile Target)"`. |
| **Swath Margin Truncation** | `status.is_touching_boundary = true` | Displays `"Border Warning: Candidate truncated at swath edge"`. |
| **Low Contrast Region** | `status.is_low_contrast = true` | Displays `"Low SNR Warning: Weak acoustic return"`. |
| **Degenerate Bounding Box** | `status.overall = "INVALID"`, `final_confidence = 0.0` | Filtered from primary triage table or shown as rejected. |
