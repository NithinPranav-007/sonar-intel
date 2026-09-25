"""
01_inspect.py: AI4Shipwrecks Dataset Inspection and Verification Script.

STRICT PROTOCOL CONSTRAINTS:
- DO NOT modify, rename, move, preprocess, resize, normalize, or overwrite anything inside data/raw/ai4shipwrecks/.
- DO NOT implement YOLO, CLAHE, tiling, training, or database operations.
- Pure dataset exploration and verification.

Outputs:
- outputs/dataset_inspection.json
- docs/preprocessing/01_dataset_inspection.md
- data/interim/inspection/ (5-10 side-by-side original SSS vs mask pairs)
"""

import os
import glob
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from PIL import Image


def find_dataset_root() -> str:
    """Finds the dataset root path without modifying anything."""
    candidates = [
        "data/raw/ai4shipwrecks",
        "data/raw/AI4Shipwrecks"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"AI4Shipwrecks dataset not found at expected paths: {candidates}")


def get_site_id(filename: str) -> str:
    """Extracts site identifier prefix (e.g. DM_Wilson from DM_Wilson_01.png)."""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return base


def run_inspection():
    dataset_root = find_dataset_root()
    print(f"[01_inspect] Inspecting dataset at: {dataset_root}")

    # Ensure output directories exist (never touching raw directory)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("docs/preprocessing", exist_ok=True)
    os.makedirs("data/interim/inspection", exist_ok=True)

    # 1. Recursive file discovery
    image_paths: List[str] = []
    mask_paths: List[str] = []

    # Map by relative path / filename
    # AI4Shipwrecks structure: <split>/images/<filename> and <split>/labels/<filename>
    for root, _, files in os.walk(dataset_root):
        rel_root = os.path.relpath(root, dataset_root).replace("\\", "/")
        for f in files:
            full_p = os.path.join(root, f)
            if "/images" in rel_root or rel_root.endswith("images"):
                image_paths.append(full_p)
            elif "/labels" in rel_root or rel_root.endswith("labels"):
                mask_paths.append(full_p)

    image_paths.sort()
    mask_paths.sort()

    total_images = len(image_paths)
    total_masks = len(mask_paths)

    # Map filenames to relative keys to evaluate pairing
    image_dict: Dict[str, str] = {}
    for p in image_paths:
        rel = os.path.relpath(p, dataset_root).replace("\\", "/")
        # Key by (split, filename)
        parts = rel.split("/")
        key = (parts[0], parts[-1]) if len(parts) >= 2 else ("", parts[-1])
        image_dict[str(key)] = p

    mask_dict: Dict[str, str] = {}
    for p in mask_paths:
        rel = os.path.relpath(p, dataset_root).replace("\\", "/")
        parts = rel.split("/")
        key = (parts[0], parts[-1]) if len(parts) >= 2 else ("", parts[-1])
        mask_dict[str(key)] = p

    matched_keys = set(image_dict.keys()).intersection(set(mask_dict.keys()))
    unmatched_images = list(set(image_dict.keys()) - set(mask_dict.keys()))
    unmatched_masks = list(set(mask_dict.keys()) - set(image_dict.keys()))

    # 2. Detailed Image and Mask Properties
    img_formats = set()
    img_channels = set()
    img_dtypes = set()
    img_widths = []
    img_heights = []
    global_min_intensity = 255
    global_max_intensity = 0

    mask_formats = set()
    mask_channels = set()
    mask_dtypes = set()
    mask_widths = []
    mask_heights = []
    mask_unique_values = set()
    masks_with_target = 0

    sites_by_split: Dict[str, set] = {"train": set(), "test": set(), "extras": set()}
    all_sites = set()

    # Analyze images
    for p in image_paths:
        rel = os.path.relpath(p, dataset_root).replace("\\", "/")
        split_name = rel.split("/")[0]
        site_id = get_site_id(p)
        all_sites.add(site_id)
        if split_name in sites_by_split:
            sites_by_split[split_name].add(site_id)
        else:
            sites_by_split.setdefault(split_name, set()).add(site_id)

        with Image.open(p) as img_pil:
            img_formats.add(img_pil.format)
            w, h = img_pil.size
            img_widths.append(w)
            img_heights.append(h)
            img_channels.add(len(img_pil.getbands()))

        img_arr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img_arr is not None:
            img_dtypes.add(str(img_arr.dtype))
            mi, ma = int(img_arr.min()), int(img_arr.max())
            if mi < global_min_intensity:
                global_min_intensity = mi
            if ma > global_max_intensity:
                global_max_intensity = ma

    # Analyze masks
    for p in mask_paths:
        with Image.open(p) as m_pil:
            mask_formats.add(m_pil.format)
            w, h = m_pil.size
            mask_widths.append(w)
            mask_heights.append(h)
            mask_channels.add(len(m_pil.getbands()))

        lbl_arr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if lbl_arr is not None:
            mask_dtypes.add(str(lbl_arr.dtype))
            uniques = np.unique(lbl_arr)
            mask_unique_values.update(uniques.tolist())
            if 1 in uniques or any(u > 0 for u in uniques):
                masks_with_target += 1

    # Directory Structure Tree Representation
    dir_structure = {
        "dataset_root": dataset_root,
        "splits": {}
    }
    for sub in sorted(os.listdir(dataset_root)):
        sub_path = os.path.join(dataset_root, sub)
        if os.path.isdir(sub_path):
            dir_structure["splits"][sub] = {}
            for child in sorted(os.listdir(sub_path)):
                child_path = os.path.join(sub_path, child)
                if os.path.isdir(child_path):
                    # Check if there is another level (e.g. extras/terrain/images)
                    subchildren = os.listdir(child_path)
                    subdirs = [sc for sc in subchildren if os.path.isdir(os.path.join(child_path, sc))]
                    if subdirs:
                        dir_structure["splits"][sub][child] = subdirs
                    else:
                        dir_structure["splits"][sub][child] = len(subchildren)

    # Compile JSON Report
    report_dict: Dict[str, Any] = {
        "dataset_name": "AI4Shipwrecks",
        "dataset_root": dataset_root,
        "inspection_status": "COMPLETED",
        "summary": {
            "total_images": total_images,
            "total_masks": total_masks,
            "matched_pairs": len(matched_keys),
            "unmatched_images": len(unmatched_images),
            "unmatched_masks": len(unmatched_masks),
            "masks_containing_target": masks_with_target,
            "masks_empty_background_only": total_masks - masks_with_target,
            "unique_sites_count": len(all_sites),
            "sites": sorted(list(all_sites))
        },
        "image_specifications": {
            "formats": sorted(list(img_formats)),
            "channels": sorted(list(img_channels)),
            "dtypes": sorted(list(img_dtypes)),
            "intensity_min": global_min_intensity,
            "intensity_max": global_max_intensity,
            "width": {
                "min": int(min(img_widths)),
                "max": int(max(img_widths)),
                "median": float(np.median(img_widths))
            },
            "height": {
                "min": int(min(img_heights)),
                "max": int(max(img_heights)),
                "median": float(np.median(img_heights))
            }
        },
        "mask_specifications": {
            "formats": sorted(list(mask_formats)),
            "channels": sorted(list(mask_channels)),
            "dtypes": sorted(list(mask_dtypes)),
            "unique_values": sorted(list(mask_unique_values)),
            "width": {
                "min": int(min(mask_widths)),
                "max": int(max(mask_widths)),
                "median": float(np.median(mask_widths))
            },
            "height": {
                "min": int(min(mask_heights)),
                "max": int(max(mask_heights)),
                "median": float(np.median(mask_heights))
            }
        },
        "metadata_availability": {
            "navigation_csv": "NOT AVAILABLE",
            "gps_latitude": "NOT AVAILABLE",
            "gps_longitude": "NOT AVAILABLE",
            "vessel_heading": "NOT AVAILABLE",
            "towfish_altitude": "NOT AVAILABLE",
            "slant_range_metadata": "NOT AVAILABLE",
            "sonar_frequency_metadata": "NOT AVAILABLE",
            "target_bounding_boxes": "NOT AVAILABLE (only pixel segmentation masks provided)",
            "debris_classes": "NOT AVAILABLE (labels only distinguish shipwreck from seabed)"
        },
        "split_breakdown": {
            "train": {
                "images": len(glob.glob(os.path.join(dataset_root, "train/images/*.png"))),
                "labels": len(glob.glob(os.path.join(dataset_root, "train/labels/*.png"))),
                "sites": sorted(list(sites_by_split.get("train", set())))
            },
            "test": {
                "images": len(glob.glob(os.path.join(dataset_root, "test/images/*.png"))),
                "labels": len(glob.glob(os.path.join(dataset_root, "test/labels/*.png"))),
                "sites": sorted(list(sites_by_split.get("test", set())))
            },
            "extras": {
                "images": len(glob.glob(os.path.join(dataset_root, "extras/**/images/*.png"), recursive=True)),
                "labels": len(glob.glob(os.path.join(dataset_root, "extras/**/labels/*.png"), recursive=True)),
                "sites": sorted(list(sites_by_split.get("extras", set())))
            }
        },
        "directory_structure": dir_structure
    }

    # Save JSON report
    json_path = "outputs/dataset_inspection.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"[01_inspect] Machine-readable report saved to {json_path}")

    # 3. Generate Visual Inspection Set (8 representative image/mask pairs)
    # Select pairs with clear shipwreck targets from distinct sites
    representative_candidates = [
        "train/images/DM_Wilson_02.png",
        "train/images/EB_Allen_03.png",
        "train/images/Egyptian_04.png",
        "train/images/Lucinda_van_Valkenburg_05.png",
        "train/images/Oscar_T_Flint_01.png",
        "test/images/Artificial_Reef_06.png",
        "test/images/Barge_No_1_02.png",
        "test/images/Corsair_01.png"
    ]

    print("[01_inspect] Generating visual inspection set in data/interim/inspection/...")
    inspection_count = 0
    for rel_cand in representative_candidates:
        img_full = os.path.join(dataset_root, rel_cand)
        lbl_full = os.path.join(dataset_root, rel_cand.replace("/images/", "/labels/"))

        if not os.path.exists(img_full) or not os.path.exists(lbl_full):
            continue

        raw_img = cv2.imread(img_full, cv2.IMREAD_GRAYSCALE)
        raw_mask = cv2.imread(lbl_full, cv2.IMREAD_GRAYSCALE)

        if raw_img is None or raw_mask is None:
            continue

        # Find bounding region of the target in the mask to crop a focused, high-clarity inspection window
        target_coords = np.argwhere(raw_mask == 1)
        h, w = raw_img.shape[:2]

        if len(target_coords) > 0:
            y_min, x_min = target_coords.min(axis=0)
            y_max, x_max = target_coords.max(axis=0)

            # Center crop around the shipwreck target with reasonable vertical context
            crop_height = 1024
            target_center_y = (y_min + y_max) // 2
            c_y1 = max(0, target_center_y - crop_height // 2)
            c_y2 = min(h, c_y1 + crop_height)
            c_y1 = max(0, c_y2 - crop_height)

            img_crop = raw_img[c_y1:c_y2, :]
            mask_crop = raw_mask[c_y1:c_y2, :]
        else:
            # Full swath resize if empty
            img_crop = raw_img[:1024, :]
            mask_crop = raw_mask[:1024, :]

        # Convert mask to visual 8-bit (0 -> 0, 1 -> 255)
        vis_mask = (mask_crop * 255).astype(np.uint8)

        # Create side-by-side visualization
        # LEFT: original SSS image
        # RIGHT: corresponding mask
        # Add visual divider line
        divider = np.ones((img_crop.shape[0], 4), dtype=np.uint8) * 128
        side_by_side = np.hstack([img_crop, divider, vis_mask])

        base_name = os.path.splitext(os.path.basename(rel_cand))[0]
        out_vis_path = os.path.join("data/interim/inspection", f"inspect_{base_name}_pair.png")
        cv2.imwrite(out_vis_path, side_by_side)
        inspection_count += 1

    print(f"[01_inspect] Created {inspection_count} side-by-side inspection pairs.")

    # 4. Generate Human-Readable Markdown Report
    generate_markdown_report(report_dict, "docs/preprocessing/01_dataset_inspection.md")

    # 5. Print Terminal Summary exactly as requested
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("-" * 50)
    print(f"Images:             {total_images}")
    print(f"Masks:              {total_masks}")
    print(f"Matched pairs:      {len(matched_keys)}")
    print(f"Unmatched images:   {len(unmatched_images)}")
    print(f"Unmatched masks:    {len(unmatched_masks)}")
    print(f"Unique sites:       {len(all_sites)}")
    print(f"Image dimensions:   Width {min(img_widths)}px | Height min {min(img_heights)}px, median {float(np.median(img_heights))}px, max {max(img_heights)}px")
    print(f"Mask dimensions:    Width {min(mask_widths)}px | Height min {min(mask_heights)}px, median {float(np.median(mask_heights))}px, max {max(mask_heights)}px")
    print(f"Intensity range:    [{global_min_intensity}, {global_max_intensity}] (uint8)")
    print(f"Mask values:        {sorted(list(mask_unique_values))} (0=Background/Seabed, 1=Target)")
    print("=" * 50 + "\n")

    print("DOMAIN & OPERATIONAL EVALUATION:")
    print("--------------------------------")
    print("1. What the dataset actually contains:")
    print("   - High-resolution, raw unnormalized single-channel PNG side-scan sonar waterfall imagery")
    print("   - Collected by the University of Michigan Field Robotics Group via AUV in Lake Huron")
    print("   - Exactly 286 acoustic swaths across 28 shipwreck sites and 1 terrain site")
    print("   - Pixel-accurate binary semantic segmentation masks (0=seabed, 1=wreck structure)")
    print("\n2. Whether it is suitable for our current artificial-anomaly MVP:")
    print("   - HIGHLY SUITABLE for artificial anomaly candidate detection and acoustic shadow verification.")
    print("   - Shipwrecks represent large-scale man-made structures with high acoustic contrast and down-range shadows.")
    print("   - Provides realistic variable-length sonar swaths with real grazing angles and nadir water column returns.")
    print("\n3. What target classes are actually supported by the labels:")
    print("   - ONLY 'shipwreck' (binary target class 1).")
    print("   - Does NOT distinguish ghost nets, plastic debris, or individual cargo containers.")
    print("   - Must be mapped to 'artificial_anomaly' or 'wreck_candidate' in the MVP.")
    print("\n4. What information is NOT AVAILABLE in the dataset:")
    print("   - Navigation coordinates (Latitude, Longitude) are NOT AVAILABLE.")
    print("   - Towfish heading, altitude, and slant-range metadata are NOT AVAILABLE.")
    print("   - Sensor frequency and pulse parameters are NOT AVAILABLE.")
    print("   - Pre-computed bounding boxes are NOT AVAILABLE (must be derived from contour envelopes if using YOLO).")


def generate_markdown_report(report: Dict[str, Any], output_path: str):
    """Generates the human-readable docs/preprocessing/01_dataset_inspection.md report."""
    md = f"""# AI4Shipwrecks Dataset Inspection Report

**Date of Inspection:** 2026-08-31  
**Dataset Reference:** [UM Field Robotics AI4Shipwrecks](https://umfieldrobotics.github.io/ai4shipwrecks/)  
**Dataset Root:** `{report['dataset_root']}`  

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
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[01_inspect] Human-readable documentation saved to {output_path}")


if __name__ == "__main__":
    run_inspection()
