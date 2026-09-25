"""
08_mask_to_yolo.py: Converts Tiled SSS Binary Masks to Ultralytics YOLOv8 Annotations.

STRICT CONSTRAINTS:
- DO NOT modify raw dataset in data/raw/
- Single class: 0 = artificial_anomaly
- Strategy: Hybrid Local Proximity Grouping (MERGE_DISTANCE_PX = 20) with Tiny-Component Filtering
- Generates 640x640 image links and normalized YOLO .txt label files
- All 8,356 tiles (positive and negative) receive verified YOLO representations

Outputs:
- data/interim/yolo/images/
- data/interim/yolo/labels/
- outputs/yolo_conversion_report.json
- outputs/yolo_conversion_report.csv
- outputs/yolo_verification/ (20+ visual verification panels)
"""

import os
import csv
import json
import shutil
from typing import Dict, Any, List, Tuple, Set
import cv2
import numpy as np


# ==============================================================================
# CONFIGURABLE CONVERSION PARAMETERS
# ==============================================================================
YOLO_CONFIG = {
    "class_id": 0,
    "class_name": "artificial_anomaly",
    "tile_size": 640,
    "merge_distance_px": 20,    # Empirical image-space boundary-to-boundary gap threshold
    "min_width_px": 10,         # Minimum width for an isolated component
    "min_height_px": 10,        # Minimum height for an isolated component
    "min_area_px": 50,          # Minimum pixel area for an isolated component
    "max_runaway_span_px": 550, # Guard against excessive transitive chaining across tiles
    "rationale": (
        "Hybrid Local Proximity Grouping bridges internal acoustic shadow voids and deck bulkheads "
        "within 20px gap, absorbing nearby micro-debris into parent hulls while filtering isolated "
        "speckle noise. Negative control tiles receive valid 0-byte .txt label files."
    )
}


def compute_bbox_distance(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    """
    Computes minimum boundary-to-boundary Euclidean distance between two bounding boxes.
    b = (x, y, w, h)
    """
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    dx = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
    dy = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
    return float(np.sqrt(dx * dx + dy * dy))


def group_and_filter_components(
    components: List[Dict[str, Any]],
    merge_dist: int = 20,
    min_w: int = 10,
    min_h: int = 10,
    min_area: int = 50,
    max_span: int = 550
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Groups components within merge_dist boundary gap and filters isolated tiny speckles.
    Returns: (final_detection_boxes, filtered_components_count, merged_components_count)
    """
    n = len(components)
    if n == 0:
        return [], 0, 0

    # Build adjacency matrix based on boundary-to-boundary distance
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        b1 = (components[i]["x"], components[i]["y"], components[i]["w"], components[i]["h"])
        for j in range(i + 1, n):
            b2 = (components[j]["x"], components[j]["y"], components[j]["w"], components[j]["h"])
            dist = compute_bbox_distance(b1, b2)
            if dist <= merge_dist:
                adj[i].append(j)
                adj[j].append(i)

    # Connected component clustering on the proximity graph
    visited = [False] * n
    clusters: List[List[int]] = []

    for i in range(n):
        if not visited[i]:
            cluster = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            clusters.append(cluster)

    final_boxes: List[Dict[str, Any]] = []
    filtered_count = 0
    merged_components_count = 0

    for cluster_indices in clusters:
        cluster_comps = [components[idx] for idx in cluster_indices]

        # Check if the cluster is an isolated tiny component
        if len(cluster_comps) == 1:
            comp = cluster_comps[0]
            if comp["w"] < min_w and comp["h"] < min_h and comp["area"] < min_area:
                # Genuinely isolated tiny component -> filter as speckle noise
                filtered_count += 1
                continue

        # Compute union bounding box
        x_min = min(c["x"] for c in cluster_comps)
        y_min = min(c["y"] for c in cluster_comps)
        x_max = max(c["x"] + c["w"] for c in cluster_comps)
        y_max = max(c["y"] + c["h"] for c in cluster_comps)

        bw = x_max - x_min
        bh = y_max - y_min

        # Guard against runaway transitive chaining across entire tile
        if bw > max_span and bh > max_span and len(cluster_comps) > 3:
            # Revert to individual components for this anomalous runaway cluster
            for c in cluster_comps:
                if c["w"] >= min_w or c["h"] >= min_h or c["area"] >= min_area:
                    final_boxes.append({
                        "x_min": c["x"],
                        "y_min": c["y"],
                        "x_max": c["x"] + c["w"],
                        "y_max": c["y"] + c["h"],
                        "bw": c["w"],
                        "bh": c["h"],
                        "member_count": 1,
                        "pixel_area": c["area"],
                        "is_merged": False
                    })
                else:
                    filtered_count += 1
            continue

        if len(cluster_comps) > 1:
            merged_components_count += len(cluster_comps)

        total_px = sum(c["area"] for c in cluster_comps)
        final_boxes.append({
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "bw": bw,
            "bh": bh,
            "member_count": len(cluster_comps),
            "pixel_area": total_px,
            "is_merged": bool(len(cluster_comps) > 1)
        })

    return final_boxes, filtered_count, merged_components_count


def link_or_copy(src: str, dst: str):
    """Creates a hardlink on Windows NTFS if possible, otherwise copies."""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copyfile(src, dst)


def run_yolo_conversion():
    print("==================================================")
    print("SONAR-INTEL: Mask to YOLO Detection Conversion")
    print(f"Merge Distance: {YOLO_CONFIG['merge_distance_px']} px | Min Area: {YOLO_CONFIG['min_area_px']} px")
    print("==================================================")

    csv_in = "outputs/tiling_report.csv"
    if not os.path.exists(csv_in):
        raise FileNotFoundError("outputs/tiling_report.csv not found!")

    with open(csv_in, "r", encoding="utf-8") as f:
        tile_records = list(csv.DictReader(f))

    total_tiles = len(tile_records)
    print(f"[08_mask_to_yolo] Processing all {total_tiles} tiled samples...")

    # Output directories
    yolo_dir = "data/interim/yolo"
    img_out_dir = os.path.join(yolo_dir, "images")
    lbl_out_dir = os.path.join(yolo_dir, "labels")
    verif_out_dir = "outputs/yolo_verification"

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)
    os.makedirs(verif_out_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    conversion_records: List[Dict[str, Any]] = []
    all_final_boxes: List[Dict[str, Any]] = []

    total_raw_components = 0
    total_filtered_components = 0
    total_merged_components = 0
    total_final_boxes = 0
    total_boundary_boxes = 0

    # Suspicious anomaly counters
    anom_over_90pct = 0
    anom_ultra_thin = 0
    anom_under_10px = 0

    # Verification candidates selector
    verification_samples = []

    for idx, r in enumerate(tile_records):
        tile_id = r["tile_id"]
        fg_pixels = int(r["foreground_pixels"])
        pad_r = int(r["padding_right"])
        pad_b = int(r["padding_bottom"])
        valid_w = YOLO_CONFIG["tile_size"] - pad_r
        valid_h = YOLO_CONFIG["tile_size"] - pad_b

        src_img_p = os.path.join("data/interim/tiled/images", f"{tile_id}.png")
        dst_img_p = os.path.join(img_out_dir, f"{tile_id}.png")
        dst_lbl_p = os.path.join(lbl_out_dir, f"{tile_id}.txt")

        # Step 1: Link image into YOLO images dir
        link_or_copy(src_img_p, dst_img_p)

        # Step 2: Process annotations
        yolo_lines: List[str] = []
        raw_comps_in_tile = 0
        filtered_in_tile = 0
        merged_in_tile = 0
        boxes_in_tile = 0
        boundary_boxes_in_tile = 0

        mask_p = os.path.join("data/interim/tiled/masks", f"{tile_id}.png")

        if fg_pixels > 0 and os.path.exists(mask_p):
            mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)
            if mask is not None:
                # 8-connectivity connected components
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    (mask == 1).astype(np.uint8), connectivity=8
                )

                extracted_comps = []
                for i in range(1, num_labels):
                    x, y, w, h, area = stats[i]
                    extracted_comps.append({
                        "x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)
                    })

                raw_comps_in_tile = len(extracted_comps)
                total_raw_components += raw_comps_in_tile

                # Group and filter
                final_boxes, filtered_in_tile, merged_in_tile = group_and_filter_components(
                    extracted_comps,
                    merge_dist=YOLO_CONFIG["merge_distance_px"],
                    min_w=YOLO_CONFIG["min_width_px"],
                    min_h=YOLO_CONFIG["min_height_px"],
                    min_area=YOLO_CONFIG["min_area_px"],
                    max_span=YOLO_CONFIG["max_runaway_span_px"]
                )

                boxes_in_tile = len(final_boxes)
                total_final_boxes += boxes_in_tile
                total_filtered_components += filtered_in_tile
                total_merged_components += merged_in_tile

                for b in final_boxes:
                    bx1, by1, bx2, by2 = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
                    bw, bh = b["bw"], b["bh"]

                    # Boundary touch checks
                    t_left = bool(bx1 == 0)
                    t_right = bool(bx2 >= valid_w)
                    t_top = bool(by1 == 0)
                    t_bottom = bool(by2 >= valid_h)
                    is_boundary = bool(t_left or t_right or t_top or t_bottom)
                    if is_boundary:
                        boundary_boxes_in_tile += 1
                        total_boundary_boxes += 1

                    # Normalized YOLO coordinates
                    x_center = (bx1 + bx2) / (2.0 * YOLO_CONFIG["tile_size"])
                    y_center = (by1 + by2) / (2.0 * YOLO_CONFIG["tile_size"])
                    w_norm = bw / float(YOLO_CONFIG["tile_size"])
                    h_norm = bh / float(YOLO_CONFIG["tile_size"])

                    # Clamp strictly to [0, 1]
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    w_norm = max(1e-6, min(1.0, w_norm))
                    h_norm = max(1e-6, min(1.0, h_norm))

                    # Suspicious anomaly detection
                    if (w_norm * h_norm) > 0.90:
                        anom_over_90pct += 1
                    asp = bw / max(1, bh)
                    if asp > 15.0 or asp < (1.0 / 15.0):
                        anom_ultra_thin += 1
                    if bw < 10 or bh < 10:
                        anom_under_10px += 1

                    line = f"{YOLO_CONFIG['class_id']} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                    yolo_lines.append(line)

                    all_final_boxes.append({
                        "tile_id": tile_id,
                        "bw_px": bw,
                        "bh_px": bh,
                        "area_px": bw * bh,
                        "is_boundary": is_boundary,
                        "member_count": b["member_count"]
                    })

                # Select representative tiles for visual verification
                if len(verification_samples) < 24:
                    if raw_comps_in_tile == 1 and boxes_in_tile == 1 and bw > 150:
                        verification_samples.append((tile_id, "Single Coherent Shipwreck Target", final_boxes, False))
                    elif raw_comps_in_tile >= 3 and boxes_in_tile < raw_comps_in_tile:
                        verification_samples.append((tile_id, f"Proximity-Merged Wreck ({raw_comps_in_tile} comps -> {boxes_in_tile} boxes)", final_boxes, True))
                    elif raw_comps_in_tile >= 2 and boxes_in_tile == raw_comps_in_tile:
                        verification_samples.append((tile_id, f"Distant Separate Targets ({boxes_in_tile} boxes unmerged)", final_boxes, False))
                    elif filtered_in_tile > 0 and boxes_in_tile > 0:
                        verification_samples.append((tile_id, f"Tiny Component Filtered ({filtered_in_tile} speckle removed)", final_boxes, True))
                    elif boundary_boxes_in_tile > 0 and bw > 100:
                        verification_samples.append((tile_id, "Boundary-Crossing Shipwreck Target", final_boxes, False))
                    elif bw > 400 or bh > 400:
                        verification_samples.append((tile_id, "Large Target Footprint (>400px span)", final_boxes, False))
                    elif "Corsair" in tile_id and boxes_in_tile >= 1:
                        verification_samples.append((tile_id, "Corsair Aircraft Wreckage Detection", final_boxes, True))
                    elif "Lucinda" in tile_id and raw_comps_in_tile >= 5:
                        verification_samples.append((tile_id, "Dense Debris Field (Lucinda Site)", final_boxes, True))

        # Select negative verification sample
        if fg_pixels == 0 and len([v for v in verification_samples if "Negative" in v[1]]) < 2:
            verification_samples.append((tile_id, "Negative Seabed Control Tile (0 Detections)", [], False))

        # Step 3: Write YOLO label file (.txt)
        with open(dst_lbl_p, "w", encoding="utf-8") as lf:
            if len(yolo_lines) > 0:
                lf.write("\n".join(yolo_lines) + "\n")
            else:
                pass # Empty 0-byte file for negative image

        rec = {
            "tile_id": tile_id,
            "source_image": r["source_image"],
            "source_site": r["source_site"],
            "source_split": r["source_split"],
            "number_of_components": raw_comps_in_tile,
            "number_of_filtered_components": filtered_in_tile,
            "number_of_groups": boxes_in_tile,
            "number_of_final_boxes": boxes_in_tile,
            "boundary_box_count": boundary_boxes_in_tile
        }
        conversion_records.append(rec)

    # -----------------------------------------------------------------
    # Save CSV Report
    # -----------------------------------------------------------------
    csv_out_path = "outputs/yolo_conversion_report.csv"
    with open(csv_out_path, "w", newline="", encoding="utf-8") as cf:
        fieldnames = [
            "tile_id", "source_image", "source_site", "source_split",
            "number_of_components", "number_of_filtered_components",
            "number_of_groups", "number_of_final_boxes", "boundary_box_count"
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for cr in conversion_records:
            writer.writerow(cr)
    print(f"[08_mask_to_yolo] Conversion CSV written to {csv_out_path}")

    # -----------------------------------------------------------------
    # Run Automated Validations
    # -----------------------------------------------------------------
    print("[08_mask_to_yolo] Executing 12 automated integrity assertions...")
    yolo_imgs = os.listdir(img_out_dir)
    yolo_lbls = os.listdir(lbl_out_dir)

    assert len(yolo_imgs) == total_tiles, f"Assertion Failed: Expected {total_tiles} images, got {len(yolo_imgs)}"
    assert len(yolo_lbls) == total_tiles, f"Assertion Failed: Expected {total_tiles} labels, got {len(yolo_lbls)}"
    assert len(set(yolo_imgs)) == total_tiles, "Assertion Failed: Duplicate image IDs found!"

    # Validate label contents
    validated_boxes = 0
    empty_label_count = 0
    for lbl_fn in yolo_lbls:
        lp = os.path.join(lbl_out_dir, lbl_fn)
        with open(lp, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        if len(lines) == 0:
            empty_label_count += 1
            continue

        for ln in lines:
            parts = ln.split()
            assert len(parts) == 5, f"Label {lbl_fn} does not have 5 fields: {ln}"
            cls_id = int(parts[0])
            xc, yc, wn, hn = map(float, parts[1:])
            assert cls_id == 0, f"Invalid class ID {cls_id} in {lbl_fn}"
            assert 0.0 <= xc <= 1.0, f"Invalid x_center {xc} in {lbl_fn}"
            assert 0.0 <= yc <= 1.0, f"Invalid y_center {yc} in {lbl_fn}"
            assert 0.0 < wn <= 1.0, f"Invalid width {wn} in {lbl_fn}"
            assert 0.0 < hn <= 1.0, f"Invalid height {hn} in {lbl_fn}"
            assert (xc - wn / 2.0) >= -1e-4, f"Box exceeds left boundary in {lbl_fn}"
            assert (yc - hn / 2.0) >= -1e-4, f"Box exceeds top boundary in {lbl_fn}"
            validated_boxes += 1

    assert validated_boxes == total_final_boxes, "Box count mismatch in validation!"
    print("  -> All 12 automated integrity assertions PASSED.")

    # -----------------------------------------------------------------
    # Compute Statistics
    # -----------------------------------------------------------------
    pos_tiles_count = sum(1 for cr in conversion_records if cr["number_of_final_boxes"] > 0)
    neg_tiles_count = sum(1 for cr in conversion_records if cr["number_of_final_boxes"] == 0)

    final_widths = np.array([b["bw_px"] for b in all_final_boxes])
    final_heights = np.array([b["bh_px"] for b in all_final_boxes])
    final_areas = np.array([b["area_px"] for b in all_final_boxes])

    pct_keys = [10, 25, 50, 75, 90, 95]
    fw_pcts = {f"P{k}": round(float(np.percentile(final_widths, k)), 1) for k in pct_keys} if len(final_widths) > 0 else {}
    fh_pcts = {f"P{k}": round(float(np.percentile(final_heights, k)), 1) for k in pct_keys} if len(final_heights) > 0 else {}
    fa_pcts = {f"P{k}": round(float(np.percentile(final_areas, k)), 1) for k in pct_keys} if len(final_areas) > 0 else {}

    json_report = {
        "dataset_name": "AI4Shipwrecks (YOLOv8 Detection)",
        "configuration": YOLO_CONFIG,
        "summary": {
            "total_tiles": total_tiles,
            "positive_tiles": pos_tiles_count,
            "negative_tiles": neg_tiles_count,
            "total_raw_components": total_raw_components,
            "filtered_components": total_filtered_components,
            "filtered_percentage": round((total_filtered_components / max(1, total_raw_components)) * 100.0, 2),
            "merged_components": total_merged_components,
            "merged_percentage": round((total_merged_components / max(1, total_raw_components)) * 100.0, 2),
            "final_boxes": total_final_boxes,
            "boxes_per_positive_tile": round(total_final_boxes / max(1, pos_tiles_count), 2),
            "boundary_boxes": total_boundary_boxes,
            "boundary_box_percentage": round((total_boundary_boxes / max(1, total_final_boxes)) * 100.0, 2),
            "images_written": len(yolo_imgs),
            "labels_written": len(yolo_lbls),
            "empty_negative_labels": empty_label_count
        },
        "box_dimension_percentiles": {
            "width_px": fw_pcts,
            "height_px": fh_pcts,
            "area_px": fa_pcts
        },
        "critical_validation_flags": {
            "boxes_over_90pct_tile_area": anom_over_90pct,
            "boxes_ultra_thin_aspect_ratio": anom_ultra_thin,
            "boxes_under_10px_dimension": anom_under_10px
        },
        "validation_status": "PASS"
    }

    json_out_path = "outputs/yolo_conversion_report.json"
    with open(json_out_path, "w", encoding="utf-8") as jf:
        json.dump(json_report, jf, indent=2)
    print(f"[08_mask_to_yolo] Summary JSON written to {json_out_path}")

    # -----------------------------------------------------------------
    # Generate Visual Verification Panels (At least 20 panels)
    # -----------------------------------------------------------------
    generate_verification_visuals(verification_samples[:22], verif_out_dir)

    # -----------------------------------------------------------------
    # Final Terminal Printout
    # -----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("YOLO CONVERSION REPORT")
    print("----------------------\n")
    print(f"Total tiles:       {total_tiles}")
    print(f"Positive tiles:    {pos_tiles_count}")
    print(f"Negative tiles:    {neg_tiles_count}\n")
    print(f"Raw components:    {total_raw_components}")
    print(f"Filtered:          {total_filtered_components} ({json_report['summary']['filtered_percentage']}%)")
    print(f"Merged:            {total_merged_components} ({json_report['summary']['merged_percentage']}%)")
    print(f"Final boxes:       {total_final_boxes}\n")
    print(f"Boundary boxes:    {total_boundary_boxes} ({json_report['summary']['boundary_box_percentage']}%)\n")
    print(f"Images written:    {len(yolo_imgs)}")
    print(f"Labels written:    {len(yolo_lbls)}\n")
    print("Validation:")
    print("PASS")
    print("=" * 50 + "\n")


def generate_verification_visuals(
    samples: List[Tuple[str, str, List[Dict[str, Any]], bool]],
    output_dir: str
):
    """Generates 2-panel and 3-panel visual verification images."""
    print(f"[08_mask_to_yolo] Generating {len(samples)} visual verification panels in {output_dir}...")

    for idx, (tile_id, desc, boxes, show_3_panel) in enumerate(samples):
        img_p = os.path.join("data/interim/tiled/images", f"{tile_id}.png")
        mask_p = os.path.join("data/interim/tiled/masks", f"{tile_id}.png")

        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED) if os.path.exists(mask_p) else None

        if img is None:
            continue

        p_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        p_det = p_img.copy()

        # Draw YOLO bounding boxes on right panel
        for b in boxes:
            x1, y1, x2, y2 = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
            cv2.rectangle(p_det, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(p_det, f"anomaly {b['bw']}x{b['bh']}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        if len(boxes) == 0:
            cv2.putText(p_det, "0 DETECTIONS (NEGATIVE TILE)", (140, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 180, 240), 2, cv2.LINE_AA)

        div = np.ones((640, 5, 3), dtype=np.uint8) * 160

        if show_3_panel and mask is not None:
            # 3-panel: [MASK | SSS IMAGE | SSS + YOLO BOXES]
            p_mask = np.zeros((640, 640, 3), dtype=np.uint8)
            p_mask[mask == 1] = (0, 230, 255) # Yellow mask
            body = np.hstack([p_mask, div, p_img, div, p_det])
            hdr_text = f"[LEFT: Ground-Truth Mask | MID: 640x640 SSS | RIGHT: YOLO Detections ({len(boxes)} boxes)]"
        else:
            # 2-panel: [SSS IMAGE | SSS + YOLO BOXES]
            body = np.hstack([p_img, div, p_det])
            hdr_text = f"[LEFT: 640x640 SSS Image | RIGHT: YOLO Detections ({len(boxes)} boxes)]"

        hdr = np.zeros((44, body.shape[1], 3), dtype=np.uint8)
        hdr[:] = 25

        cv2.putText(hdr, f"VERIF {idx+1:02d}: {tile_id} | {desc}", (15, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(hdr, hdr_text, (body.shape[1] - 620, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 210, 240), 1, cv2.LINE_AA)

        panel = np.vstack([hdr, body])
        out_fn = f"yolo_verif_{idx+1:02d}_{tile_id}.png"
        cv2.imwrite(os.path.join(output_dir, out_fn), panel)


if __name__ == "__main__":
    run_yolo_conversion()
