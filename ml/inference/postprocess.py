"""
Contact Post-Processing, Spatial Deduplication, and Candidate Ranking.

Filters raw YOLO tile predictions:
- Eliminates degenerate boundary slivers
- Merges duplicate detections across 20% stride overlaps
- Suppresses sub-box containments
- Ranks candidates by composite acoustic strength
"""

from typing import List, Dict, Any


def compute_iou(box1: Dict[str, int], box2: Dict[str, int]) -> float:
    """Computes Intersection-over-Union between two bounding boxes."""
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    intersection_w = max(0, x2 - x1)
    intersection_h = max(0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0
    return float(intersection_area) / float(union_area)


def compute_containment(box1: Dict[str, int], box2: Dict[str, int]) -> float:
    """Computes intersection over the minimum area of the two boxes."""
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    intersection_w = max(0, x2 - x1)
    intersection_h = max(0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    min_area = min(area1, area2)

    if min_area <= 0:
        return 0.0
    return float(intersection_area) / float(min_area)


def deduplicate_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.35,
    min_box_size: int = 15
) -> List[Dict[str, Any]]:
    """
    Suppresses degenerate boundary slivers and performs Non-Maximum Suppression (NMS)
    plus containment filtering across overlapping tile bounds.
    """
    if not detections:
        return []

    # 1. Filter out degenerate tile boundary slivers
    valid_dets = [
        d for d in detections
        if (d["bbox"]["x2"] - d["bbox"]["x1"] >= min_box_size) and
           (d["bbox"]["y2"] - d["bbox"]["y1"] >= min_box_size)
    ]

    # 2. Sort descending by detector confidence
    sorted_dets = sorted(valid_dets, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        kept.append(best)

        remaining = []
        for other in sorted_dets:
            iou = compute_iou(best["bbox"], other["bbox"])
            containment = compute_containment(best["bbox"], other["bbox"])
            # Suppress if significant IoU overlap or one is contained within the other
            if iou < iou_threshold and containment < 0.65:
                remaining.append(other)
        sorted_dets = remaining

    return kept
