"""
Sonar Waterfall Tiling and Coordinate Mapping.

Side-scan sonar waterfall images are frequently thousands of pixels long
along-track while 1000-2000 pixels across-track. This module extracts
overlapping sliding-window tiles for YOLOv8n (default 640x640) and translates
candidate bounding boxes back to global image coordinates.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def generate_tiles(
    image: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.20
) -> List[Dict[str, Any]]:
    """
    Slices an image into overlapping tiles of size tile_size x tile_size.
    
    Returns a list of tile dicts:
        [
            {
                "tile_id": int,
                "tile_image": np.ndarray,
                "offset_x": int,
                "offset_y": int,
                "width": int,
                "height": int
            }, ...
        ]
    """
    h, w = image.shape[:2]
    step = int(tile_size * (1.0 - overlap))
    tiles = []
    tile_id = 0

    y = 0
    while y < h:
        y_end = min(y + tile_size, h)
        y_start = max(0, y_end - tile_size)

        x = 0
        while x < w:
            x_end = min(x + tile_size, w)
            x_start = max(0, x_end - tile_size)

            tile_crop = image[y_start:y_end, x_start:x_end]

            tiles.append({
                "tile_id": tile_id,
                "tile_image": tile_crop,
                "offset_x": x_start,
                "offset_y": y_start,
                "width": x_end - x_start,
                "height": y_end - y_start
            })
            tile_id += 1

            if x_end >= w:
                break
            x += step

        if y_end >= h:
            break
        y += step

    return tiles


def map_tile_bbox_to_global(
    bbox: Tuple[int, int, int, int],
    offset_x: int,
    offset_y: int
) -> Tuple[int, int, int, int]:
    """
    Maps tile-local (x1, y1, x2, y2) back to parent survey image coordinates.
    """
    x1, y1, x2, y2 = bbox
    return (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
