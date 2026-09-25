"""
Sonar Waterfall Tiling and Coordinate Mapping.

Side-scan sonar waterfall images are frequently thousands of pixels long
along-track while 1000-2000 pixels across-track. This module extracts
overlapping sliding-window tiles for YOLOv8n (default 640x640) and translates
candidate bounding boxes back to global image coordinates.
"""

from typing import List, Dict, Any, Tuple, Iterator
import numpy as np


def generate_tiles_iter(
    image: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.20
) -> Iterator[Dict[str, Any]]:
    """
    Memory-safe generator that yields one overlapping tile at a time.
    Avoids allocating a massive list containing all tile image arrays simultaneously in RAM.
    """
    h, w = image.shape[:2]
    step = int(tile_size * (1.0 - overlap))
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

            yield {
                "tile_id": tile_id,
                "tile_image": tile_crop,
                "offset_x": x_start,
                "offset_y": y_start,
                "width": x_end - x_start,
                "height": y_end - y_start
            }
            tile_id += 1

            if x_end >= w:
                break
            x += step

        if y_end >= h:
            break
        y += step


def generate_tiles(
    image: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.20
) -> List[Dict[str, Any]]:
    """
    Backward-compatible list generator.
    """
    return list(generate_tiles_iter(image, tile_size=tile_size, overlap=overlap))


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
