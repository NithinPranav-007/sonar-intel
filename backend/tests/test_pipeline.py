"""
Tests for ML Sonar Normalization, Quality Checks, and Acoustic Context Pipeline.
"""

import numpy as np
import pytest
from ml.preprocessing.normalize import normalize_sonar_intensity, apply_clahe, handle_water_column
from ml.preprocessing.quality import compute_image_quality
from ml.preprocessing.tiling import generate_tiles, map_tile_bbox_to_global
from ml.inference.context import extract_acoustic_context
from ml.inference.scoring import PriorityScorer


def test_normalization_and_clahe():
    # Synthetic 8-bit image with dynamic range
    img = np.random.randint(40, 200, (800, 600), dtype=np.uint8)
    norm = normalize_sonar_intensity(img)
    assert norm.shape == (800, 600)
    assert norm.dtype == np.uint8

    enhanced = apply_clahe(norm)
    assert enhanced.shape == (800, 600)


def test_water_column_handling():
    img = np.ones((500, 800), dtype=np.uint8) * 100
    processed, nadir_bounds = handle_water_column(img, nadir_width_ratio=0.1, blank_nadir=True)
    start_x, end_x = nadir_bounds
    assert start_x < end_x
    assert processed[:, start_x:end_x].sum() == 0


def test_quality_metrics():
    # Good quality image with variance
    img = np.random.normal(128, 30, (400, 400)).astype(np.uint8)
    metrics = compute_image_quality(img)
    assert "quality_score" in metrics
    assert 0.0 <= metrics["quality_score"] <= 1.0
    assert metrics["is_usable"] is True

    # Degraded blank image
    blank = np.zeros((100, 100), dtype=np.uint8)
    blank_metrics = compute_image_quality(blank)
    assert blank_metrics["quality_score"] < 0.25


def test_tiling_and_bbox_remapping():
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    tiles = generate_tiles(img, tile_size=640, overlap=0.2)
    assert len(tiles) >= 4

    tile0 = tiles[0]
    global_bbox = map_tile_bbox_to_global((10, 20, 50, 60), tile0["offset_x"], tile0["offset_y"])
    assert global_bbox[0] >= 10
    assert global_bbox[1] >= 20


def test_acoustic_context_and_scoring():
    img = np.ones((800, 800), dtype=np.uint8) * 100
    # Add a highlight and shadow
    img[100:150, 400:450] = 240  # Highlight
    img[100:150, 350:400] = 20   # Shadow (left of target, assuming nadir > 450)

    context = extract_acoustic_context(img, {"x1": 400, "y1": 100, "x2": 450, "y2": 150}, nadir_x=600)
    assert "shadow_evidence" in context
    assert "context_score" in context
    assert 0.0 <= context["context_score"] <= 1.0

    scorer = PriorityScorer()
    priority, score = scorer.calculate_priority(
        model_confidence=0.90,
        context_score=context["context_score"],
        data_quality=0.95,
        localization_status="ESTIMATED"
    )
    assert priority in ["HIGH", "MEDIUM", "LOW"]
    assert 0.0 <= score <= 1.0
