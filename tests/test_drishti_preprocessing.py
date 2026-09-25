"""
Tests for DRISHTI Preprocessing Pipeline and Lee Speckle Filtering.
"""

import pytest
import numpy as np
import cv2

from ml.preprocessing.filters import apply_lee_filter
from ml.preprocessing.drishti_preprocess import drishti_preprocess, PREPROCESSING_VERSION


class TestDrishtiPreprocessing:
    @pytest.fixture
    def sample_sonar_patch(self):
        """Generates a synthetic 640x640 sonar patch with speckle noise and an anomaly."""
        np.random.seed(42)
        # Background acoustic floor
        patch = np.random.normal(loc=100, scale=25, size=(640, 640)).clip(0, 255).astype(np.uint8)
        # Add high backscatter highlight target
        patch[300:330, 250:350] = 240
        # Add down-range shadow void
        patch[330:380, 250:350] = 15
        return patch

    def test_lee_filter_preserves_dimensions_and_dtype(self, sample_sonar_patch):
        filtered = apply_lee_filter(sample_sonar_patch, window_size=5, noise_var=0.04)
        assert filtered.shape == sample_sonar_patch.shape
        assert filtered.dtype == np.uint8
        assert not np.isnan(filtered).any()
        assert not np.isinf(filtered).any()

    def test_lee_filter_noise_suppression(self, sample_sonar_patch):
        # Homogeneous region
        orig_roi = sample_sonar_patch[50:150, 50:150]
        filtered = apply_lee_filter(sample_sonar_patch, window_size=5, noise_var=0.04)
        filtered_roi = filtered[50:150, 50:150]
        
        # Lee filter should reduce local variance in homogeneous noise region
        assert np.var(filtered_roi) < np.var(orig_roi)

    def test_lee_filter_invalid_window_size(self, sample_sonar_patch):
        with pytest.raises(ValueError):
            apply_lee_filter(sample_sonar_patch, window_size=4)  # Even window size invalid
        with pytest.raises(ValueError):
            apply_lee_filter(sample_sonar_patch, window_size=1)  # Window size < 3 invalid

    def test_drishti_preprocess_immutability(self, sample_sonar_patch):
        original_copy = sample_sonar_patch.copy()
        processed, meta = drishti_preprocess(sample_sonar_patch)
        
        # Verify original array was not mutated in place
        assert np.array_equal(sample_sonar_patch, original_copy)
        # Verify 3-channel BGR format output for YOLOv8s
        assert len(processed.shape) == 3
        assert processed.shape[2] == 3
        assert processed.shape[:2] == (640, 640)
        assert processed.dtype == np.uint8

    def test_drishti_preprocess_deterministic(self, sample_sonar_patch):
        out1, meta1 = drishti_preprocess(sample_sonar_patch)
        out2, meta2 = drishti_preprocess(sample_sonar_patch)
        assert np.array_equal(out1, out2)
        assert meta1["preprocessing_version"] == PREPROCESSING_VERSION
        assert meta2["preprocessing_version"] == PREPROCESSING_VERSION

    def test_drishti_preprocess_handles_empty_image(self):
        with pytest.raises(ValueError):
            drishti_preprocess(np.array([], dtype=np.uint8))
