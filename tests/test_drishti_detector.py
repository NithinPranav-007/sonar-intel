import os
import pytest
import numpy as np
import cv2
import torch

from ml.inference.drishti_detector import DrishtiDetector, DrishtiDetection
from backend.app.core.config import settings


class TestDrishtiDetector:
    @pytest.fixture(scope="class")
    def detector(self):
        """Initializes detector instance."""
        return DrishtiDetector(confidence_threshold=0.10)

    def test_model_loads_successfully(self, detector):
        assert detector.model is not None
        assert os.path.exists(detector.model_path)
        assert detector.model_name == "DRISHTI-YOLOv8s"

    def test_expected_class_mapping(self, detector):
        names = detector.class_names
        assert 0 in names and names[0] == "crab_pot"
        assert 1 in names and names[1] == "submarine_pipeline"
        assert 2 in names and names[2] == "shipwreck"
        assert 3 in names and names[3] == "ghost_net"
        assert 4 in names and names[4] == "mine_cylinder"

    def test_model_cached_per_process(self, detector):
        detector_second = DrishtiDetector()
        # Verify underlying YOLO model object identity is shared (cached)
        assert detector.model is detector_second.model

    def test_inference_on_real_sonar_imagery(self, detector):
        img_path = "data/demo/sonar/viator_04_test_wreck.png"
        assert os.path.exists(img_path)
        raw_image = cv2.imread(img_path)
        assert raw_image is not None

        # Predict on a 640x640 crop containing the shipwreck hull
        crop = raw_image[1000:1640, 400:1040]
        detections = detector.predict(crop, tile_id="TEST_VIATOR_CROP")

        assert isinstance(detections, list)
        assert len(detections) > 0

        for d in detections:
            assert isinstance(d, DrishtiDetection)
            assert d.class_name in detector.class_names.values()
            assert 0.0 <= d.confidence <= 1.0
            x1, y1, x2, y2 = d.bbox
            assert 0 <= x1 <= d.image_width
            assert 0 <= x2 <= d.image_width
            assert 0 <= y1 <= d.image_height
            assert 0 <= y2 <= d.image_height
            assert d.tile_id == "TEST_VIATOR_CROP"
            assert d.model_name == "DRISHTI-YOLOv8s"
            assert d.model_version == "baseline-v1"

    def test_empty_detection_handling(self, detector):
        # Homogeneous black image should return empty list without crashing
        black_img = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = detector.predict(black_img)
        assert isinstance(detections, list)
        assert len(detections) == 0

    def test_crab_pot_is_tagged_as_filtered(self, detector):
        # Mock results containing a crab_pot detection
        class MockBox:
            xyxy = [torch.tensor([50.0, 60.0, 150.0, 180.0])]
            conf = [torch.tensor([0.88])]
            cls = [torch.tensor([0])]  # Class 0: crab_pot

        class MockResult:
            boxes = [MockBox()]

        decoded = detector.decode([MockResult()], image_width=640, image_height=640)
        assert len(decoded) == 1
        d = decoded[0]
        assert d.class_id == 0
        assert d.class_name == "crab_pot"
        assert d.is_filtered is True
        assert "Filtered per product policy" in d.filter_reason
