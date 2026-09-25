"""
Tests for Detection to Canonical Contact Transformation Service.
"""

import pytest
from ml.inference.drishti_detector import DrishtiDetection
from backend.app.services.transformer import transform_drishti_detections_to_contacts
from backend.app.schemas.contact import Contact


class TestContactTransformation:
    def test_filtered_classes_are_excluded_from_contacts(self):
        detections = [
            DrishtiDetection(
                class_id=0,
                class_name="crab_pot",
                confidence=0.85,
                bbox=[100, 100, 200, 200],
                image_width=640,
                image_height=640,
                is_filtered=True,
                filter_reason="Filtered per product policy"
            ),
            DrishtiDetection(
                class_id=2,
                class_name="shipwreck",
                confidence=0.92,
                bbox=[300, 300, 450, 400],
                image_width=640,
                image_height=640,
                is_filtered=False
            )
        ]

        contacts = transform_drishti_detections_to_contacts(
            detections=detections,
            survey_id="SURVEY_TEST_001"
        )

        # crab_pot must be filtered out; only shipwreck becomes a Contact
        assert len(contacts) == 1
        contact = contacts[0]
        assert contact.class_name == "shipwreck"
        assert contact.confidence == 0.92
        assert contact.model_score == 0.92
        assert contact.calibrated_confidence is None
        assert contact.review_status == "AI_CANDIDATE"
        assert contact.survey_id == "SURVEY_TEST_001"

    def test_no_coordinate_fabrication(self):
        detections = [
            DrishtiDetection(
                class_id=3,
                class_name="ghost_net",
                confidence=0.78,
                bbox=[50, 50, 120, 120],
                image_width=640,
                image_height=640,
                is_filtered=False
            )
        ]

        # No geolocation service provided
        contacts = transform_drishti_detections_to_contacts(
            detections=detections,
            survey_id="SURVEY_NO_NAV",
            geo_service=None
        )

        assert len(contacts) == 1
        c = contacts[0]
        assert c.latitude is None
        assert c.longitude is None
        assert c.localization_status == "UNAVAILABLE"

    def test_coordinate_attachment_when_navigation_valid(self):
        class MockGeoService:
            def estimate_contact_location(self, bbox_center_x, bbox_center_y, image_width, image_height):
                return 54.12345, 12.67890, "ESTIMATED"

        detections = [
            DrishtiDetection(
                class_id=1,
                class_name="submarine_pipeline",
                confidence=0.81,
                bbox=[200, 200, 300, 250],
                image_width=640,
                image_height=640,
                is_filtered=False
            )
        ]

        contacts = transform_drishti_detections_to_contacts(
            detections=detections,
            survey_id="SURVEY_WITH_NAV",
            geo_service=MockGeoService()
        )

        assert len(contacts) == 1
        c = contacts[0]
        assert c.latitude == 54.12345
        assert c.longitude == 12.67890
        assert c.localization_status == "ESTIMATED"
        assert c.location_uncertainty is not None

    def test_priority_assignment_rules(self):
        high_det = DrishtiDetection(
            class_id=2, class_name="shipwreck", confidence=0.85,
            bbox=[10, 10, 50, 50], image_width=640, image_height=640
        )
        med_det = DrishtiDetection(
            class_id=4, class_name="mine_cylinder", confidence=0.55,
            bbox=[10, 10, 50, 50], image_width=640, image_height=640
        )
        low_det = DrishtiDetection(
            class_id=3, class_name="ghost_net", confidence=0.30,
            bbox=[10, 10, 50, 50], image_width=640, image_height=640
        )

        contacts = transform_drishti_detections_to_contacts(
            detections=[high_det, med_det, low_det],
            survey_id="SURVEY_PRIO"
        )

        assert len(contacts) == 3
        assert contacts[0].priority == "HIGH"
        assert contacts[1].priority == "MEDIUM"
        assert contacts[2].priority == "LOW"
