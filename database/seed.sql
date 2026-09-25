-- SONAR-INTEL Seed Data for Verification & Testing

INSERT INTO surveys (survey_id, filename, raw_image_path, processed_image_path, nav_file_path, image_width, image_height, data_quality, has_navigation)
VALUES (
    'SURVEY_001',
    'survey_001_raw.png',
    'data/demo/sonar/survey_001_raw.png',
    'data/demo/sonar/survey_001_processed.png',
    'data/demo/navigation/survey_001_nav.csv',
    1280,
    1800,
    0.96,
    TRUE
) ON CONFLICT (survey_id) DO NOTHING;

-- Contact 1: Strong cylindrical artificial anomaly with clear acoustic shadow (High Priority)
INSERT INTO contacts (
    contact_id, survey_id, class_name, confidence,
    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
    data_quality, shadow_evidence, context_score, priority,
    latitude, longitude, location, localization_status,
    review_status, review_note, model_version
) VALUES (
    'C001', 'SURVEY_001', 'artificial_anomaly', 0.91,
    412, 188, 531, 302,
    0.96, 0.82, 0.87, 'HIGH',
    11.23451, 76.54321, ST_SetSRID(ST_MakePoint(76.54321, 11.23451), 4326), 'ESTIMATED',
    'AI_CANDIDATE', NULL, 'yolov8n-v1'
) ON CONFLICT (contact_id) DO NOTHING;

-- Contact 2: Linear anomaly (possible abandoned fishing net / rope fragment) (Medium Priority)
INSERT INTO contacts (
    contact_id, survey_id, class_name, confidence,
    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
    data_quality, shadow_evidence, context_score, priority,
    latitude, longitude, location, localization_status,
    review_status, review_note, model_version
) VALUES (
    'C002', 'SURVEY_001', 'artificial_anomaly', 0.74,
    780, 520, 910, 680,
    0.94, 0.58, 0.71, 'MEDIUM',
    11.23485, 76.54362, ST_SetSRID(ST_MakePoint(76.54362, 11.23485), 4326), 'ESTIMATED',
    'AI_CANDIDATE', NULL, 'yolov8n-v1'
) ON CONFLICT (contact_id) DO NOTHING;

-- Contact 3: Low profile anomalous patch (Low Priority / Uncertain)
INSERT INTO contacts (
    contact_id, survey_id, class_name, confidence,
    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
    data_quality, shadow_evidence, context_score, priority,
    latitude, longitude, location, localization_status,
    review_status, review_note, model_version
) VALUES (
    'C003', 'SURVEY_001', 'artificial_anomaly', 0.52,
    220, 1120, 310, 1210,
    0.91, 0.28, 0.44, 'LOW',
    11.23520, 76.54398, ST_SetSRID(ST_MakePoint(76.54398, 11.23520), 4326), 'ESTIMATED',
    'AI_CANDIDATE', NULL, 'yolov8n-v1'
) ON CONFLICT (contact_id) DO NOTHING;
