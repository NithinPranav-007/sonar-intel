-- SONAR-INTEL PostGIS Database Schema
-- Run in PostgreSQL with PostGIS extension enabled

CREATE EXTENSION IF NOT EXISTS postgis;

-- 1. Surveys Table: Track ingested sonar datasets & provenance
CREATE TABLE IF NOT EXISTS surveys (
    survey_id VARCHAR(64) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    raw_image_path VARCHAR(512) NOT NULL,
    processed_image_path VARCHAR(512),
    nav_file_path VARCHAR(512),
    image_width INTEGER,
    image_height INTEGER,
    data_quality DOUBLE PRECISION DEFAULT 1.0,
    has_navigation BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Contacts Table: Candidate targets with acoustic metrics and spatial point
CREATE TABLE IF NOT EXISTS contacts (
    contact_id VARCHAR(64) PRIMARY KEY,
    survey_id VARCHAR(64) NOT NULL REFERENCES surveys(survey_id) ON DELETE CASCADE,
    class_name VARCHAR(64) NOT NULL DEFAULT 'artificial_anomaly',
    confidence DOUBLE PRECISION NOT NULL,
    
    -- Pixel bounding box relative to survey image
    bbox_x1 INTEGER NOT NULL,
    bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL,
    bbox_y2 INTEGER NOT NULL,
    
    -- Acoustic & Intelligence evidence
    data_quality DOUBLE PRECISION DEFAULT 1.0,
    shadow_evidence DOUBLE PRECISION DEFAULT 0.0,
    context_score DOUBLE PRECISION DEFAULT 0.0,
    priority VARCHAR(16) NOT NULL DEFAULT 'MEDIUM', -- HIGH, MEDIUM, LOW
    
    -- Geospatial coordinates (WGS84)
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    location GEOMETRY(Point, 4326),
    localization_status VARCHAR(24) NOT NULL DEFAULT 'UNAVAILABLE', -- ESTIMATED, VERIFIED, UNCERTAIN, UNAVAILABLE
    
    -- Human-in-the-loop triage
    review_status VARCHAR(24) NOT NULL DEFAULT 'AI_CANDIDATE', -- AI_CANDIDATE, CONFIRMED, FALSE_POSITIVE, UNCERTAIN
    review_note TEXT,
    
    model_version VARCHAR(32) NOT NULL DEFAULT 'yolov8n-v1',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Spatial and relational indexes
CREATE INDEX IF NOT EXISTS idx_contacts_location ON contacts USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_contacts_survey_id ON contacts(survey_id);
CREATE INDEX IF NOT EXISTS idx_contacts_priority ON contacts(priority);
CREATE INDEX IF NOT EXISTS idx_contacts_review_status ON contacts(review_status);

-- 3. Reviews Audit Table: Historical tracking for active learning feedback
CREATE TABLE IF NOT EXISTS reviews (
    review_id SERIAL PRIMARY KEY,
    contact_id VARCHAR(64) NOT NULL REFERENCES contacts(contact_id) ON DELETE CASCADE,
    survey_id VARCHAR(64) NOT NULL,
    review_status VARCHAR(24) NOT NULL,
    review_note TEXT,
    model_version VARCHAR(32) NOT NULL,
    reviewed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reviews_contact_id ON reviews(contact_id);
