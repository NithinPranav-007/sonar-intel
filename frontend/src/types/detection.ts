/**
 * Canonical Contact Data Type.
 * Authoritative contract shared across ML, FastAPI, PostGIS, and React.
 */

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export type PriorityLevel = 'HIGH' | 'MEDIUM' | 'LOW';

export type LocalizationStatus = 'ESTIMATED' | 'VERIFIED' | 'UNCERTAIN' | 'UNAVAILABLE';

export type ReviewStatus = 'AI_CANDIDATE' | 'CONFIRMED' | 'FALSE_POSITIVE' | 'UNCERTAIN';

export interface Contact {
  contact_id: string;
  survey_id: string;
  class_name: string;
  confidence: number;
  bbox: BoundingBox;
  data_quality: number;
  shadow_evidence: number;
  context_score: number;
  priority: PriorityLevel;
  latitude: number | null;
  longitude: number | null;
  localization_status: LocalizationStatus;
  review_status: ReviewStatus;
  review_note: string | null;
  model_version: string;
}

export interface SurveyUploadResponse {
  survey_id: string;
  filename: string;
  image_width: number;
  image_height: number;
  data_quality: number;
  has_navigation: boolean;
  raw_image_url: string;
  processed_image_url?: string;
  message: string;
}

export interface SurveySummary {
  survey_id: string;
  filename: string;
  total_contacts: number;
  high_priority: number;
  medium_priority: number;
  low_priority: number;
  reviewed_count: number;
  pending_count: number;
  data_quality_avg: number;
}

export interface NavWaypoint {
  latitude: number;
  longitude: number;
  ping_id: number;
  heading: number;
}
