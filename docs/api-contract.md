# SONAR-INTEL: API Contract & Data Specifications

## 1. Canonical Contact JSON Specification

The **Canonical Contact** object is the unified data contract shared across the ML inference pipeline, FastAPI backend services, PostGIS relational storage, and the React frontend.

All modules must adhere strictly to this schema.

```json
{
  "contact_id": "C001",
  "survey_id": "SURVEY_001",
  "class_name": "artificial_anomaly",
  "confidence": 0.91,
  "bbox": {
    "x1": 412,
    "y1": 188,
    "x2": 531,
    "y2": 302
  },
  "data_quality": 0.96,
  "shadow_evidence": 0.82,
  "context_score": 0.87,
  "priority": "HIGH",
  "latitude": 11.23451,
  "longitude": 76.54321,
  "localization_status": "ESTIMATED",
  "review_status": "AI_CANDIDATE",
  "review_note": null,
  "model_version": "yolov8n-v1"
}
```

### Field Definitions

| Field | Type | Allowed Values / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `contact_id` | `string` | Unique identifier (e.g. `C001`, UUID) | Unique identifier for the sonar contact |
| `survey_id` | `string` | Foreign key matching parent survey | Identifier of the ingested survey dataset |
| `class_name` | `string` | `artificial_anomaly`, `debris_candidate`, etc. | Candidate classification label |
| `confidence` | `float` | `0.0` - `1.0` | Raw confidence output from YOLO detector |
| `bbox` | `object` | `{ x1: int, y1: int, x2: int, y2: int }` | Pixel bounding box relative to survey image |
| `data_quality` | `float` | `0.0` - `1.0` | Computed signal quality of the parent sonar tile |
| `shadow_evidence`| `float` | `0.0` - `1.0` | Acoustic shadow deficit ratio behind highlight |
| `context_score` | `float` | `0.0` - `1.0` | Combined acoustic plausibility score |
| `priority` | `string` | `"HIGH"`, `"MEDIUM"`, `"LOW"` | Operational triage priority |
| `latitude` | `float \| null`| WGS84 Latitude (`-90` to `90`) or `null` | Estimated geographic latitude |
| `longitude`| `float \| null`| WGS84 Longitude (`-180` to `180`) or `null`| Estimated geographic longitude |
| `localization_status` | `string` | `"ESTIMATED"`, `"VERIFIED"`, `"UNCERTAIN"`, `"UNAVAILABLE"` | Quality state of spatial coordinates |
| `review_status` | `string` | `"AI_CANDIDATE"`, `"CONFIRMED"`, `"FALSE_POSITIVE"`, `"UNCERTAIN"` | Human-in-the-loop triage status |
| `review_note` | `string \| null`| Free text string or `null` | Optional analyst remarks |
| `model_version` | `string` | e.g. `"yolov8n-v1"` | Model version identifier for provenance |

---

## 2. API Endpoints

### 1. Health Check
- **`GET /api/health`**
- **Response**: `200 OK`
```json
{
  "status": "healthy",
  "database": "connected",
  "model_loaded": true,
  "timestamp": "2026-08-31T20:55:00Z"
}
```

### 2. Survey Upload
- **`POST /api/surveys/upload`**
- **Form Data**:
  - `sonar_file`: File (image/png, image/jpeg, image/tiff) - Required
  - `nav_file`: File (text/csv) - Optional
- **Response**: `201 Created`
```json
{
  "survey_id": "SURV_20260831_143022",
  "filename": "demo_sonar.png",
  "image_width": 1280,
  "image_height": 2048,
  "data_quality": 0.94,
  "has_navigation": true,
  "message": "Survey uploaded and validated successfully."
}
```

### 3. Trigger Analysis
- **`POST /api/surveys/{survey_id}/analyze`**
- **Body**: `{ "clahe": true, "confidence_threshold": 0.25 }` (Optional)
- **Response**: `200 OK`
```json
{
  "survey_id": "SURV_20260831_143022",
  "contacts_count": 3,
  "contacts": [ /* Array of Canonical Contact JSONs */ ],
  "execution_time_ms": 342.5
}
```

### 4. Get Contacts for Survey
- **`GET /api/surveys/{survey_id}/contacts`**
- **Response**: `200 OK`
```json
[
  /* Array of Canonical Contact JSONs */
]
```

### 5. Get Single Contact
- **`GET /api/contacts/{contact_id}`**
- **Response**: `200 OK`
```json
{
  /* Canonical Contact JSON */
}
```

### 6. Submit Human Review
- **`POST /api/contacts/{contact_id}/review`**
- **Request Body**:
```json
{
  "review_status": "CONFIRMED",
  "review_note": "Distinct cylindrical return with 3.2m acoustic shadow. Matches lost container/crate profile."
}
```
- **Response**: `200 OK` (Returns updated Canonical Contact JSON)

### 7. GeoJSON Export
- **`GET /api/surveys/{survey_id}/geojson`**
- **Response**: `200 OK` (`application/geo+json`)
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [76.54321, 11.23451]
      },
      "properties": {
        "contact_id": "C001",
        "class_name": "artificial_anomaly",
        "confidence": 0.91,
        "priority": "HIGH",
        "shadow_evidence": 0.82,
        "context_score": 0.87,
        "data_quality": 0.96,
        "review_status": "CONFIRMED",
        "model_version": "yolov8n-v1"
      }
    }
  ]
}
```

### 8. Survey Summary
- **`GET /api/surveys/{survey_id}/summary`**
- **Response**: `200 OK`
```json
{
  "survey_id": "SURV_20260831_143022",
  "total_contacts": 5,
  "high_priority": 2,
  "medium_priority": 2,
  "low_priority": 1,
  "reviewed_count": 3,
  "pending_count": 2,
  "data_quality_avg": 0.93
}
```
