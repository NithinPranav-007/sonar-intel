# SONAR-INTEL: System Architecture

## 1. Overview
**SONAR-INTEL** is an operational marine triage system designed for Side-Scan Sonar (SSS) data. It automates the detection, acoustic context verification, spatial localization, and human-in-the-loop review of artificial anomalies (such as marine debris, abandoned fishing gear/ghost nets, lost cargo, and maritime hazards).

The system follows an 8-stage operational framework ensuring clean separation of concerns, scientific rigor, and rapid developer turnaround.

```
[1. INPUT] SSS Image + Optional Navigation CSV (ping_id, timestamp, lat, lon, heading, altitude, range)
     │
     ▼
[2. INGESTION] FastAPI Multipart Upload, Validation, Provenance ID Generation, Disk Storage
     │
     ▼
[3. NORMALIZATION] Quality Metrics Check, Grayscale Normalization, Water-Column Blanking, CLAHE, Tiling
     │
     ▼
[4. FAST PROCESSING] YOLOv8n Candidate Anomaly Object Detection
     │
     ▼
[5. INTELLIGENCE] Acoustic-Context Analysis (Shadow Evidence, Local Contrast, Target Geometry)
     │
     ▼
[6. DECISION] Multi-Factor Priority Scoring (HIGH, MEDIUM, LOW)
     │
     ▼
[7. EXPLANATION / ACTION] Marine Analyst Dashboard, Sonar Viewer, MapLibre Sync, Human Triage
     │
     ▼
[8. STORAGE + FEEDBACK] PostgreSQL + PostGIS Spatial Storage, Human Review Audit Log, GeoJSON/CSV Exports
```

---

## 2. Stage Breakdown & Layer Responsibilities

### Stage 1: INPUT
- **Artifacts**: Side-scan sonar waterfall/strip images (`.png`, `.jpg`, `.tiff`) and optional navigation sensor logs (`.csv`).
- **Characteristics**: High dynamic range backscatter returns, speckle noise, varying grazing angles, nadir blind zone (water column).
- **Navigation**: Ping ID, timestamp, vessel/towfish latitude and longitude, heading, altitude, and slant range.

### Stage 2: INGESTION
- **Component**: `backend/app/api/upload.py` & `backend/app/services/sonar_service.py`
- **Responsibilities**:
  - Validates file formats, dimensions, and non-empty file bodies.
  - Generates immutable `survey_id` tracking survey provenance.
  - Stores unmodified raw sonar imagery in `data/raw/`. Raw input is **never** overwritten.
  - Parses and caches navigation metadata if provided.

### Stage 3: NORMALIZATION
- **Component**: `ml/preprocessing/` (`normalize.py`, `quality.py`, `tiling.py`, `pipeline.py`)
- **Responsibilities**:
  - **Quality Check**: Computes signal-to-noise ratio (SNR), dynamic range, blur index, and intensity distribution.
  - **Normalization**: Min-max acoustic intensity stretching to `[0, 255]`.
  - **Water Column / Nadir Handling**: Identifies nadir region and suppresses blind zone artifacts.
  - **Enhancement**: Contrast-Limited Adaptive Histogram Equalization (CLAHE) for highlighting subtle acoustic shadows and seabed returns.
  - **Tiling**: Splits large sonar swaths into overlapping tiles (e.g. 640x640) with coordinate projection back to survey space.

### Stage 4: FAST PROCESSING
- **Component**: `ml/inference/detector.py`
- **Responsibilities**:
  - Executes YOLOv8n candidate target detection on normalized tiles.
  - Detects candidate regions with bounding boxes (`x1, y1, x2, y2`) and raw model confidence.
  - Maps tile coordinates back to parent survey pixel coordinates.
  - Returns structured candidate dictionaries without coupling to databases.

### Stage 5: INTELLIGENCE
- **Component**: `ml/inference/context.py`
- **Responsibilities**:
  - Evaluates local acoustic context around each candidate.
  - **Local Contrast**: Compares candidate backscatter intensity against immediate surrounding seabed ambient backscatter.
  - **Acoustic Shadow Evidence**: Detects acoustic shadow (low/zero backscatter region) trailing the highlight down-range from acoustic incidence.
  - **Geometric Regularity**: Analyzes target aspect ratio, compactness, and linearity (distinguishing linear anthropogenic features like nets/cables from natural rocks/reefs).
  - Note: Low or absent shadow does not instantly disqualify a low-profile target; it produces graded evidence.

### Stage 6: DECISION
- **Component**: `ml/inference/scoring.py`
- **Responsibilities**:
  - Synthesizes multi-factor evidence into an operational **Priority Score**:
    $$\text{Priority} = w_1 \cdot \text{Confidence} + w_2 \cdot \text{Context} + w_3 \cdot \text{DataQuality} + w_4 \cdot \text{LocalizationQuality}$$
  - Classifies into triage tiers: `HIGH`, `MEDIUM`, `LOW`.
  - Assigns initial review state: `AI_CANDIDATE`.

### Stage 7: EXPLANATION / ACTION
- **Component**: React Dashboard (`frontend/src/`)
- **Responsibilities**:
  - High-density tactical hydrographic dark interface.
  - Dual-panel synchronized view: Sonar high-res viewer (with toggleable raw/processed/overlay) and MapLibre GL geospatial map.
  - Evidence inspection panel displaying "Why was this flagged?" breakdown.
  - One-click analyst review: `CONFIRM`, `FALSE POSITIVE`, `UNCERTAIN`, plus analyst notes.

### Stage 8: STORAGE + FEEDBACK
- **Component**: PostgreSQL 15 + PostGIS (`backend/app/database/`) & `outputs/`
- **Responsibilities**:
  - Spatial persistence of contacts using `GEOMETRY(Point, 4326)`.
  - Audit trail of human reviews with timestamps and model versions.
  - GeoJSON and CSV export for integration with ECDIS, QGIS, or mission logs.
  - Labeled dataset accumulation for future fine-tuning iterations.

---

## 3. Separation of Concerns & Scientific Discipline
The system enforces strict domain honesty by separating:
1. **Detection**: Statistical target anomaly extraction via YOLOv8n.
2. **Context**: Acoustic physics plausibility (backscatter highlight + shadow).
3. **Localization**: Mathematical estimation from towfish position and range (never claimed as survey-grade geodesy without acoustic USBL/DVL positioning).
4. **Confirmation**: Reserved exclusively for human sonar analysts.
