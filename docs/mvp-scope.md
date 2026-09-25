# SONAR-INTEL: Scope Boundaries

## 1. IN MVP Scope (4-Hour Delivery)

### Frontend
- **Framework**: React 18 + TypeScript + Vite.
- **Styling**: Dark tactical hydrographic theme with high information density, clean typography, and zero clutter.
- **Components**:
  - `SurveyUpload`: Drag-and-drop sonar image and optional navigation CSV with instant quality scoring.
  - `SonarViewer`: High-resolution canvas/SVG view with bounding box overlays, contact selection, and Raw vs. Processed (CLAHE) comparison toggle.
  - `MapView`: MapLibre GL geospatial map plotting vessel/towfish track and priority-coded contact markers (Red = HIGH, Amber = MEDIUM, Blue = LOW).
  - `EvidencePanel`: "Why was this flagged?" acoustic evidence breakdown (highlight, shadow, local contrast, quality).
  - `ReviewActions`: One-click triage (`CONFIRM`, `FALSE POSITIVE`, `UNCERTAIN`) and analyst annotation.
  - `ExportPanel`: Client and API export to GeoJSON and CSV.

### Backend
- **Framework**: Python 3.11+ with FastAPI & Pydantic v2.
- **Database**: PostgreSQL 15 + PostGIS with SQLAlchemy ORM and spatial indexing, including an automatic local fallback engine for zero-friction local execution.
- **Endpoints**: 8 focused REST endpoints covering upload, analysis, contact retrieval, review, GeoJSON export, and survey summary.
- **Resilience**: Graceful error handling for missing navigation, corrupted imagery, empty detections, or database outages.

### ML & Geospatial
- **Normalization**: SSS image normalization, CLAHE enhancement, water column / nadir handling, 640x640 tile generator.
- **Fast Processing**: Ultralytics YOLOv8n candidate anomaly detector with coordinate re-projection.
- **Acoustic Context**: Interpretable feature extraction (local contrast, acoustic shadow deficit, candidate geometry).
- **Decision Engine**: Multi-factor Priority Scoring (HIGH, MEDIUM, LOW) combining confidence, context, data quality, and localization quality.
- **Geolocation**: Estimated coordinates based on ping/timestamp navigation interpolation.

---

## 2. OUT OF MVP Scope (Future Extension Points)

The following capabilities are deliberately excluded from the initial 4-hour MVP to prevent over-engineering, but extension interfaces are preserved:

- **DeepLabV3-ResNet50 / Mask R-CNN**: Pixel-level mask segmentation of debris.
- **Foundation Models / SonarSAM**: Segment Anything adapted to sonar backscatter.
- **Discrete Wavelet Transform (DWT)**: Multi-scale speckle reduction.
- **Advanced Hydroacoustic Ray-Tracing**: Physical slant-range correction with sound velocity profiles (SVP) and bathymetric ray paths.
- **First Bottom Return (FBR) Automatic Tracking**: Continuous dynamic nadir depth tracking.
- **Graph Convolutional Networks (GCN)**: Modeling multi-fragment continuity of torn ghost nets across swaths.
- **Cross-swath Contact Merging**: Feature matching across adjacent overlapping survey survey lines.
- **Streaming Waterfall**: WebGL 60 FPS real-time sonar waterfall streaming from raw serial NMEA/XTF feeds.
- **Kubernetes / Microservices / Kafka / Redis**: Complex distributed infrastructure.
- **User Authentication / RBAC**: Multi-tenant authorization and enterprise SSO.
