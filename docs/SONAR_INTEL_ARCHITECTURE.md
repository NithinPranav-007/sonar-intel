# SONAR-INTEL: System Architecture Specification

**Document Identifier:** `DOC-SONAR-INTEL-ARCH-2026.09`  
**Classification:** Internal Technical Architecture & System Specification  
**Author:** Lead Technical Documentation Engineer  
**Date:** September 5, 2026  
**Repository:** `salmonangelo/Sonar-Intel`  
**Status:** Implemented MVP vs. Target Production Architecture  

---

## 1. Architectural Overview

SONAR-INTEL is an edge-native, decoupled client-server platform designed for hydrographic anomaly screening and human-in-the-loop triage. The system bridges raw acoustic side-scan sonar waterfall imagery with enterprise geospatial information systems (GIS) and maritime electronic navigational chart (ENC/ECDIS) displays.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             SONAR-INTEL ARCHITECTURE                             │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   [ SSS Waterfall Image ]       [ Towfish Nav Log (CSV/XTF) ]                     │
│              │                               │                                   │
│              ▼                               ▼                                   │
│   ┌───────────────────────────────────────────────────────┐                      │
│   │            FastAPI INGESTION & QUALITY SERVICE        │                      │
│   │  • Bit-depth validation   • SNR dynamic range check   │                      │
│   │  • 1-99% Percentile Norm  • Towfish time-sync         │                      │
│   └───────────────────────────┬───────────────────────────┘                      │
│                               │                                                  │
│                               ▼                                                  │
│   ┌───────────────────────────────────────────────────────┐                      │
│   │              ML PREPROCESSING & INFERENCE             │                      │
│   │  • Vectorized Lee MMSE    • Deterministic 640x640 Tile │                     │
│   │  • Adaptive CLAHE Filter  • YOLOv8s + SSS-Net (FP16)  │                      │
│   └───────────────────────────┬───────────────────────────┘                      │
│                               │                                                  │
│                               ▼                                                  │
│   ┌───────────────────────────────────────────────────────┐                      │
│   │            POSTPROCESSING & EVIDENCE ENGINE           │                      │
│   │  • Spatial Border NMS     • Shadow Deficit Ratio      │                      │
│   │  • Class Policy Filter    • Composite Priority Score  │                      │
│   └───────────────────────────┬───────────────────────────┘                      │
│                               │                                                  │
│                               ▼                                                  │
│   ┌───────────────────────────────────────────────────────┐                      │
│   │             GEODESIC POSITIONING ENGINE               │                      │
│   │  • Ping index association • Slant-to-ground projection│                      │
│   │  • Geodesic azimuth fix   • WGS-84 coordinate mapping │                      │
│   └───────────────────────────┬───────────────────────────┘                      │
│                               │                                                  │
│                               ▼                                                  │
│   ┌───────────────────────────────────────────────────────┐                      │
│   │             PERSISTENCE & SPATIAL STORAGE             │                      │
│   │  • PostgreSQL 16 / PostGIS (EPSG:4326 Geography)      │                      │
│   │  • Automated SQLite Local Fallback Mode               │                      │
│   └───────────────────────────┬───────────────────────────┘                      │
│                               │                                                  │
│                               ▼                                                  │
│   ┌───────────────────────────────────────────────────────┐                      │
│   │            OPERATOR CLIENT CONSOLE (React 18)         │                      │
│   │  • Sonar Waterfall View   • MapLibre GL Nautical GIS  │                      │
│   │  • Contact Triage Console • Pipeline Telemetry Monitor│                      │
│   │  • RFC 7946 GeoJSON Gen   • Tabular CSV Data Exporter │                      │
│   └───────────────────────────────────────────────────────┘                      │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture

### 2.1 Backend Services Layer (FastAPI)
- **Framework:** FastAPI 0.110 running on Uvicorn asynchronous ASGI server.
- **Key Modules:**
  - `backend/app/api/surveys.py`: Manages swath ingestion, file storage, and triggering detection pipelines.
  - `backend/app/api/inference.py`: Standalone multi-class inference endpoint with multipart image uploading.
  - `backend/app/api/contacts.py`: Contact management, human-in-the-loop review state updates, and audit logging.
  - `backend/app/api/demo.py`: Curated benchmark loader linking real sonar swaths with verified navigation tracks.
  - `backend/app/api/pipeline.py`: Telemetry reporting for model card specs, active stages, and latency audit logs.

### 2.2 Deep Learning & Machine Learning Engine (`ml/`)
- **Framework:** PyTorch 2.6.0 with CUDA 12.6 acceleration (automatic CPU fallback).
- **Inference Runtime (`ml/inference/drishti_detector.py`):**
  - **Singleton Pattern:** Thread-safe, process-level model loading preventing redundant GPU memory allocation.
  - **Precision:** FP16 Automatic Mixed Precision (AMP) on CUDA Tensor Cores.
  - **Class Filtering Policy:** Automatically preserves raw detector predictions while tagging unwanted clutter (`crab_pot`) as filtered.
- **Preprocessing Runtime (`ml/preprocessing/drishti_preprocess.py`):**
  - **Lee Speckle MMSE Filter:** Vectorized NumPy / OpenCV uniform box-filter implementation ($5\times 5$ window) executing in $<2.0\text{ ms}$.
  - **Adaptive CLAHE:** Equalizes contrast boundaries between specular acoustic highlights and dark shadow voids.

### 2.3 Persistence & Spatial Storage Layer (`backend/app/database/`)
- **Primary Engine:** PostgreSQL 16 with PostGIS extension (`Geography(Point, 4326)`).
- **Resilience Architecture:** Automated connection fallback to local SQLite (`backend/sonar_intel_fallback.db`) with dynamic geometry string serialization when PostgreSQL is unavailable during field operations.
- **Core Entities:**
  - `Survey`: Metadata for raw swath files, dimensions, timestamps, and processing status.
  - `Contact`: Spatial entity storing pixel bounding boxes, WGS-84 fixes, classification, confidence, priority, and review status.
  - `AuditLog`: Immutable history of human operator decisions, notes, and timestamps.

### 2.4 Frontend Operations Console (`frontend-new/`)
- **Technology:** React 18.2, TypeScript 5.2, Vite, TailwindCSS.
- **Visual Engine:** MapLibre GL 4.1 hardware-accelerated WebGL vector chart renderer.
- **Design System:** Placely Design System with dedicated high-density dark hydrographic theme.

---

## 3. End-to-End Data Flow & Lifecycle

```
[Raw SSS Image] + [Nav CSV]
         │
         ├───> POST /api/surveys/upload
         │
         ▼
[Ingestion Service]
  ├── Bit-depth verification (uint8 / uint16)
  ├── 1-99% Dynamic Range Percentile Stretch
  └── Storage in `data/uploads/`
         │
         ▼
[Inference Service: run_survey_analysis()]
  ├── Deterministic Slicing: 640x640 tiles with 20% stride overlap (512px step)
  ├── Vectorized Lee MMSE Filter (5x5, noise_var=0.04)
  ├── CLAHE Equalization (clipLimit=2.0, tileGrid=(8,8))
  ├── Batched YOLOv8s FP16 Inference
  └── Bounding Box coordinate global remapping
         │
         ▼
[Postprocessing & Candidate Ranking]
  ├── Non-Maximum Suppression (IoU >= 0.45) across tile borders
  ├── Sliver filter (removes edge-clipped fragments < 15px)
  ├── Acoustic Highlight-to-Shadow Deficit Ratio calculation
  └── Product Class Policy Filtering (excludes crab_pot from active queue)
         │
         ▼
[Geolocation Service: attach_georeferencing()]
  ├── Along-track ping timestamp matching
  ├── Slant-range ground projection
  ├── Forward Geodesic Vincenty / Haversine position calculation
  └── Coordinate status tagging (ESTIMATED vs UNAVAILABLE)
         │
         ▼
[Database Persistence]
  └── Insert `Contact` records with status = `AI_CANDIDATE`
         │
         ▼
[Operator Client Console]
  ├── Interactive 2D Waterfall viewport with bounding boxes
  ├── MapLibre GL Nautical Chart with towfish track & pulsating contact pins
  ├── Operator Triage: POST /api/contacts/{id}/review -> `CONFIRMED`
  └── Data Product Export: GET /api/surveys/{id}/geojson -> RFC 7946 Standard
```

---

## 4. Geodesic Positioning & Slant-Range Mathematics

The Geolocation Service (`backend/app/services/geolocation_service.py`) calculates real-world coordinates from pixel coordinates $(u, v)$ without coordinate fabrication:

1. **Along-Track Row to Ping Index Association:**
   $$\text{ping\_idx} = \text{round}\left(\frac{v}{H_{\text{image}}} \cdot (N_{\text{pings}} - 1)\right)$$
2. **Towfish Primary Telemetry:**
   $$\phi_{\text{vessel}}, \lambda_{\text{vessel}}, \theta_{\text{heading}} = \text{Lookup}(\text{ping\_idx})$$
3. **Across-Track Pixel to Slant-Range Conversion:**
   $$x_{\text{pixel}} = u - u_{\text{nadir}}$$
   $$R_{\text{slant}} = |x_{\text{pixel}}| \cdot \Delta r$$
   where $\Delta r = \frac{\text{swath\_width\_meters}}{\text{image\_width\_pixels}}$.
4. **Pythagorean Ground-Range Projection:**
   $$R_{\text{ground}} = \sqrt{\max\left(0, R_{\text{slant}}^2 - h_{\text{altitude}}^2\right)}$$
5. **Target Offset Azimuth:**
   $$\theta_{\text{target}} = \begin{cases} \theta_{\text{heading}} - 90^\circ & \text{if } x_{\text{pixel}} < 0 \text{ (Portside)} \\ \theta_{\text{heading}} + 90^\circ & \text{if } x_{\text{pixel}} \ge 0 \text{ (Starboard)} \end{cases}$$
6. **Geodesic Target Fix (WGS-84):**
   $$\phi_{\text{target}} = \arcsin\left(\sin \phi_{\text{vessel}} \cos\left(\frac{R_{\text{ground}}}{R_E}\right) + \cos \phi_{\text{vessel}} \sin\left(\frac{R_{\text{ground}}}{R_E}\right) \cos \theta_{\text{target}}\right)$$
   $$\lambda_{\text{target}} = \lambda_{\text{vessel}} + \operatorname{atan2}\left(\sin \theta_{\text{target}} \sin\left(\frac{R_{\text{ground}}}{R_E}\right) \cos \phi_{\text{vessel}}, \cos\left(\frac{R_{\text{ground}}}{R_E}\right) - \sin \phi_{\text{vessel}} \sin \phi_{\text{target}}\right)$$

---

## 5. Implemented vs. Target Production Architecture

| Subsystem | Implemented Architecture (Current Status) | Target Production Architecture (Future Roadmap) | Status |
| :--- | :--- | :--- | :--- |
| **Ingestion Engine** | Image files (`.png`, `.jpg`, `.tif`) + optional CSV navigation logs via REST multipart upload. | Real-time streaming parser for raw binary `.xtf`, `.jsf`, and `.all` acoustic datagrams via WebSockets/gRPC. | **PARTIAL** |
| **Acoustic Preprocessing** | Vectorized Lee MMSE Filter ($5\times 5$) + Adaptive CLAHE + 1–99% Percentile Stretch. | Multi-frequency acoustic beam-pattern EGN correction and CTD sound velocity profile ray-tracing. | **IMPLEMENTED** |
| **Neural Anomaly Detector** | Dual Baseline: (1) Trained Single-Class YOLOv8n, (2) Pretrained Multi-Class DRISHTI YOLOv8s. | Multi-scale Masked Autoencoder Foundation Model + Rotated Bounding Box (OBB) heads. | **IMPLEMENTED** |
| **Postprocessing & NMS** | Spatial IoU NMS + boundary sliver filter + acoustic shadow deficit heuristic scoring. | Learned multi-ping tracking association via spatio-temporal Kalman filter / Graph Neural Networks. | **IMPLEMENTED** |
| **Geodesic Geotagging** | Linear ping interpolation + Pythagorean ground-range projection; explicit `UNAVAILABLE` when nav absent. | Dynamic towfish catenary layback modeling, USBL acoustic positioning fusion, and INS attitude compensation. | **IMPLEMENTED** |
| **Operator Triage Console** | 6 dedicated React workspaces with 1-click classification, diagnostics, and audit trail. | Multi-operator collaborative triage with role-based access control (RBAC) and S-57 ENC chart overlays. | **IMPLEMENTED** |
| **Persistence Layer** | PostgreSQL 16 + PostGIS with automated local SQLite fallback (`sonar_intel_fallback.db`). | Distributed PostGIS cluster with spatial partitioning and S3 object storage for raw multi-gigabyte surveys. | **IMPLEMENTED** |
| **Data Products** | RFC 7946 GeoJSON spatial export, tabular CSV logs, executive audit summaries, and model cards. | Automated OGC WFS/WMS map services, direct hydrographic CARIS HIPS/SIPS and QPS Qimera exports. | **IMPLEMENTED** |
| **Mission Route Optimizer** | Prototype greedy nearest-neighbor candidate sequence preview. | Multi-vehicle MILP path planner optimizing AUV survey routes with turn radius and sea current constraints. | **PROTOTYPE** |

---

## 6. API Interface Contracts

### 6.1 Survey Ingestion (`POST /api/surveys/upload`)
- **Request:** `multipart/form-data` containing:
  - `sonar_file`: Binary SSS waterfall image (`.png`, `.jpg`, `.tif`).
  - `nav_file` *(optional)*: CSV navigation log containing `ping_id`, `latitude`, `longitude`, `heading`.
- **Response (200 OK):**
```json
{
  "survey_id": "SURVEY_20260905_120000",
  "filename": "survey_track_01.png",
  "dimensions": {"width": 1728, "height": 4500},
  "has_navigation": true,
  "status": "INGESTED"
}
```

### 6.2 Autonomous Survey Analysis (`POST /api/surveys/{id}/analyze`)
- **Request:** JSON parameters for confidence threshold and tile stride.
- **Response (200 OK):**
```json
{
  "survey_id": "SURVEY_20260905_120000",
  "total_tiles_processed": 24,
  "candidates_discovered": 4,
  "contacts": [
    {
      "contact_id": "C001",
      "classification": "shipwreck",
      "ai_confidence": 0.83,
      "priority": "HIGH",
      "review_status": "AI_CANDIDATE",
      "coordinates": {
        "latitude": 54.124577,
        "longitude": 12.680117,
        "provenance": "ESTIMATED"
      }
    }
  ]
}
```

### 6.3 Standalone Multi-Class Inference (`POST /api/inference/detect`)
- **Request:** `multipart/form-data` with single sonar tile (`file: UploadFile`).
- **Response (200 OK):**
```json
{
  "inference_id": "INF_9F574FA6",
  "model_version": "DRISHTI-YOLOv8s-v1",
  "raw_detections_count": 2,
  "valid_contacts_count": 1,
  "detections": [
    {
      "class_name": "shipwreck",
      "confidence": 0.84,
      "bbox_normalized": [0.32, 0.45, 0.18, 0.22],
      "is_filtered": false
    },
    {
      "class_name": "crab_pot",
      "confidence": 0.52,
      "bbox_normalized": [0.78, 0.12, 0.04, 0.05],
      "is_filtered": true,
      "filter_reason": "Filtered per product policy"
    }
  ]
}
```

### 6.4 Operator Triage Review (`POST /api/contacts/{id}/review`)
- **Request Body:**
```json
{
  "review_status": "CONFIRMED",
  "reviewer_id": "HYDRO_SURVEYOR_01",
  "notes": "Verified acoustic shadow matches steel hull superstructure."
}
```

### 6.5 RFC 7946 Standard GeoJSON Export (`GET /api/surveys/{id}/geojson`)
- **Response (200 OK):** Standard FeatureCollection ready for direct drag-and-drop into QGIS / ArcGIS:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [12.680117, 54.124577]
      },
      "properties": {
        "contact_id": "C001",
        "survey_id": "DEMO_VIATOR_04_1788410409",
        "classification": "shipwreck",
        "confidence": 0.83,
        "review_status": "CONFIRMED",
        "priority": "HIGH",
        "provenance": "ESTIMATED"
      }
    }
  ]
}
```

---

## 7. Storage & Persistence Schema

```sql
-- PostgreSQL / PostGIS Primary Schema

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE surveys (
    id VARCHAR(64) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    filepath VARCHAR(512) NOT NULL,
    width_px INT NOT NULL,
    height_px INT NOT NULL,
    has_navigation BOOLEAN DEFAULT FALSE,
    nav_filepath VARCHAR(512),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(32) DEFAULT 'INGESTED'
);

CREATE TABLE contacts (
    id VARCHAR(64) PRIMARY KEY,
    survey_id VARCHAR(64) REFERENCES surveys(id) ON DELETE CASCADE,
    contact_number VARCHAR(16) NOT NULL,
    classification VARCHAR(64) NOT NULL,
    confidence FLOAT NOT NULL,
    priority VARCHAR(16) NOT NULL,
    review_status VARCHAR(32) DEFAULT 'AI_CANDIDATE',
    
    -- Pixel Bounding Box
    bbox_x_min INT NOT NULL,
    bbox_y_min INT NOT NULL,
    bbox_x_max INT NOT NULL,
    bbox_y_max INT NOT NULL,
    
    -- Geospatial Positioning (WGS-84 EPSG:4326)
    geom GEOGRAPHY(POINT, 4326),
    latitude FLOAT,
    longitude FLOAT,
    localization_status VARCHAR(32) DEFAULT 'UNAVAILABLE',
    
    -- Acoustic Diagnostics
    shadow_deficit_ratio FLOAT,
    slant_range_m FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    contact_id VARCHAR(64) REFERENCES contacts(id) ON DELETE CASCADE,
    reviewer_id VARCHAR(64) NOT NULL,
    previous_status VARCHAR(32) NOT NULL,
    new_status VARCHAR(32) NOT NULL,
    notes TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Spatial & Telemetry Indexes
CREATE INDEX idx_contacts_geom ON contacts USING GIST (geom);
CREATE INDEX idx_contacts_survey ON contacts (survey_id);
CREATE INDEX idx_contacts_status ON contacts (review_status);
```
