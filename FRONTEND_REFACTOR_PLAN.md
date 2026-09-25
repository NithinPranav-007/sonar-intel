# SONAR-INTEL: Frontend Refactoring Plan (Phase B)

**Document:** `FRONTEND_REFACTOR_PLAN.md`  
**Reference Document:** [`DESIGN.md`](file:///c:/Users/Asus/Desktop/SONAR-INTEL/DESIGN.md) & [`DESIGN_AUDIT.md`](file:///c:/Users/Asus/Desktop/SONAR-INTEL/DESIGN_AUDIT.md)  
**Execution Target:** Transform the current UI from cyberpunk/dark-terminal into the authoritative enterprise maritime operations portal defined in `DESIGN.md`.

---

## 1. Current Architecture Audit

### 1.1 Page & Router Structure
- **Root Shell**: [`src/pages/Dashboard.tsx`](file:///c:/Users/Asus/Desktop/SONAR-INTEL/frontend/src/pages/Dashboard.tsx) manages top-level active screen state (`activeScreen: ActiveScreen`) and renders one of 6 pages:
  1. `DashboardPage.tsx`
  2. `SonarAnalysisPage.tsx`
  3. `ContactVerificationPage.tsx`
  4. `GisMappingPage.tsx`
  5. `AiPipelinePage.tsx`
  6. `ReportsPage.tsx`
- **Navigation Container**: [`src/components/layout/MainLayout.tsx`](file:///c:/Users/Asus/Desktop/SONAR-INTEL/frontend/src/components/layout/MainLayout.tsx) wraps [`Sidebar.tsx`](file:///c:/Users/Asus/Desktop/SONAR-INTEL/frontend/src/components/layout/Sidebar.tsx) and [`Header.tsx`](file:///c:/Users/Asus/Desktop/SONAR-INTEL/frontend/src/components/layout/Header.tsx).

### 1.2 Styling & Tokens
- **Framework**: Tailwind CSS v4 (`@tailwindcss/vite` 4.3.3) with `@import "tailwindcss";` in `src/index.css`.
- **Current Aesthetic Flaws**:
  - `body` and `main` have dark backgrounds (`#050b14`, `#08101d`, `#070e1a`).
  - Ubiquitous `font-mono` applied to page headings, body labels, and buttons.
  - Saturated glowing cyan accents (`#00f0ff`, `text-cyan-300`, `border-cyan-800`, `shadow-[0_0_12px_rgba(6,182,212,0.3)]`).
- **Required Token Alignment**:
  - Main canvas background: `#f8fafc` (Slate-50)
  - Card surfaces: `#ffffff` (Pure white) with `#e2e8f0` (Slate-200) borders
  - Primary text: `#0f172a` (Slate-900), secondary: `#334155` (Slate-700), muted: `#64748b` (Slate-500)
  - Primary actions: `#0f172a` (Slate-900 solid button with white text)
  - Triage actions: Soft pastel pills (`#ecfdf5` green, `#fef2f2` red, `#fffbeb` amber)
  - Typography: `font-sans` (`Inter`) default for all UI text; `font-mono` isolated strictly to candidate IDs, coordinates, and telemetry.
  - Acoustic views: Embedded dark viewport `#070c18` strictly inside the sonar waterfall and detection crop containers.

### 1.3 State & API Integration
- **Survey Hook**: [`src/hooks/useSurvey.ts`](file:///c:/Users/Asus/Desktop/SONAR-INTEL/frontend/src/hooks/useSurvey.ts) handles:
  - `survey`, `contacts`, `selectedContact`, `navTrack`, `summary`, `analyzing`, `loading`.
  - `loadCuratedSample(sampleId)` connects directly to `POST /api/demo/load/{sample_id}`.
  - `submitReview(contactId, status, note)` updates review status and audit history in SQLite/PostGIS.
- **Backend Endpoints**:
  - `GET /api/demo/samples` & `POST /api/demo/load/{sample_id}`
  - `GET /api/dashboard/stats`
  - `GET /api/pipeline/info`
  - `GET /api/surveys/{survey_id}/geojson` & `GET /api/surveys/{survey_id}/csv`
  - `GET /api/contacts/search`

### 1.4 Data Honesty & Zero-Fabrication Audit
- **Coordinates**: The UI strictly checks `contact.latitude != null`. If missing, it displays `"Spatial coordinates unavailable"`. No fake GPS values exist.
- **AI Metrics**: Screen 5 displays verified baseline metrics (Validation mAP50: **6.45%**, Test mAP50: **10.48%**, Test Precision: **18.9%**, Test Recall: **12.9%**). No fictional 90%+ metrics are claimed.
- **Workflow Positioning**: Explicitly displayed as:
  `Raw SSS Swath -> AI Candidate Proposed -> Operator Triage -> Confirmed Anomaly`.

---

## 2. Refactoring Phases & Action Items

### Phase 2: Application Shell Refactoring
1. **`src/index.css`**:
   - Update CSS variables: `--bg-deep` to `#f8fafc`, `--text-main` to `#0f172a`, `--font-sans` to `'Inter', system-ui, sans-serif`.
   - Remove global dark body style; set background to `#f8fafc` and text to `#0f172a`.
2. **`src/components/layout/MainLayout.tsx`**:
   - Set container background to `#0b1329` (sidebar area) and `#f8fafc` (main content area).
3. **`src/components/layout/Sidebar.tsx`**:
   - Maintain deep navy `#0b1329`.
   - Amber brand badge container `#f59e0b` with dark navy trident icon.
   - Replace monospace with `font-sans` for nav item labels.
   - Active screen pill styled as `#182649` with white text.
   - Bottom operator card with white title "Dr. Clara Vance" and muted role "Senior Analyst".
4. **`src/components/layout/Header.tsx`**:
   - Background `#ffffff`, bottom border `#e2e8f0`.
   - Brand title in Slate-900, "OPERATIONAL MVP" in soft slate badge.
   - Center search bar: width 320px, light background `#f8fafc`, border `#e2e8f0`.
   - Curated demo selector dropdown styled as clean enterprise menu with 4 curated cases.
   - Status indicator in soft green.

### Phase 3: The Six Screens Refactoring
1. **`src/pages/DashboardPage.tsx`**:
   - Canvas `#f8fafc`.
   - 6 pure white stat cards (`#ffffff`) with `#e2e8f0` borders and Slate-900 numbers.
   - Clean SVG line chart on white card.
   - Clean horizontal bar chart for anomaly priorities on white card.
   - Coastal outline radar on white card.
   - Recent Platform Activity table on white surface with soft semantic status pills.
2. **`src/pages/SonarAnalysisPage.tsx`**:
   - 3-column architecture:
     - Left column: White cards for *Survey Details*, *Quality Metrics*, *Navigation Metadata*.
     - Center column: Clean white top toolbar (Zoom, Contrast slider, Normalization toggle, Box toggle); embedded dark acoustic viewport (`#070c18`) with the side-scan waterfall and 2px candidate bounding boxes; bottom 4-card acoustic evidence bar.
     - Right column: White card with *Detection Queue* list, dark square preview thumbnails, candidate IDs, confidence pills, and right arrows.
3. **`src/pages/ContactVerificationPage.tsx`**:
   - Left column: White card housing the dark acoustic crop viewport (`#070c18`); clean telemetry table below.
   - Right column:
     - White card with 3 one-click triage buttons: *[Confirm Contact]* (`#ecfdf5` / `#10b981`), *[False Positive]* (`#fef2f2` / `#ef4444`), *[Needs Review]* (`#fffbeb` / `#f59e0b`).
     - Operator notes textarea with solid dark slate *[Save & Continue]* button (`#0f172a`).
     - Verification audit log table on white surface.
4. **`src/pages/GisMappingPage.tsx`**:
   - Top bar with clean white styling and solid dark slate button *[Export GIS Layer]*.
   - MapLibre canvas with bathymetric nautical tiles.
   - Right drawer on white cards for *GIS Filters*, *Selected Contact Card*, and *Spatial Resolution & Datum*.
5. **`src/pages/AiPipelinePage.tsx`**:
   - White flowchart card showing 8 pipeline stages with green/yellow/slate status pills.
   - 4 white metric cards displaying real verified baseline numbers (6.45% val, 10.48% test) in Slate-900.
   - White model card specifications panel and execution log table.
6. **`src/pages/ReportsPage.tsx`**:
   - 4 white export cards (*Detections CSV*, *Spatial GeoJSON*, *Survey Summary*, *Baseline Model Card*) with solid dark slate buttons.
   - Consolidated triage totals card and recent exports log table.

### Phase 4: Curated Test-Set Demo Mode
- Stage 4 verified demo cases:
  1. `viator_04` (Held-Out Test Shipwreck — True Positive Benchmark, 83% confidence).
  2. `corsican_02` (Held-Out Test Shipwreck — Verified Anomaly matching ground truth, 54% confidence).
  3. `artificial_reef_02` (Held-Out Test Clutter — Operator Triage Demo, geological ridge rejection).
  4. `survey_001` (Operational Reference Swath with Towfish Navigation Track).
- Seamless switching via the topbar demo dropdown and survey hook.

---

## 3. Verification & Validation Strategy
1. Build verification with `npm run build` after each file update to guarantee zero TypeScript or CSS errors.
2. Full browser verification using `browser_subagent` across all 6 screens.
3. Verify that zero cyberpunk terminal styling remains and the enterprise visual standard is met.
4. Document the before-and-after audit in `UI_IMPLEMENTATION_AUDIT.md`.
