---
version: 1.0.0-authoritative
name: SONAR-INTEL Design System
description: Authoritative, implementation-oriented design system for SONAR-INTEL maritime intelligence, hydrographic anomaly candidate detection, and human-in-the-loop triage operations portal.
tokens:
  colors:
    workspace:
      canvas: "#f8fafc" # Slate-50: Clean neutral light gray application workspace
      surface: "#ffffff" # Pure white: Primary cards, data tables, and modal dialogs
      surface_muted: "#f1f5f9" # Slate-100: Secondary card wells, control bars, filter tracks
      surface_acoustic: "#070c18" # Deep obsidian navy: Reserved exclusively for sonar waterfall and acoustic crops
    navigation:
      background: "#0b1329" # Midnight maritime navy: Persistent enterprise navigation rail
      border: "#172342" # Muted slate navy divider
      item_default: "#94a3b8" # Slate-400: Unselected navigation icon and typography
      item_hover: "#cbd5e1" # Slate-300: Hovered navigation item
      item_active_bg: "#182649" # Slate-800 elevated navy pill for active screen
      item_active_text: "#ffffff" # Crisp white for active screen label
      brand_badge_bg: "#f59e0b" # Maritime amber trident icon container
      brand_badge_fg: "#0b1329" # Dark navy icon glyph
    text:
      primary: "#0f172a" # Slate-900: High-contrast primary headings, card titles, key metric numbers
      secondary: "#334155" # Slate-700: Body descriptions, readable observations, audit notes
      muted: "#64748b" # Slate-500: Uppercase technical labels, column headers, metadata keys
      disabled: "#94a3b8" # Slate-400: Inactive text or placeholder
      inverted: "#ffffff" # White text on dark buttons, badges, and sidebar
    borders:
      default: "#e2e8f0" # Slate-200: Subtle card perimeter and table dividers
      subtle: "#f1f5f9" # Slate-100: Secondary inner lines
      strong: "#cbd5e1" # Slate-300: Form inputs, active borders
      focus: "#0f172a" # Slate-900: Accessibility focus outline
    actions:
      primary_bg: "#0f172a" # Slate-900: Solid authoritative action button (Save, Export, Generate)
      primary_hover: "#1e293b" # Slate-800
      primary_text: "#ffffff"
      secondary_bg: "#ffffff" # White button with border
      secondary_border: "#cbd5e1" # Slate-300
      secondary_text: "#0f172a"
      secondary_hover: "#f8fafc"
    semantics:
      confirmed:
        badge_bg: "#ecfdf5" # Emerald-50
        badge_border: "#a7f3d0" # Emerald-200
        text: "#047857" # Emerald-700
        solid: "#10b981" # Emerald-500
      warning:
        badge_bg: "#fffbeb" # Amber-50
        badge_border: "#fde68a" # Amber-200
        text: "#b45309" # Amber-700
        solid: "#f59e0b" # Amber-500
      danger:
        badge_bg: "#fef2f2" # Red-50
        badge_border: "#fecaca" # Red-200
        text: "#b91c1c" # Red-700
        solid: "#ef4444" # Red-500
      info:
        badge_bg: "#f0f9ff" # Sky-50
        badge_border: "#bae6fd" # Sky-200
        text: "#0369a1" # Sky-700
        solid: "#0284c7" # Sky-600
    sonar_overlays:
      candidate_high: "#ef4444" # Red stroke: Prominent acoustic target with shadow deficit
      candidate_med: "#f59e0b" # Amber stroke: Acoustic anomaly proposal
      candidate_low: "#38bdf8" # Sky stroke: Subtle acoustic highlight
      candidate_selected: "#ffffff" # Thick white bounding stroke with focus halo
  typography:
    font_family_sans: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    font_family_mono: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"
    scales:
      display: { size: "28px", weight: "700", line_height: "36px", letter_spacing: "-0.02em" }
      page_title: { size: "20px", weight: "700", line_height: "28px", letter_spacing: "-0.01em" }
      section_title: { size: "14px", weight: "700", line_height: "20px", letter_spacing: "-0.005em" }
      card_title: { size: "12px", weight: "700", line_height: "16px", letter_spacing: "0.05em", transform: "uppercase" }
      metric_value: { size: "24px", weight: "700", line_height: "32px", letter_spacing: "-0.02em" }
      body: { size: "13px", weight: "400", line_height: "20px" }
      body_bold: { size: "13px", weight: "600", line_height: "20px" }
      caption: { size: "11px", weight: "500", line_height: "16px" }
      metadata_mono: { size: "12px", weight: "500", line_height: "16px", font: "mono" }
      badge_mono: { size: "10px", weight: "700", line_height: "14px", font: "mono" }
  spacing:
    xs: "4px"
    sm: "8px"
    md: "12px"
    lg: "16px"
    xl: "24px"
    "2xl": "32px"
  radius:
    none: "0px"
    sm: "4px"
    md: "6px"
    lg: "8px"
    xl: "12px"
    pill: "9999px"
  elevation:
    card: "0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05)"
    elevated: "0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05)"
    modal: "0 20px 25px -5px rgba(0, 0, 0, 0.15), 0 8px 10px -6px rgba(0, 0, 0, 0.1)"
---

# SONAR-INTEL Design System (DESIGN.md)

This document is the **single source of truth** for the visual design, information architecture, interface components, and styling conventions of the **SONAR-INTEL** platform.

All future visual modifications and frontend component refactoring must adhere strictly to the guidelines defined herein.

---

## 1. Overview & Design Direction

### 1.1 Product Persona & Aesthetic Target
SONAR-INTEL is an enterprise hydrographic data processing and AI anomaly candidate triage platform utilized by naval operators, hydrographic surveyors, maritime NGOs, and ocean researchers.

The interface must project:
- **Enterprise Credibility**: Clean, structured, trustworthy, and mission-critical.
- **Scientific & Cartographic Rigor**: Accurate metadata representation without embellishment.
- **Operator-Centric Ergonomics**: Low cognitive fatigue during multi-hour acoustic survey reviews.
- **Information Density with Visual Calm**: High data density organized via white-space, subtle borders, and clear typography rather than harsh outlines and saturated colors.

### 1.2 The Figma Paradigm (Light Workspace + Dark Maritime Navigation)
The attached Figma product design establishes a disciplined dual-surface paradigm:
1. **Persistent Maritime Navigation Rail (Left)**: Deep midnight navy (`#0b1329`), housing the brand badge, screen router, survey mission indicator, and operator session profile.
2. **Operational Application Canvas (Center & Content)**: Light neutral slate (`#f8fafc`), with crisp pure white card surfaces (`#ffffff`), subtle gray borders (`#e2e8f0`), and high-contrast dark slate text (`#0f172a`).
3. **Dedicated Acoustic Windows (Embedded)**: Deep obsidian canvas (`#070c18`) used **strictly and solely** for side-scan sonar waterfall imagery, acoustic candidate crops, and nautical map bathymetry.

```
+-----------------------------------------------------------------------------------------------+
| TOPBAR (h: 56px, bg: #ffffff, border-b: #e2e8f0)                                              |
| [Brand / Portal Badge]    [Global Search: anomalies, grids...]     [Demo Swath] [API Status]  |
+-------------------+---------------------------------------------------------------------------+
| SIDEBAR           | MAIN WORKSPACE CANVAS (bg: #f8fafc, p: 24px)                              |
| (w: 240px,        |                                                                           |
|  bg: #0b1329)     |  [PAGE HEADER: Title, Subtitle, Key Action]                               |
|                   |                                                                           |
|  * Dashboard      |  +-------------------+  +-------------------+  +-------------------+      |
|  * Sonar Analysis |  | STAT CARD (White) |  | STAT CARD (White) |  | STAT CARD (White) |      |
|  * Verification   |  +-------------------+  +-------------------+  +-------------------+      |
|  * GIS Mapping    |                                                                           |
|  * AI Pipeline    |  +-------------------------------------+  +----------------------------+  |
|  * Reports        |  | PRIMARY WORKSPACE / SONAR / MAP     |  | CONTEXTUAL / QUEUE PANEL   |  |
|                   |  | (Embedded dark acoustic view)       |  | (White card, list/detail)  |  |
|  [Operator Card]  |  +-------------------------------------+  +----------------------------+  |
+-------------------+---------------------------------------------------------------------------+
```

---

## 2. Color System & Design Tokens

### 2.1 Color Palette Matrix

| Token Role | Hex Code | Tailwind Equivalent | Usage Scope |
| :--- | :--- | :--- | :--- |
| `app.canvas` | `#f8fafc` | `bg-slate-50` | Primary application background for all 6 screens |
| `app.surface` | `#ffffff` | `bg-white` | Cards, panels, modal dialogs, data tables |
| `app.surface_muted` | `#f1f5f9` | `bg-slate-100` | Table headers, secondary control bars, well backgrounds |
| `app.surface_acoustic` | `#070c18` | `bg-[#070c18]` | Reserved strictly for side-scan waterfall and crops |
| `nav.background` | `#0b1329` | `bg-[#0b1329]` | Persistent left navigation rail |
| `nav.active_bg` | `#182649` | `bg-[#182649]` | Pill highlight for active screen item |
| `nav.border` | `#172342` | `border-[#172342]` | Subtle vertical divider between rail and canvas |
| `text.primary` | `#0f172a` | `text-slate-900` | Page titles, primary numbers, card headings |
| `text.secondary` | `#334155` | `text-slate-700` | Body text, readable notes, data values |
| `text.muted` | `#64748b` | `text-slate-500` | Metadata labels, column headers, units |
| `text.light` | `#f8fafc` | `text-slate-50` | Typography on dark sidebar or solid buttons |
| `border.default` | `#e2e8f0` | `border-slate-200` | Card perimeters, table cell bottom borders |
| `border.strong` | `#cbd5e1` | `border-slate-300` | Textarea, input borders, secondary button borders |
| `action.primary_bg` | `#0f172a` | `bg-slate-900` | Main buttons (*Save Notes*, *Export GIS Layer*) |
| `action.primary_hover` | `#1e293b` | `hover:bg-slate-800` | Primary button hover state |

### 2.2 Semantic & Triage Colors

Semantic colors must follow strict functional meaning. **Never use color decoratively.**

| Triage Status / Priority | Badge Background | Badge Border | Text Color | Meaning in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Confirmed Debris / Contact** | `#ecfdf5` (emerald-50) | `#a7f3d0` (emerald-200) | `#047857` (emerald-700) | Operator verified true anomaly / shipwreck |
| **Needs Review / Medium Priority** | `#fffbeb` (amber-50) | `#fde68a` (amber-200) | `#b45309` (amber-700) | Anomaly proposal requiring human scrutiny |
| **False Positive / High Priority** | `#fef2f2` (red-50) | `#fecaca` (red-200) | `#b91c1c` (red-700) | Rejected clutter OR immediate operator triage |
| **Active / Informational** | `#f0f9ff` (sky-50) | `#bae6fd` (sky-200) | `#0369a1` (sky-700) | Towfish tracking, system status, active stage |

### 2.3 Acoustic Overlays (Sonar Waterfall Canvas)
Bounding boxes overlaid on top of the dark sonar waterfall must be crisp, 2px wide, and never obscure the underlying acoustic shadow:
- **High Priority Candidate**: Solid stroke `#ef4444` (Red-500) with solid label box `#ef4444` below the target.
- **Medium Priority Candidate**: Solid stroke `#f59e0b` (Amber-500) with label box `#f59e0b`.
- **Low Priority Candidate**: Solid stroke `#38bdf8` (Sky-400) with label box `#38bdf8`.
- **Selected Candidate**: 2px solid white (`#ffffff`) bounding outline with a subtle 2px outer glow (`rgba(255, 255, 255, 0.4)`).

---

## 3. Typography System

### 3.1 Font Family Segregation Rule
To eliminate the "cyberpunk hacker terminal" look:
1. **Primary UI Typography (`font-sans`)**: Must be clean enterprise sans-serif (`Inter`, system UI font). Used for all page titles, section headings, card titles, navigation items, operator notes, buttons, and descriptive text.
2. **Selective Monospace (`font-mono`)**: Strictly isolated to numerical data, coordinates, bounding box tuples, model checkpoint paths, and telemetry strings.

```
CORRECT:
Page Title:      "Sonar Analysis Workspace"     -> font-sans (bold, slate-900)
Section Header:  "SURVEY DETAILS"               -> font-sans (uppercase, slate-500)
Candidate ID:    "C001"                         -> font-mono (bold, slate-900)
GPS Coordinates: "54° 12.428' N, 012° 08.194' E"-> font-mono (medium, slate-700)
Operator Note:   "Distinct acoustic shadow..."  -> font-sans (normal, slate-700)

INCORRECT:
Page Title in monospace:   "DASHBOARD OVERVIEW" -> PROHIBITED
Body text in monospace:    "Total surveys..."   -> PROHIBITED
Button text in monospace:  "SAVE & CONTINUE"    -> PROHIBITED
```

### 3.2 Type Scale

| Level | Size | Weight | Line Height | Tracking | Transform | Font Family |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Page Title** | 20px (`text-xl`) | Bold (700) | 28px | `-0.01em` | None | `font-sans` |
| **Section Title** | 14px (`text-sm`) | Bold (700) | 20px | Normal | None | `font-sans` |
| **Card Label** | 11px (`text-[11px]`) | SemiBold (600) | 16px | `+0.05em` | Uppercase | `font-sans` |
| **KPI Metric** | 24px (`text-2xl`) | Bold (700) | 32px | `-0.02em` | None | `font-sans` |
| **Body Regular**| 13px (`text-sm`) | Regular (400) | 20px | Normal | None | `font-sans` |
| **Body Medium** | 13px (`text-sm`) | Medium (500) | 20px | Normal | None | `font-sans` |
| **Data Monospace**| 12px (`text-xs`)| Medium (500) | 16px | Normal | None | `font-mono` |
| **Badge Label** | 10px (`text-[10px]`)| Bold (700) | 14px | `+0.02em` | Uppercase | `font-mono` |

---

## 4. Layout Architecture & Shell

### 4.1 Structural Dimensions
- **Sidebar Width**: Fixed `240px` (desktop), deep navy background `#0b1329`.
- **Topbar Height**: Fixed `56px`, white surface `#ffffff` with bottom border `#e2e8f0`.
- **Canvas Viewport**: Fluid height (`calc(100vh - 56px)`), scrollable padding `24px` (`p-6`).
- **Standard Grid**: 12-column CSS grid with `16px` (`gap-4`) or `24px` (`gap-6`) gutters.
- **Right Contextual Panels**: Width `320px` – `360px` when present.

### 4.2 Application Shell Wireframe
```
+----------------------------------------------------------------------------------------------------+
| TOPBAR (h: 56px)                                                                                    |
| [TRITON DETECT / GOVT / NGO PORTAL]      [Search anomalies or grids...]       [Demo Swath v] [API] |
+------------------+---------------------------------------------------------------------------------+
| SIDEBAR (240px)  | MAIN WORKSPACE (Light Canvas #f8fafc)                                           |
|                  |                                                                                 |
| * Dashboard      | 12-Column Responsive Layout                                                     |
| * Sonar Analysis | [Col 1-8 / 9: Primary Canvas or Table]   [Col 9-12: Contextual Sidebar/Drawer] |
| * Verification   |                                                                                 |
| * GIS Mapping    |                                                                                 |
| * AI Pipeline    |                                                                                 |
| * Reports        |                                                                                 |
|                  |                                                                                 |
| [Operator Badge] |                                                                                 |
+------------------+---------------------------------------------------------------------------------+
```

---

## 5. Elevation, Shapes & Borders

### 5.1 Card Elevation
Avoid harsh glowing shadows. Use clean, subtle drop shadows matching modern enterprise standards:
- **Default Card**: `box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);`
- **Elevated / Hover Card**: `box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05);`
- **Modal Dialog**: `box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15), 0 8px 10px -6px rgba(0, 0, 0, 0.1);`

### 5.2 Border Radius System
- **Cards & Major Panels**: `8px` (`rounded-lg`)
- **Inputs & Textareas**: `6px` (`rounded-md`)
- **Buttons**: `6px` (`rounded-md`)
- **Status Badges & Pills**: `9999px` (`rounded-full`) or `4px` (`rounded`)

---

## 6. Component Specifications

### 6.1 Persistent Navigation Rail (`Sidebar`)
- **Background**: Deep midnight navy `#0b1329`.
- **Top Brand Badge**: Rounded square (`32x32px`) in maritime amber `#f59e0b` with navy trident icon, followed by:
  - Title: **TRITON DETECT** (or **SONAR-INTEL**) in 13px Bold white (`#ffffff`).
  - Subtitle: **GOVT / NGO PORTAL** in 9px Regular muted slate (`#64748b`).
- **Navigation Links**:
  - Vertical list of 6 items: *Dashboard*, *Sonar Analysis*, *Contact Verification*, *GIS Mapping*, *AI Pipeline*, *Reports*.
  - Inactive: Slate-400 (`#94a3b8`) text and icon; hover becomes Slate-200 (`#e2e8f0`).
  - Active: Elevated navy pill background `#182649` with white text (`#ffffff`) and crisp white icon.
- **Bottom Operator Profile**:
  - Circular avatar (32px), user name "Dr. Clara Vance" (12px bold white), role "Senior Analyst" (10px slate-400).

### 6.2 Application Header (`Topbar`)
- **Background**: `#ffffff`, border-bottom `1px solid #e2e8f0`.
- **Center Search Input**:
  - Width `320px`, height `36px`, rounded `6px`.
  - Background `#f8fafc`, border `#e2e8f0`.
  - Placeholder: *"Search anomalies or grids..."* with search icon.
- **Right Action Bar**:
  - Mission Swath selector pill (e.g. `DEMO_VIATOR_04`).
  - Curated Demo Swath dropdown button with chevron.
  - API Health status pill (`Online` in soft green).

### 6.3 KPI Metric Stat Card (`StatCard`)
- **Surface**: Pure white `#ffffff`, border `1px solid #e2e8f0`, rounded `8px`, padding `16px`.
- **Layout**:
  - Row 1: Uppercase label in 11px SemiBold Slate-500 (e.g. `TOTAL SURVEYS`).
  - Row 2: Large primary metric in 24px Bold Slate-900 (e.g. `142`).
  - Row 3: Trend or category badge (e.g. `+12%` in green pill `#ecfdf5` or `YOLOv8n Candidates` in Slate-500).

### 6.4 Detection Queue List Item (`DetectionCard`)
- **Surface**: Pure white `#ffffff`, border `1px solid #e2e8f0`, rounded `6px`, padding `10px 12px`.
- **Layout**:
  - Left: 36x36px dark acoustic thumbnail preview.
  - Middle: Contact ID in 12px Bold (e.g. `DET-209`), with confidence badge (e.g. `92% YOLOv8n`).
  - Right: Chevron arrow (`→`) in Slate-400.
  - Hover: Subtle background shift to `#f8fafc` and border `#cbd5e1`.
  - Selected: Border `2px solid #0f172a` with soft elevation.

### 6.5 Acoustic Context Verification Bar (`AcousticCards`)
Positioned directly beneath the side-scan sonar image:
- Container: White card surface with `12px` padding and 4-column horizontal grid.
- Items:
  1. **Object-Shadow Analysis**: Highlight vs Shadow deficit (e.g. `SHADOW MATCHED`, badge: `PASS` in green).
  2. **Seabed Texture Match**: Dynamic range and backscatter floor (e.g. `SANDY / GRAVEL`, badge: `87%` in green).
  3. **False Positive Score**: Structural vs geological clutter (e.g. `ANOMALOUS STRUCTURE`, badge: `12/100` in red/amber).
  4. **Overall Confidence**: Composite score (e.g. `HI-RES MATCH`, badge: `83.0%`).

### 6.6 One-Click Operator Triage Buttons (`ReviewActions`)
Used in the Contact Verification screen:
- **Confirm Contact Button**:
  - Background: Soft emerald `#ecfdf5`
  - Border: `1.5px solid #10b981`
  - Text & Icon: `#047857` (Bold 12px)
  - Label: `✓ Confirm Contact`
- **False Positive Button**:
  - Background: Soft red `#fef2f2`
  - Border: `1.5px solid #ef4444`
  - Text & Icon: `#b91c1c` (Bold 12px)
  - Label: `✗ False Positive`
- **Needs Review Button**:
  - Background: Soft amber `#fffbeb`
  - Border: `1.5px solid #f59e0b`
  - Text & Icon: `#b45309` (Bold 12px)
  - Label: `? Needs Review`

### 6.7 Data Table (`DataTable`)
- **Container**: White surface `#ffffff`, border `1px solid #e2e8f0`, rounded `8px`.
- **Header**: Background `#f8fafc`, text 11px uppercase SemiBold Slate-500, padding `8px 16px`.
- **Row**: Height `40px`, padding `8px 16px`, border-bottom `1px solid #f1f5f9`.
- **Hover**: Background `#f8fafc`.

---

## 7. Screen-by-Screen Specifications

### 7.1 Screen 1: Dashboard Overview
- **Header**: Title *"Dashboard Overview"*, search bar on right.
- **Top Row (6 KPI Cards)**:
  1. `TOTAL SURVEYS`: Live database tally (e.g. `8`), badge `Ingested`.
  2. `TOTAL DETECTIONS`: Real candidate proposal count (e.g. `11`), badge `YOLOv8n Proposals`.
  3. `CONFIRMED DEBRIS`: Verified contacts count, badge `Operator Verified`.
  4. `HIGH PRIORITY`: Critical candidate count, badge `Immediate Triage`.
  5. `FALSE POSITIVES`: Rejected geological clutter count.
  6. `TRIAGE ASSIGNED`: Reviewed percentage.
- **Middle Row (Analytical Charts & Breakdown)**:
  - Left (Col 1–6): *Monthly / Survey Detection Trends* line chart on clean white card.
  - Middle (Col 7–9): *Debris & Anomaly Distribution* horizontal bar chart (High, Medium, Low).
  - Right (Col 10–12): *Survey Coverage Area (Sq Km)* vertical bar chart.
- **Bottom Row (Spatial & Audit Feed)**:
  - Left (Col 1–6): *Confirmed Contact Locations (Simplified Coastal Outline)* white card with coastal vector line and colored contact dots.
  - Right (Col 7–12): *Recent Platform Activity* data table with Time, Event, Description, and Status pill.

### 7.2 Screen 2: Sonar Analysis Workspace (Core Experience)
- **Visual Hierarchy**:
  ```
  Survey Details -> Sonar Image Canvas -> Detection Queue -> Selected Anomaly -> Acoustic Evidence
  ```
- **3-Column Architecture**:
  - **Left Column (w: 240px)**: White metadata card containing:
    - *Survey Details*: Survey ID, Date, Source Vessel, Operator.
    - *Quality Metrics*: Dynamic range (18.4 dB), Resolution (15 cm/px), Quality score.
    - *Navigation Metadata*: Heading, Speed, Towfish depth, Line spacing (or *"Unavailable"*).
  - **Center Column (Fluid Workspace)**:
    - Clean top toolbar on white/muted surface: Zoom controls (`-`, `+`), Contrast slider (`50% - 180%`), Normalization toggle (`RAW` vs `1–99% NORMALIZED`), Overlay toggle (`Show Boxes`).
    - Embedded dark acoustic canvas (`#070c18`) displaying the high-resolution side-scan waterfall with candidate bounding boxes.
    - Bottom Acoustic Evidence Bar: 4 diagnostic cards (*Shadow Matched*, *Texture Match*, *False Positive Risk*, *Overall Score*).
  - **Right Column (w: 280px)**:
    - White card header *"DETECTION QUEUE"* with count badge.
    - Scrollable vertical stack of detection cards sorted by rank score (`C001, C002, ...`).

### 7.3 Screen 3: Contact Verification Workflow
- **Header**: *"Contact Verification Workflow"* with active candidate selector dropdown.
- **Left Column (Col 1–7)**:
  - White card *"Acoustic Detection Crop (Candidate ID)"*.
  - Embedded dark viewport displaying the zoomed target highlight and down-range shadow void.
  - Telemetry card below crop:
    - `CONFIDENCE SCORE`: e.g. `83% (High Confidence)`
    - `SPATIAL COORDINATES`: Real estimated Lat/Lon OR *"Spatial coordinates unavailable"*.
    - `ESTIMATED TARGET SIZE`: Width &times; Height in pixels.
    - `DISTANCE FROM TRACKLINE`: Along-track offset.
    - `CURRENT PRIORITY`: High / Medium / Low badge.
- **Right Column (Col 8–12)**:
  - White card *"ONE-CLICK OPERATOR WORKFLOW"* with the 3 large action buttons (`[Confirm Contact]`, `[False Positive]`, `[Needs Review]`).
  - White card *"TARGET OPERATOR NOTES"* with clean textarea and `[ Save & Continue ]` button.
  - White card *"VERIFICATION HISTORY LOG"* tracking past timestamps and reviewer decisions.

### 7.4 Screen 4: GIS Mapping & Cleanup Planning
- **Header**: *"GIS Mapping & Cleanup Planning"*, top map control bar with *Layers*, *Measure Distance*, and dark button `[ Export GIS Layer ]`.
- **Main Map Canvas (Col 1–8)**:
  - Full MapLibre GL canvas with nautical bathymetric tile styling.
  - Candidate pins color-coded by priority (Red = High, Amber = Medium, Sky = Low).
  - Bottom legend card: Confirmed Debris (red dot), Ghost Nets (amber dot), Survey Tracks (dashed line).
- **Right Contextual Drawer (Col 9–12)**:
  - *GIS Filters*: Pill buttons `[ All Areas ]` (active dark navy) and `[ High Priority Only ]`.
  - *Selected Contact Card*: Contact ID, class name, confidence, coordinates.
  - *Spatial Resolution & Datum Card*: WGS 84 (EPSG:4326), along-track dead-reckoning provenance note.

### 7.5 Screen 5: AI Deep Learning Pipeline Monitor
- **Header**: *"AI Deep Learning Pipeline Monitor"*, model provenance pill `yolov8n-sonar-baseline`.
- **Active Edge AI Pipeline Architecture Card**:
  - Horizontal flowchart showing 8 stages: *Raw Ingest*, *Quality SNR*, *1-99% Normalization*, *640x640 Tiling*, *YOLOv8n GPU*, *NMS & Ranking*, *Operator Triage*, *GIS & Export*.
  - Status badges: `COMPLETE` (emerald), `ACTIVE` (sky), `PENDING` (slate).
- **4 KPI Metric Cards (Real Measured Baseline Values)**:
  1. `VALIDATION mAP@50`: **6.45%** (Measured on 1,256 validation tiles across 55 sites).
  2. `FROZEN TEST mAP@50`: **10.48%** (Measured on 1,256 held-out test tiles across 46 sites).
  3. `TEST PRECISION / RECALL`: **18.9% / 12.9%** (Pre-human triage proposal mode).
  4. `AVERAGE PROCESSING SPEED`: **18.7 ms / tile (52.3 FPS)** on RTX 3050 GPU.
- **Bottom Two Panels**:
  - Left: *Active Inference Pipeline Log* table with Timestamp, Filename, Stage, Status, Speed.
  - Right: *Core Convolutional Models* specs card with architecture parameters (3.01M params, 8.2 GFLOPs).

### 7.6 Screen 6: Reports & Export Central
- **Header**: *"Reports & Export Central"*, active survey reference indicator.
- **Top Row (4 Export Cards on White Surfaces)**:
  1. *Full Survey Hydrographic Report*: Narrative summary document, button `[ Generate Report ]`.
  2. *Spatial GeoJSON*: RFC 7946 compliant FeatureCollection of Point geometries, button `[ Export GeoJSON ]`.
  3. *Tabular Detections CSV*: Spreadsheet of IDs, bounding boxes, scores, and review statuses, button `[ Download CSV ]`.
  4. *Baseline Model Card*: Model documentation for `yolov8n-sonar-baseline`, button `[ Open Model Card ]`.
- **Bottom Row**:
  - Left: *Consolidated Triage Audit Totals* card with confirmed count, false positive count, and operator triage rate.
  - Right: *Recent Exports Log* table tracking recent generated artifacts and download timestamps.

---

## 8. Data Honesty & Scientific Positioning Rules

### 8.1 Zero-Fabrication Mandate
The UI must **never** fabricate operational metrics or spatial coordinates:
1. **Coordinates**: If source imagery lacks a synchronized navigation log (`has_navigation == false`), display:
   `Spatial coordinates unavailable (Awaiting towfish nav log)`
   Never invent random GPS numbers.
2. **Model Metrics**: Display real measured results (**Val mAP: 6.45%**, **Test mAP: 10.48%**). Never replace them with synthetic "94%" numbers from the Figma mockup.
3. **Dimensions & Depths**: If target dimensions are in pixels, report `198 × 539 px`. Only report physical meters if slant-range altitude is known.

### 8.2 AI Candidate vs Confirmed Contact Pipeline
The interface must visually reinforce that YOLOv8n is an **AI anomaly candidate generator**, not an autonomous shipwreck classifier:
```
[Raw Side-Scan Swath] -> [AI Candidate Proposed] -> [Operator Triage] -> [Confirmed Anomaly]
```
Terminology must consistently use:
- `AI Candidate` / `Anomaly Proposal` (prior to human review)
- `Confirmed Contact` / `False Positive` (after human review)

---

## 9. Do's and Don'ts (Anti-Pattern Matrix)

| Category | DO (Authoritative Design System) | DON'T (Prohibited Anti-Patterns) |
| :--- | :--- | :--- |
| **Workspace Canvas** | Light neutral background (`#f8fafc`) with white card surfaces (`#ffffff`). | Pitch-black or dark-navy background covering the whole screen. |
| **Acoustic Imagery** | Embed dark canvas (`#070c18`) **strictly inside** the sonar viewer and crop tools. | Turn the entire page, sidebars, and forms into a dark acoustic viewer. |
| **Color Usage** | Reserve cyan/teal as a restrained interaction accent; use dark slate for primary text. | Drown the entire UI in glowing neon cyan borders, cyan text, and cyan icons. |
| **Typography** | Use crisp enterprise sans-serif (`Inter`) for all UI headings, labels, cards, and buttons. | Apply monospace font to the whole page like a hacker terminal. |
| **Monospace Scope** | Isolate monospace strictly to candidate IDs (`C001`), GPS coordinates, and raw telemetry. | Put page titles, buttons, and audit trail sentences in monospace. |
| **Borders & Shadows**| Use subtle `1px solid #e2e8f0` borders and soft `0 1px 3px rgba(0,0,0,0.05)` elevation. | Add glowing neon cyan box-shadows (`box-shadow: 0 0 12px #06b6d4`). |
| **Buttons** | Solid dark slate `#0f172a` for primary actions; soft tinted pills for triage actions. | Neon gradient buttons with animated glowing rings. |
| **Detection Boxes** | Clean 2px colored outlines with compact ID badge below the target. | Giant translucent colored overlays that obscure the acoustic shadow. |
| **Metrics** | Display verified measured benchmarks (Val mAP: 6.45%, Test mAP: 10.48%). | Fabricate 90%+ metrics to match the fictional Figma placeholders. |
| **Telemetry** | Display `"Spatial coordinates unavailable"` when no navigation log is provided. | Manufacture synthetic coordinates or fake depth readings. |
