# SONAR-INTEL: Visual Design Audit & Gap Analysis

**Date:** September 1, 2026  
**Document:** `DESIGN_AUDIT.md`  
**Reference Document:** [`DESIGN.md`](file:///c:/Users/Asus/Desktop/SONAR-INTEL/DESIGN.md)  
**Status:** Audit Complete — Implementation Ready  

---

## 1. Executive Summary

This document audits the visual drift between the reference Figma product designs and the current working implementation of **SONAR-INTEL**. While the platform is **functionally verified** (with working real-time inference, candidate ranking, database persistence, MapLibre GIS integration, and export capabilities), its **visual presentation has drifted into an overly dark, terminal-style aesthetic** with excessive glowing cyan accents, pitch-black cards, and ubiquitous monospace typography.

This audit details:
1. What was extracted from the reference Figma designs.
2. The exact architectural and aesthetic flaws of the current UI.
3. The proposed enterprise design system formalized in `DESIGN.md`.
4. The file-by-file and component-by-component changes required.
5. Technical implementation risks and mitigation strategies.

---

## 2. What Was Extracted from the Figma References

Careful inspection of the 6 Figma reference screenshots reveals an **enterprise hydrographic operations platform**, not a sci-fi hacker terminal:

| Design Dimension | Figma Reference Design Language | Current Implementation Flaw |
| :--- | :--- | :--- |
| **Workspace Canvas** | **Light neutral workspace (`#f8fafc`)** with crisp white cards (`#ffffff`) and subtle gray borders (`#e2e8f0`). | Pitch-black / dark-navy background (`#08101d` and `#070e1a`) spanning the entire application. |
| **Navigation Rail** | **Deep midnight navy (`#0b1329`)** persistent rail with amber brand badge (`#f59e0b`), muted icons, and active pill state (`#182649`). | Dark rail, but with harsh cyan borders, glowing highlights, and terminal-style labels. |
| **Acoustic Imagery** | **Embedded dark viewport (`#070c18`)** used *strictly and solely* for sonar waterfall and zoomed target crops. | Everything (tables, textareas, forms, sidebars) is rendered in dark acoustic style. |
| **Typography** | **Clean enterprise sans-serif (`Inter`)** for headings, card titles, labels, tables, and notes. Monospace is used *strictly* for technical IDs and GPS strings. | Monospace (`font-mono`) applied globally across nearly all headers, body text, buttons, and badges. |
| **Color Restraint** | **Subtle, functional palette**: Dark slate primary text (`#0f172a`), muted slate metadata (`#64748b`), and semantic green/amber/red badges. | Dominant glowing neon cyan (`#06b6d4`, `#38bdf8`) applied to borders, text, shadows, and icons everywhere. |
| **Action Buttons** | **Solid dark slate (`#0f172a`)** for primary actions; clean soft-tinted pills (`#ecfdf5`, `#fef2f2`, `#fffbeb`) for triage decisions. | Glowing cyan neon buttons (`shadow-[0_0_12px_rgba(6,182,212,0.3)]`) with harsh high-contrast borders. |
| **Information Density** | High data density achieved through **whitespace, subtle 1px dividers, and crisp hierarchy** rather than box borders. | Heavy-handed borders on every element, creating visual clutter and fatigue. |

---

## 3. Detailed Gap Analysis by Operational Screen

### Screen 1: Dashboard Overview
- **Figma Architecture**: Light gray canvas, 6 pure white stat cards with subtle borders, monthly detection line chart on white card, horizontal debris distribution bar chart, simplified coastal outline with colored dots, recent platform activity table with white background and soft status pills.
- **Current UI Issues**: Dark blue background with neon cyan text, glowing cyan card borders, dark chart containers, glowing cyan KPI numbers.
- **Remediation**: Convert canvas to `#f8fafc`, cards to `#ffffff` with `#e2e8f0` borders, primary metric numbers to Slate-900 (`#0f172a`), and activity table to clean enterprise white surface.

### Screen 2: Sonar Analysis Workspace
- **Figma Architecture**: 3-column operational layout. Left column: white metadata cards (*Survey Details*, *Quality Metrics*, *Navigation Metadata*). Center column: clean top toolbar (Zoom, Contrast, Normalization toggle, Detection toggle), embedded dark sonar waterfall canvas with crisp 2px colored candidate bounding boxes, bottom 4-card acoustic evidence bar. Right column: white detection queue card with vertical candidate cards (dark preview thumbnail, candidate ID, confidence pill, right arrow).
- **Current UI Issues**: Center column has harsh cyan headers; bottom acoustic bar is dark navy with glowing icons; right queue items have heavy dark-blue styling.
- **Remediation**: Wrap workspace in light neutral canvas; maintain dark acoustic canvas *only* for the sonar waterfall; convert left and right panels into crisp white cards; style detection queue items with white surfaces, dark preview thumbnails, and subtle hover states.

### Screen 3: Contact Verification Workflow
- **Figma Architecture**: Left column: white card housing the dark acoustic crop viewport, with a structured telemetry table below (Confidence Score, GPS Coordinates, Target Bounds, Distance from Trackline, Priority). Right column: white card with 3 large one-click triage buttons (*Confirm Debris* in soft green, *False Positive* in soft red, *Needs Review* in soft amber), clean operator notes textarea with dark solid *[Save & Continue]* button, and verification audit log table.
- **Current UI Issues**: Entire screen is dark blue; triage buttons use dark background with glowing borders; telemetry uses cyan labels.
- **Remediation**: Render the crop inside an embedded obsidian container (`#070c18`); make the surrounding card white; style triage buttons with soft pastel backgrounds (`#ecfdf5`, `#fef2f2`, `#fffbeb`) and clean colored borders; use solid dark slate `#0f172a` for the save button.

### Screen 4: GIS Mapping & Cleanup Planning
- **Figma Architecture**: Full-height MapLibre bathymetric canvas, top map bar with *Layers*, *Measure Distance*, and dark solid button *[Export GIS Layer]*. Right drawer: clean white cards for *GIS Filters* (pill buttons `[All Areas]` and `[High Priority Only]`), *Selected Contact Card*, and *Spatial Resolution & Datum*.
- **Current UI Issues**: Right drawer is dark blue with cyan borders; map controls use dark cyan styling.
- **Remediation**: Style right drawer with white cards on light background; style map controls cleanly; use dark solid slate button for GeoJSON export.

### Screen 5: AI Pipeline & Inference Monitor
- **Figma Architecture**: Light canvas, top card *Active Edge AI Pipeline Architecture* with 8 clean stage boxes (green, yellow, gray borders), 4 white stat cards for real measured metrics (mAP50, Speed, Total Parsed, Active Queue), bottom active inference log table on white surface, and core models specifications card.
- **Current UI Issues**: Flowchart is dark navy with glowing cyan borders; stat cards are dark navy with cyan numbers; log table is dark with high contrast cyan text.
- **Remediation**: Convert pipeline stages into clean white cards with colored semantic status tags; make KPI cards white with bold Slate-900 numbers; render execution log on clean white table with alternating row hover.

### Screen 6: Reports & Export Central
- **Figma Architecture**: 4 top export cards on white surfaces (*Full Survey Report*, *Spatial GeoJSON*, *Tabular Detections CSV*, *Baseline Model Card*), each with an icon, description, and action button (dark slate or white outline). Bottom row: consolidated triage totals card and recent exports log table.
- **Current UI Issues**: Export cards are dark navy with cyan borders and glowing buttons.
- **Remediation**: Convert export cards to pure white with subtle gray borders; primary export buttons styled in solid dark slate `#0f172a`.

---

## 4. Component Refactoring Roadmap

| Component | Target File | Current Styling Problem | Proposed Authoritative Refactoring |
| :--- | :--- | :--- | :--- |
| **Main Shell** | `src/components/layout/MainLayout.tsx` | Dark background `#050b14` applied to main container. | Split shell: Dark sidebar `#0b1329` + Light canvas `#f8fafc` for main container. |
| **Header** | `src/components/layout/Header.tsx` | Dark `#070e1a` with glowing cyan badges. | White surface `#ffffff`, border `#e2e8f0`, search input `#f8fafc`, dark slate typography. |
| **Sidebar** | `src/components/layout/Sidebar.tsx` | Dark background with cyan text and neon search box. | Retain dark navy `#0b1329`, but refine typography to clean sans-serif; active state as `#182649` pill; amber brand badge. |
| **Dashboard** | `src/pages/DashboardPage.tsx` | Dark navy cards `#0b1626` and cyan typography. | White cards `#ffffff`, Slate-900 metric values, enterprise sans-serif typography, clean gray borders `#e2e8f0`. |
| **Sonar Analysis** | `src/pages/SonarAnalysisPage.tsx` | All 3 columns dark with cyan accents. | Left & right columns on white cards; center canvas embeds dark waterfall `#070c18`; white detection queue cards. |
| **Verification** | `src/pages/ContactVerificationPage.tsx` | Dark layout with dark glowing triage buttons. | White cards; embedded dark crop; soft tinted triage buttons (`#ecfdf5`, `#fef2f2`, `#fffbeb`); solid dark slate save button. |
| **GIS Mapping** | `src/pages/GisMappingPage.tsx` | Dark drawer with cyan badges. | White drawer panels; dark solid action buttons; clean pill filters. |
| **AI Pipeline** | `src/pages/AiPipelinePage.tsx` | Dark flowchart and dark metric cards. | Clean white stage cards; white metric cards; verified metrics (6.45% val, 10.48% test) in bold Slate-900. |
| **Reports** | `src/pages/ReportsPage.tsx` | Dark navy export cards with cyan buttons. | White export cards; dark slate primary action buttons; clean data tables. |

---

## 5. Technical Implementation Risks & Mitigation

1. **Risk: Breaking Sonar Waterfall Rendering or Contrast Filtering**
   - *Mitigation*: The central canvas element displaying the side-scan sonar image must retain its dedicated dark background (`#070c18` or black) and existing CSS contrast filter. Only the *surrounding* page and toolbars transition to the light palette.
2. **Risk: MapLibre Inverted Styling Conflicts**
   - *Mitigation*: MapLibre GL is already styled with bathymetric nautical tiles. It will be mounted seamlessly inside the white card wrapper without affecting tile layers.
3. **Risk: Regression in Existing Real-Time Triage Logic**
   - *Mitigation*: The refactoring is purely visual (JSX class names, CSS tokens, and layout wrappers). All React hooks (`useSurvey`), API service calls, and backend contracts remain 100% intact.
4. **Risk: Typography Replacement Overhead**
   - *Mitigation*: Replace global `font-mono` with `font-sans` across pages, keeping `font-mono` strictly on candidate IDs (`C001`), coordinates (`54° 12.428' N`), and telemetry values.

---

## 6. Verification Plan for Next Phase

Once approved to begin frontend modifications:
1. Verify `npm run build` succeeds with zero TypeScript errors after every component update.
2. Launch browser subagent to capture high-resolution screenshots of all 6 screens.
3. Cross-compare each screen against the corresponding Figma screenshot to confirm 100% fidelity to the enterprise visual standard.
