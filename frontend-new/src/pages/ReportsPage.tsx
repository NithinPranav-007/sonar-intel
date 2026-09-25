import React, { useState } from 'react';
import { Contact, SurveyUploadResponse } from '../types/detection';
import { 
  FileText, 
  Download, 
  CheckCircle2, 
  Table, 
  FileSpreadsheet, 
  Globe, 
  FileCheck,
  Sparkles,
  Layers,
  ArrowRight,
  ArrowLeft,
  ChevronLeft,
  Eye,
  X,
  Copy,
  ExternalLink
} from 'lucide-react';

interface ReportsPageProps {
  survey: SurveyUploadResponse | null;
  contacts: Contact[];
  onNavigateToDashboard?: () => void;
  onSelectContact?: (contact: Contact) => void;
  onNavigateToVerify?: () => void;
}

export const ReportsPage: React.FC<ReportsPageProps> = ({ 
  survey, 
  contacts,
  onNavigateToDashboard,
  onSelectContact,
  onNavigateToVerify
}) => {
  const [downloadMsg, setDownloadMsg] = useState<string | null>(null);
  const [previewModal, setPreviewModal] = useState<'csv' | 'geojson' | 'summary' | null>(null);
  const [copyNotice, setCopyNotice] = useState<string | null>(null);

  const handleExportCSV = () => {
    if (!survey) return;
    window.open(`/api/surveys/${survey.survey_id}/csv`, '_blank');
    setDownloadMsg('Generated detection CSV export successfully.');
  };

  const handleExportGeoJSON = () => {
    if (!survey) return;
    window.open(`/api/surveys/${survey.survey_id}/geojson`, '_blank');
    setDownloadMsg('Generated spatial GeoJSON export successfully.');
  };

  const handleExportSummary = () => {
    if (!survey) return;
    window.open(`/api/surveys/${survey.survey_id}/summary`, '_blank');
    setDownloadMsg('Generated hydrographic executive summary report.');
  };

  const handleRowClick = (contact: Contact) => {
    if (onSelectContact) onSelectContact(contact);
    if (onNavigateToVerify) onNavigateToVerify();
  };

  const handleCopyPreview = (content: string) => {
    navigator.clipboard.writeText(content);
    setCopyNotice('Copied preview content to clipboard!');
    setTimeout(() => setCopyNotice(null), 2500);
  };

  // Sample representations for preview modals
  const csvSample = `contact_id,priority,confidence,class_name,review_status,latitude,longitude,bbox_x1,bbox_y1,bbox_x2,bbox_y2\n` +
    contacts.slice(0, 5).map(c => 
      `${c.contact_id},${c.priority},${c.confidence.toFixed(2)},${c.class_name},${c.review_status},${c.latitude?.toFixed(6) ?? '13.086396'},${c.longitude?.toFixed(6) ?? '80.383111'},${c.bbox.x1},${c.bbox.y1},${c.bbox.x2},${c.bbox.y2}`
    ).join('\n');

  const geoJsonSample = JSON.stringify({
    type: "FeatureCollection",
    crs: { type: "name", properties: { name: "urn:ogc:def:crs:OGC:1.3:CRS84" } },
    features: contacts.slice(0, 3).map(c => ({
      type: "Feature",
      geometry: {
        type: "Point",
        coordinates: [c.longitude ?? 80.383111, c.latitude ?? 13.086396, 0.0]
      },
      properties: {
        contact_id: c.contact_id,
        survey_id: survey?.survey_id || "SURV_VIATOR_04_BENCHMARK",
        priority: c.priority,
        confidence: c.confidence,
        class_name: c.class_name,
        review_status: c.review_status,
        model_version: c.model_version || "Acoustic-YOLOv8s-v1.0"
      }
    }))
  }, null, 2);

  const summarySample = JSON.stringify({
    survey_id: survey?.survey_id || "SURV_VIATOR_04_BENCHMARK",
    filename: survey?.filename || "viator_04_benchmark.png",
    timestamp: new Date().toISOString(),
    geodetic_datum: "WGS-84 (EPSG:4326)",
    total_detections: contacts.length,
    high_priority_count: contacts.filter(c => c.priority === 'HIGH').length,
    medium_priority_count: contacts.filter(c => c.priority === 'MEDIUM').length,
    low_priority_count: contacts.filter(c => c.priority === 'LOW').length,
    confirmed_targets: contacts.filter(c => c.review_status === 'CONFIRMED').length,
    pipeline_model: "Acoustic-YOLOv8s + SSS-Net Fusion",
    operator_triage_resolution_rate: "100%"
  }, null, 2);

  return (
    <div className="p-6 lg:p-8 max-w-[1600px] mx-auto space-y-8 font-sans">
      
      {/* 1. Header with Top-Left Back Button & Breadcrumbs */}
      <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        <div className="space-y-2">
          {/* Top-Left: Back Button & Breadcrumb */}
          <div className="flex items-center gap-3">
            {onNavigateToDashboard && (
              <button
                onClick={onNavigateToDashboard}
                className="px-3.5 py-1.5 rounded-xl bg-[#f8fafc] hover:bg-slate-100 text-[#0f172a] hover:text-[#1d4ed8] border border-[#e2e8f0] font-semibold text-xs transition-all duration-200 shadow-tactile flex items-center gap-1.5 cursor-pointer group"
              >
                <ChevronLeft className="w-3.5 h-3.5 text-[#64748b] group-hover:text-[#1d4ed8] group-hover:-translate-x-0.5 transition-transform" />
                <span>Back to Dashboard Overview</span>
              </button>
            )}

            <div className="flex items-center gap-2 text-xs font-semibold">
              <button
                onClick={onNavigateToDashboard}
                className="text-[#64748b] hover:text-[#1d4ed8] hover:underline cursor-pointer transition-colors"
              >
                Dashboard Overview
              </button>
              <span className="text-[#cbd5e1]">&gt;</span>
              <span className="text-[#1d4ed8] font-bold">
                Reports & Export
              </span>
            </div>
          </div>

          <h2 className="text-2xl font-extrabold text-[#0f172a] font-display flex items-center gap-2.5">
            <FileText className="w-6 h-6 text-[#1d4ed8]" />
            Reports & Export Central
          </h2>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold px-3.5 py-1.5 rounded-full bg-[#f8fafc] border border-[#e2e8f0] text-[#0f172a] shadow-tactile">
            Active Swath: <strong className="font-mono text-[#1d4ed8]">{survey?.filename || 'Viator-04'}</strong>
          </span>
          <span className="text-xs font-bold px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-100">
            {contacts.length} Records Ready
          </span>
        </div>
      </div>

      {/* Success Download Banner */}
      {downloadMsg && (
        <div className="p-4 rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold flex items-center justify-between shadow-soft">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            <span>{downloadMsg}</span>
          </div>
          <button 
            onClick={() => setDownloadMsg(null)} 
            className="text-emerald-700 hover:text-emerald-900 font-bold underline cursor-pointer text-[11px]"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* 2. Three Core Hydrographic Data Products with Preview Buttons */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Product 1: Tabular CSV */}
        <div className="group relative overflow-hidden bg-white rounded-[24px] border border-[#e2e8f0] p-7 shadow-soft hover:shadow-card-hover transition-all duration-300 flex flex-col justify-between space-y-6">
          <div className="absolute -right-8 -bottom-8 w-36 h-36 bg-[#1d4ed8]/5 rounded-full blur-3xl pointer-events-none transition-all duration-500 group-hover:bg-[#1d4ed8]/15" />

          <div className="relative z-10 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-50 border border-blue-100 text-[#1d4ed8] transition-all duration-300 group-hover:bg-[#1d4ed8] group-hover:text-white group-hover:border-[#1d4ed8] shadow-xs">
                <FileSpreadsheet className="w-5 h-5" />
              </div>
              <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-slate-100 text-[#64748b]">
                .CSV Spreadsheet
              </span>
            </div>

            <div>
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-[#0f172a] font-display">Tabular Detections CSV</h3>
              </div>
              <p className="text-xs text-[#64748b] leading-relaxed mt-1">
                Full spreadsheet export with Candidate IDs, pixel bounds, AI confidences, acoustic evidence scores, and review status logs.
              </p>
            </div>

            {/* Preview Link */}
            <div className="pt-1">
              <button
                onClick={() => setPreviewModal('csv')}
                className="text-xs font-bold text-[#1d4ed8] hover:text-[#1e40af] hover:underline flex items-center gap-1.5 cursor-pointer py-1 px-2.5 rounded-lg bg-blue-50/60 hover:bg-blue-100/70 transition-colors w-fit"
              >
                <Eye className="w-3.5 h-3.5" />
                <span>Preview CSV Schema</span>
              </button>
            </div>
          </div>

          <button
            onClick={handleExportCSV}
            disabled={!survey}
            className="relative z-10 w-full py-3 rounded-full bg-[#1d4ed8] hover:bg-[#1e40af] text-white font-semibold text-xs transition-all duration-200 shadow-tactile flex items-center justify-center gap-2 cursor-pointer shadow-blue-glow disabled:opacity-50"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download Detections CSV</span>
          </button>
        </div>

        {/* Product 2: Spatial GeoJSON */}
        <div className="group relative overflow-hidden bg-white rounded-[24px] border border-[#e2e8f0] p-7 shadow-soft hover:shadow-card-hover transition-all duration-300 flex flex-col justify-between space-y-6">
          <div className="absolute -right-8 -bottom-8 w-36 h-36 bg-emerald-500/5 rounded-full blur-3xl pointer-events-none transition-all duration-500 group-hover:bg-emerald-500/15" />

          <div className="relative z-10 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] text-emerald-600 transition-all duration-300 group-hover:bg-emerald-600 group-hover:text-white group-hover:border-emerald-600">
                <Globe className="w-5 h-5" />
              </div>
              <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700">
                RFC 7946 Standard
              </span>
            </div>

            <div>
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-[#0f172a] font-display">Spatial RFC 7946 GeoJSON</h3>
              </div>
              <p className="text-xs text-[#64748b] leading-relaxed mt-1">
                Standardized FeatureCollection of Point geometries ready for immediate drag-and-drop ingestion into QGIS, ArcGIS, or MapStore.
              </p>
            </div>

            {/* Preview Link */}
            <div className="pt-1">
              <button
                onClick={() => setPreviewModal('geojson')}
                className="text-xs font-bold text-emerald-700 hover:text-emerald-900 hover:underline flex items-center gap-1.5 cursor-pointer py-1 px-2.5 rounded-lg bg-emerald-50 hover:bg-emerald-100/70 transition-colors w-fit"
              >
                <Eye className="w-3.5 h-3.5" />
                <span>Preview GeoJSON Schema</span>
              </button>
            </div>
          </div>

          <button
            onClick={handleExportGeoJSON}
            disabled={!survey}
            className="relative z-10 w-full py-3 rounded-full bg-[#0f172a] hover:bg-slate-800 text-white font-semibold text-xs transition-all duration-200 shadow-tactile flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download Spatial GeoJSON</span>
          </button>
        </div>

        {/* Product 3: Executive Summary */}
        <div className="group relative overflow-hidden bg-white rounded-[24px] border border-[#e2e8f0] p-7 shadow-soft hover:shadow-card-hover transition-all duration-300 flex flex-col justify-between space-y-6">
          <div className="absolute -right-8 -bottom-8 w-36 h-36 bg-indigo-600/5 rounded-full blur-3xl pointer-events-none transition-all duration-500 group-hover:bg-indigo-600/15" />

          <div className="relative z-10 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-indigo-50 border border-indigo-100 text-indigo-600 transition-all duration-300 group-hover:bg-indigo-600 group-hover:text-white group-hover:border-indigo-600 shadow-xs">
                <FileCheck className="w-5 h-5" />
              </div>
              <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-100">
                Audit Summary
              </span>
            </div>

            <div>
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-[#0f172a] font-display">Executive Survey Summary</h3>
              </div>
              <p className="text-xs text-[#64748b] leading-relaxed mt-1">
                Structured hydrographic report covering swath coverage, dynamic range, candidate counts, and operator triage resolution rates.
              </p>
            </div>

            {/* Preview Link */}
            <div className="pt-1">
              <button
                onClick={() => setPreviewModal('summary')}
                className="text-xs font-bold text-indigo-700 hover:text-indigo-900 hover:underline flex items-center gap-1.5 cursor-pointer py-1 px-2.5 rounded-lg bg-indigo-50/80 hover:bg-indigo-100/90 transition-colors w-fit border border-indigo-100"
              >
                <Eye className="w-3.5 h-3.5" />
                <span>Preview Summary JSON</span>
              </button>
            </div>
          </div>

          <button
            onClick={handleExportSummary}
            disabled={!survey}
            className="relative z-10 w-full py-3 rounded-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-xs transition-all duration-200 shadow-tactile shadow-[0_4px_14px_-2px_rgba(79,70,229,0.35)] flex items-center justify-center gap-2 cursor-pointer hover:scale-[1.01] active:scale-[0.99] disabled:opacity-50"
          >
            <Download className="w-3.5 h-3.5 text-white" />
            <span>Download Summary JSON</span>
          </button>
        </div>

      </section>

      {/* 3. Live Data Preview Table Card (Clickable Rows to Contact Triage) */}
      <section className="bg-white rounded-[24px] border border-[#e2e8f0] p-7 shadow-soft space-y-5">
        <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
          <div>
            <span className="section-label block">Live Export Preview</span>
            <h3 className="text-base font-bold text-[#0f172a] font-display mt-0.5 flex items-center gap-2">
              <Table className="w-4 h-4 text-[#1d4ed8]" />
              Survey Detections Export Table ({contacts.length} Records — Click row to verify)
            </h3>
          </div>
          <span className="text-xs font-semibold text-[#64748b]">
            Format: RFC 4180 CSV
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-sans">
            <thead>
              <tr className="border-b border-[#e2e8f0] text-[#64748b] text-[10px] uppercase font-bold tracking-wider">
                <th className="pb-3 pl-2">Contact ID</th>
                <th className="pb-3">Priority</th>
                <th className="pb-3">Confidence</th>
                <th className="pb-3">Bounding Box</th>
                <th className="pb-3">Latitude / Longitude</th>
                <th className="pb-3">Triage Status</th>
                <th className="pb-3 text-right pr-2">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {contacts.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-xs text-[#64748b]">
                    No contacts loaded for preview. Select a benchmark swath from the top header.
                  </td>
                </tr>
              ) : (
                contacts.map((c) => {
                  const isHigh = c.priority === 'HIGH';
                  const isMedium = c.priority === 'MEDIUM';
                  const isConfirmed = c.review_status === 'CONFIRMED';
                  const isFalseAlarm = c.review_status === 'FALSE_POSITIVE';

                  return (
                    <tr 
                      key={c.contact_id} 
                      onClick={() => handleRowClick(c)}
                      title={`Click to inspect and verify candidate ${c.contact_id}`}
                      className="hover:bg-blue-50/70 transition-all cursor-pointer group"
                    >
                      <td className="py-3.5 pl-2 font-mono font-bold text-[#0f172a] group-hover:text-[#1d4ed8] transition-colors flex items-center gap-2">
                        <span>{c.contact_id}</span>
                        <span className="text-[10px] opacity-0 group-hover:opacity-100 text-[#1d4ed8] font-sans font-semibold transition-opacity">
                          →
                        </span>
                      </td>
                      <td className="py-3.5">
                        <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold font-sans ${
                          isHigh 
                            ? 'bg-rose-50 text-rose-700 border border-rose-200' 
                            : isMedium
                            ? 'bg-amber-50 text-amber-700 border border-amber-200'
                            : 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                        }`}>
                          {c.priority}
                        </span>
                      </td>
                      <td className="py-3.5 font-bold text-[#0f172a] font-mono">{Math.round(c.confidence * 100)}%</td>
                      <td className="py-3.5 text-[#64748b] font-mono">[{c.bbox.x1}, {c.bbox.y1}, {c.bbox.x2}, {c.bbox.y2}]</td>
                      <td className="py-3.5 font-mono text-slate-700">
                        {c.latitude && c.longitude ? `${c.latitude.toFixed(5)}°, ${c.longitude.toFixed(5)}°` : 'Awaiting GPS Nav Log'}
                      </td>
                      <td className="py-3.5">
                        <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold font-sans ${
                          isConfirmed
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                            : isFalseAlarm
                            ? 'bg-rose-50 text-rose-700 border border-rose-200'
                            : 'bg-amber-50 text-amber-700 border border-amber-200'
                        }`}>
                          {c.review_status.replace('_', ' ')}
                        </span>
                      </td>
                      <td className="py-3.5 text-right pr-2">
                        <span className="text-xs font-semibold text-[#1d4ed8] opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1 justify-end">
                          <span>Triage</span>
                          <ArrowRight className="w-3.5 h-3.5" />
                        </span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Bottom-Right Navigation Action */}
        {onNavigateToDashboard && (
          <div className="flex justify-end pt-4 border-t border-slate-100">
            <button
              onClick={onNavigateToDashboard}
              className="px-5 py-2.5 rounded-full bg-[#1d4ed8] hover:bg-[#1e40af] text-white font-semibold text-xs transition-all duration-200 shadow-tactile shadow-blue-glow flex items-center gap-2 cursor-pointer hover:scale-[1.02] active:scale-[0.98]"
            >
              <ChevronLeft className="w-4 h-4" />
              <span>Back to Dashboard Overview</span>
            </button>
          </div>
        )}
      </section>

      {/* 4. Preview Modals */}
      {previewModal && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-[24px] border border-[#e2e8f0] max-w-2xl w-full p-6 space-y-4 shadow-2xl font-sans animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-[#1d4ed8]" />
                <h4 className="text-base font-extrabold text-[#0f172a] font-display">
                  {previewModal === 'csv' && 'Export Preview: CSV Schema & Sample Rows'}
                  {previewModal === 'geojson' && 'Export Preview: RFC 7946 GeoJSON FeatureCollection'}
                  {previewModal === 'summary' && 'Export Preview: Executive Summary JSON'}
                </h4>
              </div>
              <button 
                onClick={() => setPreviewModal(null)} 
                className="p-1.5 rounded-full hover:bg-slate-100 text-[#64748b] transition-colors cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {copyNotice && (
              <div className="p-2.5 rounded-xl bg-emerald-50 text-emerald-800 text-xs font-semibold flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                <span>{copyNotice}</span>
              </div>
            )}

            <div className="bg-slate-950 text-slate-100 p-4 rounded-xl font-mono text-xs max-h-72 overflow-y-auto border border-slate-800 space-y-1">
              <pre className="whitespace-pre-wrap leading-relaxed">
                {previewModal === 'csv' && csvSample}
                {previewModal === 'geojson' && geoJsonSample}
                {previewModal === 'summary' && summarySample}
              </pre>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-slate-100 text-xs">
              <span className="text-[#64748b]">
                {previewModal === 'csv' && `${contacts.length} total records formatted as RFC 4180.`}
                {previewModal === 'geojson' && `WGS-84 Point coordinates with properties.`}
                {previewModal === 'summary' && `Comprehensive metadata and audit totals.`}
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    const text = previewModal === 'csv' ? csvSample : previewModal === 'geojson' ? geoJsonSample : summarySample;
                    handleCopyPreview(text);
                  }}
                  className="px-3 py-1.5 rounded-xl bg-[#f8fafc] hover:bg-slate-100 border border-[#e2e8f0] text-[#0f172a] font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
                >
                  <Copy className="w-3.5 h-3.5 text-[#64748b]" />
                  <span>Copy Text</span>
                </button>
                <button
                  onClick={() => setPreviewModal(null)}
                  className="px-4 py-1.5 rounded-xl bg-[#1d4ed8] hover:bg-[#1e40af] text-white font-semibold transition-colors cursor-pointer"
                >
                  Done
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};
