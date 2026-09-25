import React, { useState } from 'react';
import { Contact, SurveyUploadResponse } from '../types/detection';
import { 
  FileText, 
  FileDown, 
  MapPin, 
  Cpu, 
  Clock,
  Download,
  CheckCircle2,
  Table,
  FileSpreadsheet,
  Globe,
  FileCheck
} from 'lucide-react';

interface ReportsPageProps {
  survey: SurveyUploadResponse | null;
  contacts: Contact[];
}

export const ReportsPage: React.FC<ReportsPageProps> = ({ survey, contacts }) => {
  const [downloadMsg, setDownloadMsg] = useState<string | null>(null);

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

  const confirmedCount = contacts.filter(c => c.review_status === 'CONFIRMED').length;
  const fpCount = contacts.filter(c => c.review_status === 'FALSE_POSITIVE').length;

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-[#050a14] text-slate-100 font-sans select-none">
      
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#142244] pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-xl font-extrabold tracking-tight text-white font-mono flex items-center gap-2">
              <FileText className="w-5 h-5 text-cyan-400" />
              HYDROGRAPHIC DATA PRODUCTS & EXPORT CENTRAL
            </h1>
            <span className="text-[10px] px-2.5 py-0.5 rounded-full bg-cyan-950 text-cyan-300 font-mono font-bold border border-cyan-800 uppercase">
              STANDARDIZED EXPORTS
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1 font-medium">
            Generate RFC 7946 GeoJSON GIS Layers, Tabular Hydrographic Audit CSVs & Executive Marine Remediation Summaries
          </p>
        </div>

        <div className="text-xs text-slate-400 font-mono">
          Active Swath: <span className="text-cyan-300 font-bold">{survey?.survey_id || 'None'}</span>
        </div>
      </div>

      {downloadMsg && (
        <div className="p-3.5 rounded-xl bg-emerald-950/80 border border-emerald-700 text-emerald-300 text-xs font-mono font-bold flex items-center justify-between">
          <span>✓ {downloadMsg}</span>
          <button onClick={() => setDownloadMsg(null)} className="underline text-emerald-400 text-[11px]">Dismiss</button>
        </div>
      )}

      {/* 3 Core Hydrographic Data Products */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Product 1: Tabular CSV */}
        <div className="p-5 rounded-2xl bg-[#091226] border border-[#15274f] space-y-4 shadow-lg flex flex-col justify-between hover:border-cyan-500/50 transition-all group">
          <div className="space-y-3">
            <div className="w-10 h-10 rounded-xl bg-[#0e1c38] border border-[#1e3b75] flex items-center justify-center text-cyan-400 group-hover:scale-110 transition-transform">
              <FileSpreadsheet className="w-5 h-5" />
            </div>
            <h3 className="font-mono font-bold text-sm text-white">TABULAR DETECTIONS CSV</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Full spreadsheet export with Candidate IDs, pixel bounds, AI confidences, acoustic evidence scores, and review status logs.
            </p>
          </div>

          <button
            onClick={handleExportCSV}
            disabled={!survey}
            className="w-full py-2.5 rounded-xl bg-[#0e1c38] hover:bg-cyan-600 hover:text-white text-cyan-300 font-bold text-xs border border-[#1f3b75] transition-all flex items-center justify-center gap-2 disabled:opacity-50"
          >
            <Download className="w-4 h-4" />
            <span>Download Detections CSV</span>
          </button>
        </div>

        {/* Product 2: Spatial GeoJSON */}
        <div className="p-5 rounded-2xl bg-[#091226] border border-[#15274f] space-y-4 shadow-lg flex flex-col justify-between hover:border-cyan-500/50 transition-all group">
          <div className="space-y-3">
            <div className="w-10 h-10 rounded-xl bg-[#0e1c38] border border-[#1e3b75] flex items-center justify-center text-emerald-400 group-hover:scale-110 transition-transform">
              <Globe className="w-5 h-5" />
            </div>
            <h3 className="font-mono font-bold text-sm text-white">SPATIAL RFC 7946 GeoJSON</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Standardized FeatureCollection of Point geometries ready for immediate drag-and-drop ingestion into QGIS, ArcGIS, or MapStore.
            </p>
          </div>

          <button
            onClick={handleExportGeoJSON}
            disabled={!survey}
            className="w-full py-2.5 rounded-xl bg-[#0e1c38] hover:bg-emerald-600 hover:text-white text-emerald-300 font-bold text-xs border border-[#1f3b75] transition-all flex items-center justify-center gap-2 disabled:opacity-50"
          >
            <Download className="w-4 h-4" />
            <span>Download Spatial GeoJSON</span>
          </button>
        </div>

        {/* Product 3: Executive Summary */}
        <div className="p-5 rounded-2xl bg-[#091226] border border-[#15274f] space-y-4 shadow-lg flex flex-col justify-between hover:border-cyan-500/50 transition-all group">
          <div className="space-y-3">
            <div className="w-10 h-10 rounded-xl bg-[#0e1c38] border border-[#1e3b75] flex items-center justify-center text-amber-400 group-hover:scale-110 transition-transform">
              <FileCheck className="w-5 h-5" />
            </div>
            <h3 className="font-mono font-bold text-sm text-white">EXECUTIVE SURVEY SUMMARY</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Structured hydrographic summary report covering swath coverage, dynamic range, candidate counts, and operator triage resolution rates.
            </p>
          </div>

          <button
            onClick={handleExportSummary}
            disabled={!survey}
            className="w-full py-2.5 rounded-xl bg-[#0e1c38] hover:bg-amber-600 hover:text-white text-amber-300 font-bold text-xs border border-[#1f3b75] transition-all flex items-center justify-center gap-2 disabled:opacity-50"
          >
            <Download className="w-4 h-4" />
            <span>Download Summary JSON</span>
          </button>
        </div>
      </div>

      {/* Live Data Preview Table */}
      <div className="p-5 rounded-2xl bg-[#091226] border border-[#15274f] space-y-4 shadow-lg">
        <div className="flex items-center justify-between border-b border-[#142244] pb-3">
          <div className="flex items-center gap-2">
            <Table className="w-4 h-4 text-cyan-400" />
            <h3 className="font-mono font-bold text-sm text-white">SURVEY DETECTIONS EXPORT PREVIEW ({contacts.length} ROWS)</h3>
          </div>
          <span className="text-[10px] font-mono text-slate-400">TABLE PREVIEW</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-[#142244] text-slate-400 text-[10px] uppercase">
                <th className="pb-2">Contact ID</th>
                <th className="pb-2">Priority</th>
                <th className="pb-2">Confidence</th>
                <th className="pb-2">Bounding Coordinates</th>
                <th className="pb-2">Latitude / Longitude</th>
                <th className="pb-2">Triage Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#122144]">
              {contacts.map((c) => (
                <tr key={c.contact_id} className="hover:bg-[#0c1833] transition-colors">
                  <td className="py-2.5 font-bold text-cyan-300">{c.contact_id}</td>
                  <td className="py-2.5">
                    <span className={`px-2 py-0.5 rounded text-[9px] font-bold ${
                      c.priority === 'HIGH' ? 'bg-red-950 text-red-300 border border-red-800' : 'bg-amber-950 text-amber-300 border border-amber-800'
                    }`}>
                      {c.priority}
                    </span>
                  </td>
                  <td className="py-2.5 font-bold text-white">{Math.round(c.confidence * 100)}%</td>
                  <td className="py-2.5 text-slate-400">[{c.bbox.x1}, {c.bbox.y1}, {c.bbox.x2}, {c.bbox.y2}]</td>
                  <td className="py-2.5 text-slate-300">
                    {c.latitude && c.longitude ? `${c.latitude.toFixed(5)}°, ${c.longitude.toFixed(5)}°` : 'Awaiting GPS Nav Log'}
                  </td>
                  <td className="py-2.5">
                    <span className="px-2 py-0.5 rounded bg-[#060b17] border border-[#142244] text-slate-300 text-[10px]">
                      {c.review_status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
