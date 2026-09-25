import React from 'react';
import { Contact, SurveyUploadResponse } from '../../types/detection';
import { FileDown, Globe, Table } from 'lucide-react';
import { apiService } from '../../services/api';

interface ExportPanelProps {
  survey: SurveyUploadResponse;
  contacts: Contact[];
}

export const ExportPanel: React.FC<ExportPanelProps> = ({ survey, contacts }) => {
  const handleExportGeoJSON = async () => {
    try {
      const data = await apiService.getSurveyGeoJSON(survey.survey_id);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/geo+json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${survey.survey_id}_contacts.geojson`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export GeoJSON error:', err);
    }
  };

  const handleExportCSV = () => {
    window.open(`/api/surveys/${survey.survey_id}/csv`, '_blank');
  };

  return (
    <div className="p-3 bg-[#0a1322] border border-[#1a2f4c] rounded flex items-center justify-between font-mono text-xs">
      <div>
        <span className="font-bold text-slate-200">MISSION REPORTS & EXPORTS</span>
        <p className="text-[11px] text-slate-400 font-sans">
          Download geospatial contacts for QGIS, ECDIS, or post-survey retraining.
        </p>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={handleExportGeoJSON}
          className="px-3 py-1.5 rounded bg-[#10223b] hover:bg-[#163054] border border-cyan-800 text-cyan-300 flex items-center gap-1.5 transition-colors"
        >
          <Globe className="w-3.5 h-3.5" />
          <span>GeoJSON</span>
        </button>

        <button
          onClick={handleExportCSV}
          className="px-3 py-1.5 rounded bg-[#10223b] hover:bg-[#163054] border border-[#1a2f4c] text-slate-200 flex items-center gap-1.5 transition-colors"
        >
          <Table className="w-3.5 h-3.5" />
          <span>CSV Export</span>
        </button>
      </div>
    </div>
  );
};
