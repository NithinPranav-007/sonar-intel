import React from 'react';
import { Contact } from '../../types/detection';
import { MapPin, Cpu, Box, FileText, Info } from 'lucide-react';

interface ConfidencePanelProps {
  contact: Contact;
}

export const ConfidencePanel: React.FC<ConfidencePanelProps> = ({ contact }) => {
  const getPriorityBadgeClass = (priority: string) => {
    switch (priority) {
      case 'HIGH':
        return 'bg-red-500/20 text-red-300 border-red-500/50 shadow-[0_0_8px_rgba(239,68,68,0.4)]';
      case 'MEDIUM':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/50 shadow-[0_0_8px_rgba(245,158,11,0.4)]';
      default:
        return 'bg-sky-500/20 text-sky-300 border-sky-500/50 shadow-[0_0_8px_rgba(56,189,248,0.4)]';
    }
  };

  const getReviewBadgeClass = (status: string) => {
    switch (status) {
      case 'CONFIRMED':
        return 'bg-emerald-950 text-emerald-300 border-emerald-700';
      case 'FALSE_POSITIVE':
        return 'bg-red-950 text-red-300 border-red-800';
      case 'UNCERTAIN':
        return 'bg-purple-950 text-purple-300 border-purple-700';
      default:
        return 'bg-cyan-950 text-cyan-300 border-cyan-800';
    }
  };

  return (
    <div className="space-y-3 font-mono text-xs select-none">
      {/* 1. Header Info: Contact ID, Classification, Priority & Review Badges */}
      <div className="border-b border-[#1a2f4c] pb-2 space-y-1.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-base font-bold text-slate-100">{contact.contact_id}</span>
            <span className="text-[10px] text-slate-400 font-sans">({contact.class_name})</span>
          </div>
          <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getPriorityBadgeClass(contact.priority)}`}>
            {contact.priority} PRIORITY
          </span>
        </div>

        <div className="flex items-center justify-between text-[10px]">
          <span className="text-slate-400">REVIEW STATUS:</span>
          <span className={`px-1.5 py-0.2 rounded font-bold border ${getReviewBadgeClass(contact.review_status)}`}>
            {contact.review_status.replace('_', ' ')}
          </span>
        </div>
      </div>

      {/* 2. Metadata Provenance: Model, Source, Tile Coordinates */}
      <div className="p-2 rounded bg-[#070e1a] border border-[#1a2f4c] space-y-1 text-[11px]">
        <div className="flex items-center justify-between text-slate-400">
          <span className="flex items-center gap-1"><Cpu className="w-3 h-3 text-cyan-400" /> MODEL:</span>
          <span className="text-slate-200 font-semibold">{contact.model_version || 'yolov8n-sonar-baseline'}</span>
        </div>
        <div className="flex items-center justify-between text-slate-400">
          <span className="flex items-center gap-1"><FileText className="w-3 h-3 text-cyan-400" /> SURVEY:</span>
          <span className="text-slate-200 truncate max-w-[150px]">{contact.survey_id}</span>
        </div>
        <div className="flex items-center justify-between text-slate-400">
          <span className="flex items-center gap-1"><Box className="w-3 h-3 text-cyan-400" /> BOUNDING BOX:</span>
          <span className="text-slate-300 text-[10px]">
            [{contact.bbox.x1}, {contact.bbox.y1}] - [{contact.bbox.x2}, {contact.bbox.y2}]
          </span>
        </div>
      </div>

      {/* 3. Acoustic Metrics & Heuristic Scores (Clearly Labeled as Scores) */}
      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <div className="p-2 rounded bg-[#070e1a] border border-[#1a2f4c]">
          <div className="text-slate-400 text-[10px]">DETECTOR CONFIDENCE</div>
          <div className="text-base font-bold text-cyan-400">
            {Math.round(contact.confidence * 100)}%
          </div>
          <div className="w-full bg-[#122238] h-1 rounded mt-1 overflow-hidden">
            <div className="bg-cyan-400 h-full" style={{ width: `${contact.confidence * 100}%` }} />
          </div>
        </div>

        <div className="p-2 rounded bg-[#070e1a] border border-[#1a2f4c]">
          <div className="text-slate-400 text-[10px]">DATA QUALITY SCORE</div>
          <div className="text-base font-bold text-emerald-400">
            {Math.round(contact.data_quality * 100)}%
          </div>
          <div className="w-full bg-[#122238] h-1 rounded mt-1 overflow-hidden">
            <div className="bg-emerald-400 h-full" style={{ width: `${contact.data_quality * 100}%` }} />
          </div>
        </div>

        <div className="p-2 rounded bg-[#070e1a] border border-[#1a2f4c]">
          <div className="text-slate-400 text-[10px]">SHADOW EVIDENCE SCORE</div>
          <div className="text-base font-bold text-amber-400">
            {Math.round(contact.shadow_evidence * 100)}%
          </div>
          <div className="w-full bg-[#122238] h-1 rounded mt-1 overflow-hidden">
            <div className="bg-amber-400 h-full" style={{ width: `${contact.shadow_evidence * 100}%` }} />
          </div>
        </div>

        <div className="p-2 rounded bg-[#070e1a] border border-[#1a2f4c]">
          <div className="text-slate-400 text-[10px]">CONTEXT SCORE</div>
          <div className="text-base font-bold text-purple-400">
            {Math.round(contact.context_score * 100)}%
          </div>
          <div className="w-full bg-[#122238] h-1 rounded mt-1 overflow-hidden">
            <div className="bg-purple-400 h-full" style={{ width: `${contact.context_score * 100}%` }} />
          </div>
        </div>
      </div>

      <div className="text-[10px] text-slate-500 font-sans flex items-start gap-1">
        <Info className="w-3 h-3 shrink-0 text-slate-600 mt-0.5" />
        <span>* Scores are heuristic acoustic physics measures, not calibrated probabilities.</span>
      </div>

      {/* 4. Geolocation Readout */}
      <div className="p-2 rounded bg-[#070e1a] border border-[#1a2f4c] flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-slate-300">
          <MapPin className="w-3.5 h-3.5 text-cyan-400" />
          <span>LOCATION</span>
        </div>
        <div className="text-right">
          {contact.latitude != null ? (
            <div className="text-cyan-300 font-semibold">
              {contact.latitude.toFixed(6)}°N, {contact.longitude?.toFixed(6)}°E
            </div>
          ) : (
            <span className="text-slate-500 italic">Location unavailable</span>
          )}
          <span className="text-[10px] text-slate-400"> ({contact.localization_status})</span>
        </div>
      </div>
    </div>
  );
};
