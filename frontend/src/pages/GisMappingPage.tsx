import React, { useState } from 'react';
import { Contact, NavWaypoint, SurveyUploadResponse } from '../types/detection';
import { MapView } from '../components/map/MapView';
import { 
  Compass, 
  MapPin, 
  Filter, 
  FileDown, 
  Layers,
  ArrowRight,
  ShieldAlert,
  Waves,
  Scan
} from 'lucide-react';

interface GisMappingPageProps {
  survey: SurveyUploadResponse | null;
  contacts: Contact[];
  selectedContact: Contact | null;
  navTrack: NavWaypoint[];
  onSelectContact: (contact: Contact) => void;
  onNavigateToAnalysis: () => void;
  onNavigateToVerify: () => void;
  onExportGeoJSON: () => void;
}

export const GisMappingPage: React.FC<GisMappingPageProps> = ({
  survey,
  contacts,
  selectedContact,
  navTrack,
  onSelectContact,
  onNavigateToAnalysis,
  onNavigateToVerify,
  onExportGeoJSON
}) => {
  const [filterMode, setFilterMode] = useState<'all' | 'high' | 'confirmed'>('all');

  const filteredContacts = contacts.filter(c => {
    if (filterMode === 'high') return c.priority === 'HIGH';
    if (filterMode === 'confirmed') return c.review_status === 'CONFIRMED';
    return true;
  });

  const activeContact = selectedContact || filteredContacts[0] || null;

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#050a14] text-slate-100 font-sans select-none">
      
      {/* Top Map Toolbar */}
      <div className="h-12 border-b border-[#142244] bg-[#070e1e] px-6 flex items-center justify-between z-10 text-xs shadow-md">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Compass className="w-4 h-4 text-cyan-400" />
            <span className="font-mono font-bold text-white text-sm">GEOSPATIAL COMMAND & NAUTICAL GIS</span>
          </div>
          <span className="text-[10px] px-2.5 py-0.5 rounded-full bg-[#0b162f] text-cyan-300 font-mono font-bold border border-[#1a3366]">
            WGS 84 (EPSG:4326)
          </span>
        </div>

        <div className="flex items-center gap-3">
          {/* Spatial Filter Buttons */}
          <div className="flex items-center gap-1.5 p-1 rounded-lg bg-[#0b1429] border border-[#172b54]">
            <button
              onClick={() => setFilterMode('all')}
              className={`px-3 py-1 rounded-md text-xs font-mono font-bold transition-all ${
                filterMode === 'all'
                  ? 'bg-cyan-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              ALL ({contacts.length})
            </button>
            <button
              onClick={() => setFilterMode('high')}
              className={`px-3 py-1 rounded-md text-xs font-mono font-bold transition-all ${
                filterMode === 'high'
                  ? 'bg-red-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              HIGH PRIORITY
            </button>
            <button
              onClick={() => setFilterMode('confirmed')}
              className={`px-3 py-1 rounded-md text-xs font-mono font-bold transition-all ${
                filterMode === 'confirmed'
                  ? 'bg-emerald-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              CONFIRMED ONLY
            </button>
          </div>

          <button
            onClick={onExportGeoJSON}
            disabled={contacts.length === 0}
            className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white text-xs font-bold transition-all flex items-center gap-1.5 border border-cyan-400/40 shadow-sm"
          >
            <FileDown className="w-3.5 h-3.5" />
            <span>Export RFC 7946 GeoJSON</span>
          </button>
        </div>
      </div>

      {/* Main Map + Side Drawer Workspace */}
      <div className="flex-1 flex overflow-hidden relative">
        
        {/* Full-Height MapLibre Vector Nautical Canvas */}
        <div className="flex-1 h-full relative bg-[#02060f]">
          <MapView
            contacts={filteredContacts}
            selectedContact={activeContact}
            navTrack={navTrack}
            onSelectContact={onSelectContact}
          />
        </div>

        {/* Spatial Target Telemetry Drawer (320px) */}
        {activeContact && (
          <div className="w-84 h-full border-l border-[#142244] bg-[#070e1e] p-4 flex flex-col justify-between overflow-y-auto space-y-4 shadow-2xl z-10">
            <div className="space-y-4">
              <div className="border-b border-[#142244] pb-2.5 flex items-center justify-between">
                <span className="font-mono font-bold text-xs text-white uppercase flex items-center gap-1.5">
                  <MapPin className="w-4 h-4 text-cyan-400" /> TARGET LOCALIZATION
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded-full font-mono font-bold bg-cyan-950 text-cyan-300 border border-cyan-800">
                  {activeContact.contact_id}
                </span>
              </div>

              {/* Coordinates Box */}
              <div className="p-3.5 rounded-xl bg-[#0b1429] border border-[#182a52] space-y-2">
                <div className="text-[10px] text-slate-400 font-mono uppercase font-bold">WGS-84 COORDINATES</div>
                {activeContact.latitude && activeContact.longitude ? (
                  <div className="font-mono text-xs text-emerald-400 font-bold space-y-1">
                    <div>LAT: {activeContact.latitude.toFixed(6)}° N</div>
                    <div>LON: {activeContact.longitude.toFixed(6)}° E</div>
                  </div>
                ) : (
                  <div className="text-[11px] text-amber-400/90 font-mono italic">
                    Coordinates unavailable (Awaiting towfish GPS track log)
                  </div>
                )}
                <div className="text-[10px] text-slate-400 pt-1 border-t border-[#142244] flex justify-between">
                  <span>Provenance:</span>
                  <span className="text-white font-mono font-bold">{activeContact.localization_status}</span>
                </div>
              </div>

              {/* Target Characteristics */}
              <div className="p-3.5 rounded-xl bg-[#0b1429] border border-[#182a52] space-y-2">
                <div className="text-[10px] text-slate-400 font-mono uppercase font-bold">TARGET CHARACTERISTICS</div>
                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Class:</span>
                    <span className="text-white font-mono font-bold">{activeContact.class_name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Priority:</span>
                    <span className={`font-mono font-bold ${
                      activeContact.priority === 'HIGH' ? 'text-red-400' : 'text-amber-400'
                    }`}>
                      {activeContact.priority}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">AI Confidence:</span>
                    <span className="text-white font-mono font-bold">{Math.round(activeContact.confidence * 100)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Triage Status:</span>
                    <span className="text-cyan-300 font-mono font-bold">{activeContact.review_status}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Quick Action Navigation */}
            <div className="space-y-2 pt-2 border-t border-[#142244]">
              <button
                onClick={onNavigateToAnalysis}
                className="w-full py-2.5 rounded-xl bg-[#0b1429] hover:bg-[#122347] text-white font-bold text-xs border border-[#172b54] transition-all flex items-center justify-center gap-2"
              >
                <Waves className="w-4 h-4 text-cyan-400" />
                <span>Inspect in Sonar Waterfall</span>
              </button>
              <button
                onClick={onNavigateToVerify}
                className="w-full py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold text-xs border border-cyan-400/40 transition-all flex items-center justify-center gap-2 shadow-md"
              >
                <span>Verify & Classify Target</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
