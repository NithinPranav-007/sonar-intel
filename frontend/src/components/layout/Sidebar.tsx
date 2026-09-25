import React, { useState } from 'react';
import { 
  LayoutDashboard, 
  Waves, 
  CheckCircle2, 
  Compass, 
  Cpu, 
  FileText, 
  Play, 
  Search, 
  Radio,
  Sliders,
  ChevronRight,
  UserCheck
} from 'lucide-react';
import { SurveyUploadResponse, Contact } from '../../types/detection';

export type ActiveScreen = 
  | 'dashboard' 
  | 'sonar-analysis' 
  | 'contact-verification' 
  | 'gis-mapping' 
  | 'ai-pipeline' 
  | 'reports';

interface SidebarProps {
  activeScreen: ActiveScreen;
  onSelectScreen: (screen: ActiveScreen) => void;
  survey: SurveyUploadResponse | null;
  contacts: Contact[];
  selectedContact?: Contact | null;
  analyzing: boolean;
  onRunAnalysis: () => void;
  onSelectContact?: (contact: Contact) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeScreen,
  onSelectScreen,
  survey,
  contacts,
  analyzing,
  onRunAnalysis,
  onSelectContact
}) => {
  const [searchQuery, setSearchQuery] = useState('');

  const filteredContacts = searchQuery.trim()
    ? contacts.filter(c => {
        const q = searchQuery.toLowerCase().trim();
        return (
          c.contact_id.toLowerCase().includes(q) ||
          c.priority.toLowerCase().includes(q) ||
          c.review_status.toLowerCase().includes(q) ||
          c.class_name.toLowerCase().includes(q)
        );
      })
    : [];

  const navItems = [
    { id: 'dashboard' as ActiveScreen, label: 'Dashboard Overview', icon: LayoutDashboard },
    { id: 'sonar-analysis' as ActiveScreen, label: 'Sonar Waterfall', icon: Waves },
    { id: 'contact-verification' as ActiveScreen, label: 'Contact Triage', icon: CheckCircle2 },
    { id: 'gis-mapping' as ActiveScreen, label: 'GIS Mapping & Spatial', icon: Compass },
    { id: 'ai-pipeline' as ActiveScreen, label: 'Pipeline Monitor', icon: Cpu },
    { id: 'reports' as ActiveScreen, label: 'Reports & Export', icon: FileText },
  ];

  return (
    <aside className="w-56 border-r border-[#172542] bg-[#091122] flex flex-col justify-between p-2.5 select-none text-xs font-sans">
      <div className="space-y-3">
        
        {/* Navigation Section */}
        <div className="space-y-0.5">
          <div className="text-[9px] uppercase tracking-wider text-slate-400 font-mono font-bold px-2 py-1">
            WORKSPACES
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeScreen === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectScreen(item.id)}
                className={`w-full py-1.5 px-2 rounded flex items-center justify-between text-xs font-medium transition-colors ${
                  isActive
                    ? 'bg-[#132242] text-white font-semibold border-l-2 border-cyan-400 pl-2.5'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-[#0d172e] border-l-2 border-transparent'
                }`}
              >
                <div className="flex items-center gap-2">
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                  <span className="text-[11px]">{item.label}</span>
                </div>
                {isActive && <span className="w-1 h-1 rounded-full bg-cyan-400"></span>}
              </button>
            );
          })}
        </div>

        {/* Active Swath Provenance Widget */}
        <div className="p-2.5 rounded bg-[#060b17] border border-[#142038] space-y-2">
          <div className="flex items-center justify-between text-[9px] font-mono font-bold">
            <span className="text-slate-400 uppercase flex items-center gap-1">
              <Radio className="w-2.5 h-2.5 text-cyan-400" /> ACTIVE SWATH
            </span>
            <span className={`px-1 rounded ${
              survey ? 'text-emerald-400 bg-emerald-950/80' : 'text-slate-500 bg-slate-900'
            }`}>
              {survey ? 'INGESTED' : 'IDLE'}
            </span>
          </div>

          {survey ? (
            <div className="space-y-1 text-[11px] font-mono">
              <div className="truncate text-slate-200 font-semibold" title={survey.filename}>
                {survey.filename}
              </div>
              <div className="flex justify-between text-slate-400 text-[10px]">
                <span>Quality Index:</span>
                <span className="text-emerald-400 font-bold">{Math.round(survey.data_quality * 100)}%</span>
              </div>
              <div className="flex justify-between text-slate-400 text-[10px]">
                <span>AI Proposals:</span>
                <span className="text-cyan-400 font-bold">{contacts.length} Contacts</span>
              </div>
            </div>
          ) : (
            <p className="text-slate-500 text-[10px] italic">No swath loaded.</p>
          )}

          {survey && (
            <button
              onClick={onRunAnalysis}
              disabled={analyzing}
              className={`w-full mt-1 py-1 rounded text-[11px] font-mono font-bold flex items-center justify-center gap-1.5 transition-colors ${
                analyzing
                  ? 'bg-slate-800 text-slate-400 cursor-wait animate-pulse'
                  : 'bg-[#15274d] hover:bg-[#1d3569] text-cyan-300 border border-[#234282]'
              }`}
            >
              <Play className="w-3 h-3 fill-current" />
              <span>{analyzing ? 'ANALYZING...' : 'RUN INFERENCE'}</span>
            </button>
          )}
        </div>

        {/* Candidate Query Search */}
        {contacts.length > 0 && (
          <div className="space-y-1">
            <div className="text-[9px] uppercase tracking-wider text-slate-400 font-mono font-bold flex items-center justify-between px-1">
              <span className="flex items-center gap-1"><Search className="w-2.5 h-2.5 text-slate-400" /> CONTACT FILTER</span>
              {searchQuery && (
                <button onClick={() => setSearchQuery('')} className="text-[9px] text-cyan-400 hover:underline">
                  CLEAR
                </button>
              )}
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter by ID (C001)..."
              className="w-full px-2 py-1 rounded bg-[#060b17] border border-[#142038] focus:border-cyan-500 text-slate-200 text-[10px] placeholder:text-slate-600 focus:outline-none font-mono"
            />
            {searchQuery.trim() && (
              <div className="max-h-28 overflow-y-auto space-y-0.5 mt-1 pr-0.5">
                {filteredContacts.map(c => (
                  <div
                    key={c.contact_id}
                    onClick={() => onSelectContact?.(c)}
                    className="p-1 rounded bg-[#0b1429] hover:bg-[#122244] border border-[#142038] cursor-pointer flex items-center justify-between text-[10px] font-mono transition-colors"
                  >
                    <div>
                      <span className="font-bold text-slate-200">{c.contact_id}</span>
                      <span className="text-slate-400 ml-1">({Math.round(c.confidence * 100)}%)</span>
                    </div>
                    <span className={`text-[8px] px-1 rounded font-bold ${
                      c.priority === 'HIGH' ? 'text-red-400 bg-red-950' : 'text-amber-400 bg-amber-950'
                    }`}>
                      {c.priority}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Operator Session Badge */}
      <div className="p-2 rounded bg-[#060b17] border border-[#142038] flex items-center gap-2 text-xs">
        <div className="w-6 h-6 rounded bg-[#101b33] border border-[#1c305c] flex items-center justify-center text-cyan-400 shrink-0 font-mono font-bold text-[10px]">
          CV
        </div>
        <div className="overflow-hidden">
          <div className="text-[11px] font-mono font-bold text-slate-200 truncate leading-tight">Dr. C. Vance</div>
          <div className="text-[9px] text-slate-400 font-mono truncate">Hydrographic Analyst</div>
        </div>
      </div>
    </aside>
  );
};
