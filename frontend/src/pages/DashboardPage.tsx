import React, { useEffect, useState } from 'react';
import { apiService } from '../services/api';
import { Contact, SurveyUploadResponse } from '../types/detection';
import { 
  Activity, 
  Layers, 
  Compass, 
  TrendingUp,
  MapPin,
  Clock,
  ArrowRight,
  ShieldCheck,
  AlertTriangle,
  Radio,
  Scan,
  Database,
  CheckCircle2,
  FileCheck2
} from 'lucide-react';

interface DashboardPageProps {
  survey: SurveyUploadResponse | null;
  contacts: Contact[];
  onSelectScreen: (screen: 'dashboard' | 'sonar-analysis' | 'contact-verification' | 'gis-mapping' | 'ai-pipeline' | 'reports') => void;
  onSelectContact: (contact: Contact) => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  survey,
  contacts,
  onSelectScreen,
  onSelectContact
}) => {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await apiService.getDashboardStats();
        setStats(res);
      } catch (err) {
        console.warn('Dashboard stats error:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, [contacts, survey]);

  const totalDetections = stats?.total_detections ?? contacts.length;
  const confirmedCount = stats?.confirmed_contacts ?? contacts.filter(c => c.review_status === 'CONFIRMED').length;
  const falsePositives = stats?.false_positives ?? contacts.filter(c => c.review_status === 'FALSE_POSITIVE').length;
  const uncertainCount = stats?.uncertain_reviews ?? contacts.filter(c => c.review_status === 'UNCERTAIN').length;
  const highPriority = stats?.priority_distribution?.high ?? contacts.filter(c => c.priority === 'HIGH').length;
  const totalSurveys = stats?.total_surveys ?? (survey ? 1 : 0);

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-[#050a14] text-slate-100 font-sans select-none">
      
      {/* Page Header */}
      <div className="flex items-center justify-between border-b border-[#142244] pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-extrabold tracking-tight text-white font-mono flex items-center gap-2">
              <Activity className="w-5 h-5 text-cyan-400" />
              MISSION INTELLIGENCE DASHBOARD
            </h1>
            <span className="text-[10px] px-2.5 py-0.5 rounded-full bg-cyan-950 text-cyan-300 font-mono font-bold border border-cyan-800 uppercase">
              LIVE TELEMETRY
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1 font-medium">
            Real-Time Side-Scan Sonar Fleet Analytics, AI Candidate Density & Human Verification Audit Stream
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => onSelectScreen('sonar-analysis')}
            className="px-4 py-2 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white text-xs font-bold transition-all shadow-lg shadow-cyan-950/50 border border-cyan-400/40 flex items-center gap-2"
          >
            <span>Open Sonar Workstation</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 6 Key Telemetry Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
        
        {/* Total Surveys */}
        <div className="p-4 rounded-xl bg-[#0a1226] border border-[#162952] space-y-2 shadow-md hover:border-cyan-500/50 transition-all">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono text-[11px] font-bold uppercase tracking-wider text-slate-400">TOTAL SURVEYS</span>
            <Layers className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-extrabold text-white font-mono">{totalSurveys}</div>
          <div className="text-[11px] text-cyan-400/90 font-medium">L01–L07 Tracklines Active</div>
        </div>

        {/* AI Proposals */}
        <div className="p-4 rounded-xl bg-[#0a1226] border border-[#162952] space-y-2 shadow-md hover:border-cyan-500/50 transition-all">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono text-[11px] font-bold uppercase tracking-wider text-slate-400">AI PROPOSALS</span>
            <Scan className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-extrabold text-white font-mono">{totalDetections}</div>
          <div className="text-[11px] text-slate-400 font-mono">yolov8n-baseline</div>
        </div>

        {/* Confirmed Debris */}
        <div className="p-4 rounded-xl bg-[#0a1226] border border-[#162952] space-y-2 shadow-md hover:border-emerald-500/50 transition-all">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono text-[11px] font-bold uppercase tracking-wider text-slate-400">CONFIRMED DEBRIS</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-extrabold text-emerald-400 font-mono">{confirmedCount}</div>
          <div className="text-[11px] text-emerald-400/90 font-medium">Verified by Operator</div>
        </div>

        {/* High Priority Targets */}
        <div className="p-4 rounded-xl bg-[#0a1226] border border-[#162952] space-y-2 shadow-md hover:border-red-500/50 transition-all">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono text-[11px] font-bold uppercase tracking-wider text-slate-400">HIGH PRIORITY</span>
            <AlertTriangle className="w-4 h-4 text-red-400" />
          </div>
          <div className="text-2xl font-extrabold text-red-400 font-mono">{highPriority}</div>
          <div className="text-[11px] text-red-400/90 font-medium">Immediate ROV Targets</div>
        </div>

        {/* False Positives Rejected */}
        <div className="p-4 rounded-xl bg-[#0a1226] border border-[#162952] space-y-2 shadow-md hover:border-slate-500/50 transition-all">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono text-[11px] font-bold uppercase tracking-wider text-slate-400">FALSE ALARMS</span>
            <ShieldCheck className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-2xl font-extrabold text-slate-300 font-mono">{falsePositives}</div>
          <div className="text-[11px] text-slate-400">Geological Clutter</div>
        </div>

        {/* Database Status */}
        <div className="p-4 rounded-xl bg-[#0a1226] border border-[#162952] space-y-2 shadow-md hover:border-cyan-500/50 transition-all">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono text-[11px] font-bold uppercase tracking-wider text-slate-400">SPATIAL DB</span>
            <Database className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-lg font-extrabold text-cyan-300 font-mono mt-1">PostGIS 15</div>
          <div className="text-[11px] text-emerald-400 font-medium flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-400"></span> EPSG:4326 Sync
          </div>
        </div>
      </div>

      {/* Middle Telemetry Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Swath Anomaly Density Chart (7 Cols) */}
        <div className="lg:col-span-7 p-5 rounded-2xl bg-[#091226] border border-[#15274f] space-y-4 shadow-lg">
          <div className="flex items-center justify-between border-b border-[#142244] pb-3">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-cyan-400" />
              <h3 className="font-mono font-bold text-sm text-white">SWATH ANOMALY DENSITY & HIT RATE</h3>
            </div>
            <span className="text-[10px] font-mono text-slate-400">LINES L01 – L07</span>
          </div>

          <div className="h-48 flex items-end justify-between gap-3 pt-6 px-4">
            {[
              { label: 'L01 (Bay)', count: 4, height: '40%', color: 'bg-cyan-500' },
              { label: 'L02 (Reef)', count: 9, height: '85%', color: 'bg-amber-500' },
              { label: 'L03 (Channel)', count: 3, height: '30%', color: 'bg-cyan-500' },
              { label: 'L04 (Wreck)', count: 11, height: '100%', color: 'bg-red-500' },
              { label: 'L05 (Slope)', count: 6, height: '55%', color: 'bg-amber-500' },
              { label: 'L06 (Deep)', count: 2, height: '20%', color: 'bg-cyan-500' },
              { label: 'L07 (Shoal)', count: 5, height: '45%', color: 'bg-cyan-500' },
            ].map((bar, idx) => (
              <div key={idx} className="flex-1 flex flex-col items-center gap-2 h-full justify-end group">
                <span className="text-[10px] font-mono text-slate-400 font-bold opacity-0 group-hover:opacity-100 transition-opacity">
                  {bar.count}
                </span>
                <div 
                  className={`w-full ${bar.color} rounded-t-lg transition-all duration-500 hover:brightness-125 shadow-lg`}
                  style={{ height: bar.height }}
                ></div>
                <span className="text-[10px] font-mono text-slate-400 text-center truncate max-w-full">
                  {bar.label.split(' ')[0]}
                </span>
              </div>
            ))}
          </div>

          <div className="p-3 rounded-xl bg-[#060b17] border border-[#121f3f] flex items-center justify-between text-xs text-slate-400">
            <span className="text-[11px] font-mono">Benchmark Peak: <strong className="text-white">L04 (Viator Shipwreck)</strong> with 11 acoustic candidate returns.</span>
            <span className="text-cyan-400 text-[11px] font-mono font-bold">52.3 FPS INFERENCE</span>
          </div>
        </div>

        {/* Priority Breakdown Radar Vector (5 Cols) */}
        <div className="lg:col-span-5 p-5 rounded-2xl bg-[#091226] border border-[#15274f] space-y-4 shadow-lg flex flex-col justify-between">
          <div className="flex items-center justify-between border-b border-[#142244] pb-3">
            <div className="flex items-center gap-2">
              <Compass className="w-4 h-4 text-amber-400" />
              <h3 className="font-mono font-bold text-sm text-white">PRIORITY TRIAGE CLASSIFICATION</h3>
            </div>
            <span className="text-[10px] font-mono text-emerald-400">CALIBRATED</span>
          </div>

          <div className="space-y-3 py-2">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-red-300 font-mono font-bold flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-red-500"></span> HIGH PRIORITY (Deficit Shadow)
                </span>
                <span className="font-mono font-bold text-white">{highPriority} Candidates</span>
              </div>
              <div className="w-full h-2 rounded-full bg-[#050b17] overflow-hidden border border-[#142244]">
                <div className="h-full bg-gradient-to-r from-red-600 to-red-400" style={{ width: `${Math.min(100, (highPriority / Math.max(1, totalDetections)) * 100)}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-amber-300 font-mono font-bold flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-amber-500"></span> MEDIUM PRIORITY (Anomaly Proposals)
                </span>
                <span className="font-mono font-bold text-white">{totalDetections - highPriority} Candidates</span>
              </div>
              <div className="w-full h-2 rounded-full bg-[#050b17] overflow-hidden border border-[#142244]">
                <div className="h-full bg-gradient-to-r from-amber-600 to-amber-400" style={{ width: '65%' }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-sky-300 font-mono font-bold flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-sky-500"></span> LOW / AMBIENT SEABED
                </span>
                <span className="font-mono font-bold text-white">0 Excluded</span>
              </div>
              <div className="w-full h-2 rounded-full bg-[#050b17] overflow-hidden border border-[#142244]">
                <div className="h-full bg-sky-500" style={{ width: '10%' }}></div>
              </div>
            </div>
          </div>

          <button
            onClick={() => onSelectScreen('gis-mapping')}
            className="w-full py-2.5 rounded-xl bg-[#0e1c38] hover:bg-[#142852] text-cyan-300 font-bold text-xs border border-[#1f3b75] transition-all flex items-center justify-center gap-2"
          >
            <Compass className="w-4 h-4 text-cyan-400" />
            <span>Open Geospatial Map Context</span>
          </button>
        </div>
      </div>

      {/* Recent Immutable Audit Log Stream */}
      <div className="p-5 rounded-2xl bg-[#091226] border border-[#15274f] space-y-4 shadow-lg">
        <div className="flex items-center justify-between border-b border-[#142244] pb-3">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-cyan-400" />
            <h3 className="font-mono font-bold text-sm text-white">RECENT PLATFORM INGESTION & OPERATOR TRIAGE AUDIT LOG</h3>
          </div>
          <span className="text-[10px] font-mono text-slate-400">WGS-84 AUDIT TRAIL</span>
        </div>

        <div className="space-y-2">
          {contacts.slice(0, 5).map((c, i) => (
            <div
              key={c.contact_id}
              onClick={() => {
                onSelectContact(c);
                onSelectScreen('sonar-analysis');
              }}
              className="p-3 rounded-xl bg-[#0b1429] hover:bg-[#111f3d] border border-[#182a52] transition-all flex items-center justify-between cursor-pointer group"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-black border border-[#1d3363] flex items-center justify-center font-mono font-bold text-xs text-cyan-300">
                  {c.contact_id}
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-white font-mono text-xs">{c.contact_id}</span>
                    <span className={`text-[9px] px-2 py-0.2 rounded-full font-mono font-bold ${
                      c.priority === 'HIGH' ? 'bg-red-950 text-red-300 border border-red-800' : 'bg-amber-950 text-amber-300 border border-amber-800'
                    }`}>
                      {c.priority}
                    </span>
                    <span className="text-[11px] text-slate-400 font-mono">• Confidence: {Math.round(c.confidence * 100)}%</span>
                  </div>
                  <div className="text-[11px] text-slate-400 mt-0.5">
                    Survey Swath: <span className="text-slate-300 font-mono">{c.survey_id}</span> • Slant BBox: [{c.bbox.x1}, {c.bbox.y1}, {c.bbox.x2}, {c.bbox.y2}]
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className="text-[10px] px-2.5 py-1 rounded-lg bg-[#060b17] border border-[#142244] text-cyan-300 font-mono font-semibold">
                  {c.review_status}
                </span>
                <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-cyan-400 transition-colors" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
