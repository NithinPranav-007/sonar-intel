import React, { useState, useEffect } from 'react';
import { Anchor, ShieldCheck, Activity, Info, ChevronDown, Search, Radio, Clock, Database, Terminal, Cpu } from 'lucide-react';
import { apiService } from '../../services/api';

interface HeaderProps {
  surveyId?: string;
  onLoadDemoSample: (sampleId: string) => void;
  onCustomUploadClick?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  surveyId,
  onLoadDemoSample,
  onCustomUploadClick
}) => {
  const [backendStatus, setBackendStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [showDisclaimer, setShowDisclaimer] = useState<boolean>(false);
  const [showDemoMenu, setShowDemoMenu] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [utcTime, setUtcTime] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC');
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const check = async () => {
      const res = await apiService.checkHealth();
      setBackendStatus(res.status === 'healthy' ? 'online' : 'offline');
    };
    check();
    const interval = setInterval(check, 20000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="h-12 border-b border-[#172542] bg-[#091122] px-4 flex items-center justify-between z-30 select-none font-sans">
      
      {/* Brand & System Station Header */}
      <div className="flex items-center gap-3">
        <div className="w-7 h-7 rounded bg-[#132242] border border-[#233b6e] flex items-center justify-center text-cyan-400 font-mono font-bold text-xs shadow-xs">
          <Anchor className="w-4 h-4 stroke-[2.2]" />
        </div>
        <div className="flex items-center gap-2.5">
          <span className="font-mono font-bold text-sm tracking-wider text-slate-100">
            SONAR-INTEL
          </span>
          <span className="text-[9px] px-1.5 py-0.2 rounded bg-[#0d1830] text-cyan-400 font-mono font-bold border border-[#1b315e] uppercase">
            STATION 01 • MoES
          </span>
        </div>
      </div>

      {/* Center Search / Coordinates Query Box */}
      <div className="hidden md:flex items-center relative w-80 lg:w-[420px]">
        <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 pointer-events-none" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Query target ID (C001), ping index, or GPS coordinate..."
          className="w-full pl-8 pr-3 py-1 text-xs bg-[#050a14] border border-[#172542] rounded text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500 font-mono text-[11px]"
        />
      </div>

      {/* Right Telemetry: UTC Clock, Curated Demos, Model & Sensor Link */}
      <div className="flex items-center gap-3 text-xs font-mono">
        
        {/* Live System UTC Clock */}
        <div className="hidden xl:flex items-center gap-1.5 px-2 py-0.5 rounded bg-[#050a14] border border-[#142038] text-[11px] text-slate-400">
          <Clock className="w-3 h-3 text-slate-400" />
          <span>{utcTime || '2026-09-02 22:30:00 UTC'}</span>
        </div>

        {/* Curated Demo Swath Selector */}
        <div className="relative">
          <button
            onClick={() => setShowDemoMenu(!showDemoMenu)}
            className="px-2.5 py-1 text-xs font-mono font-medium rounded bg-[#101b33] hover:bg-[#162647] text-slate-200 border border-[#1d3057] transition-colors flex items-center gap-1.5"
          >
            <Activity className="w-3.5 h-3.5 text-amber-400" />
            <span>Curated Swaths</span>
            <ChevronDown className="w-3 h-3 text-slate-400 ml-0.5" />
          </button>

          {showDemoMenu && (
            <div className="absolute right-0 top-9 w-84 bg-[#0a1224] border border-[#1f3561] rounded shadow-2xl p-2 z-50 text-xs space-y-1">
              <div className="px-2 py-1 text-[9px] uppercase text-slate-400 font-mono font-bold border-b border-[#142240] flex justify-between">
                <span>HELD-OUT TEST SUITES</span>
                <span className="text-cyan-400">BENCHMARKS</span>
              </div>

              <button
                onClick={() => {
                  onLoadDemoSample('viator_04');
                  setShowDemoMenu(false);
                }}
                className="w-full text-left p-2 rounded hover:bg-[#111e38] transition-colors border border-transparent hover:border-[#1d335e]"
              >
                <div className="font-mono font-bold text-slate-100 flex items-center justify-between text-xs">
                  <span>Viator-04 (True Shipwreck)</span>
                  <span className="text-[9px] px-1 py-0.2 rounded bg-red-950 text-red-300 border border-red-800 font-mono">HIGH CONF</span>
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5 font-sans">
                  Prominent hull highlight & down-range acoustic shadow void (83% YOLOv8).
                </div>
              </button>

              <button
                onClick={() => {
                  onLoadDemoSample('corsican_02');
                  setShowDemoMenu(false);
                }}
                className="w-full text-left p-2 rounded hover:bg-[#111e38] transition-colors border border-transparent hover:border-[#1d335e]"
              >
                <div className="font-mono font-bold text-slate-100 flex items-center justify-between text-xs">
                  <span>Corsican-02 (Verified Anomaly)</span>
                  <span className="text-[9px] px-1 py-0.2 rounded bg-amber-950 text-amber-300 border border-amber-800 font-mono">TARGET</span>
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5 font-sans">
                  Verified structural anomaly matching ground-truth held-out label.
                </div>
              </button>

              <button
                onClick={() => {
                  onLoadDemoSample('artificial_reef_02');
                  setShowDemoMenu(false);
                }}
                className="w-full text-left p-2 rounded hover:bg-[#111e38] transition-colors border border-transparent hover:border-[#1d335e]"
              >
                <div className="font-mono font-bold text-slate-100 flex items-center justify-between text-xs">
                  <span>Artificial-Reef-02 (Clutter)</span>
                  <span className="text-[9px] px-1 py-0.2 rounded bg-slate-800 text-slate-300 border border-slate-700 font-mono">CLUTTER</span>
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5 font-sans">
                  Geological reef ridges for human-in-the-loop false positive rejection.
                </div>
              </button>

              <button
                onClick={() => {
                  onLoadDemoSample('survey_001');
                  setShowDemoMenu(false);
                }}
                className="w-full text-left p-2 rounded hover:bg-[#111e38] transition-colors border border-transparent hover:border-[#1d335e]"
              >
                <div className="font-mono font-bold text-slate-100 flex items-center justify-between text-xs">
                  <span>Survey-001 (Nav Log Ref)</span>
                  <span className="text-[9px] px-1 py-0.2 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-mono">GPS SYNC</span>
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5 font-sans">
                  Swath with towfish heading log for WGS-84 dead-reckoning projection.
                </div>
              </button>

              {onCustomUploadClick && (
                <div className="pt-1 border-t border-[#142240]">
                  <button
                    onClick={() => {
                      onCustomUploadClick();
                      setShowDemoMenu(false);
                    }}
                    className="w-full text-center py-1.5 rounded bg-[#0d172e] hover:bg-[#142347] text-cyan-300 text-[11px] font-mono font-bold border border-[#1e3463] transition-colors"
                  >
                    + INGEST CUSTOM SSS LOG...
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Model Identifier */}
        <div className="hidden lg:flex items-center gap-1 px-2 py-0.5 rounded bg-[#050a14] border border-[#142038] text-[11px] text-slate-400">
          <span>YOLOv8n-Baseline</span>
          <span className="text-[9px] text-cyan-400">(FP16)</span>
        </div>

        {/* Operational Disclaimer Modal Trigger */}
        <button
          onClick={() => setShowDisclaimer(!showDisclaimer)}
          className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-[#121e38] transition-colors"
          title="Scientific Scope & Honesty Details"
        >
          <Info className="w-3.5 h-3.5" />
        </button>

        {/* Sensor & Backend Link Telemetry */}
        <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded bg-[#050a14] border border-[#142038] text-[11px] text-emerald-400 font-mono font-bold">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
          <span>{backendStatus === 'online' ? 'LINK 200 OK' : 'LINK DOWN'}</span>
        </div>
      </div>

      {/* Scientific Disclaimer Modal */}
      {showDisclaimer && (
        <div className="absolute top-12 right-4 w-96 bg-[#0a1224] border border-[#1f3561] rounded p-4 shadow-2xl z-50 text-xs text-slate-300 font-sans">
          <div className="flex items-center gap-2 mb-2 pb-1.5 border-b border-[#142240]">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span className="font-mono font-bold text-xs text-slate-100 uppercase">Operational & Scientific Scope</span>
          </div>
          <p className="text-slate-300 text-xs leading-relaxed mb-2">
            SONAR-INTEL generates statistical acoustic anomaly candidates. Every proposal requires human-in-the-loop triage before logging.
          </p>
          <div className="p-2 rounded bg-[#050a14] border border-[#142038] text-[11px] font-mono text-slate-400 space-y-1 mb-3">
            <div>• Measured Val mAP@50: <strong className="text-cyan-400">6.45%</strong></div>
            <div>• Measured Frozen Test mAP@50: <strong className="text-emerald-400">10.48%</strong></div>
            <div>• Geolocation: Dead-reckoning from towfish logs</div>
          </div>
          <button
            onClick={() => setShowDisclaimer(false)}
            className="w-full py-1 text-center bg-[#132242] hover:bg-[#1a2e59] text-slate-200 rounded text-xs font-mono font-bold transition-colors border border-[#1f376b]"
          >
            CLOSE
          </button>
        </div>
      )}
    </header>
  );
};
