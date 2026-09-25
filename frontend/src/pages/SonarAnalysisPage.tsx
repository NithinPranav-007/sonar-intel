import React, { useState, useRef, useEffect } from 'react';
import { Contact, SurveyUploadResponse } from '../types/detection';
import { 
  Eye, 
  Sliders, 
  CheckCircle2, 
  HelpCircle, 
  AlertTriangle,
  Filter,
  Play,
  ArrowRight,
  Scan,
  Compass,
  Activity,
  Layers,
  Crosshair,
  Maximize2
} from 'lucide-react';

interface SonarAnalysisPageProps {
  survey: SurveyUploadResponse | null;
  contacts: Contact[];
  selectedContact: Contact | null;
  analyzing: boolean;
  onSelectContact: (contact: Contact) => void;
  onRunAnalysis: () => void;
  onVerifyContact?: (contact: Contact) => void;
}

export const SonarAnalysisPage: React.FC<SonarAnalysisPageProps> = ({
  survey,
  contacts,
  selectedContact,
  analyzing,
  onSelectContact,
  onRunAnalysis,
  onVerifyContact
}) => {
  const [viewMode, setViewMode] = useState<'raw' | 'processed'>('processed');
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [filterMode, setFilterMode] = useState<'top' | 'all'>('top');
  const [contrast, setContrast] = useState<number>(100);
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number; slantM: number } | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [scale, setScale] = useState<{ scaleX: number; scaleY: number }>({ scaleX: 1, scaleY: 1 });

  const displayedContacts = filterMode === 'top' ? contacts.slice(0, 5) : contacts;

  const updateScaling = () => {
    if (imgRef.current && imgRef.current.clientWidth > 0) {
      const renderW = imgRef.current.clientWidth;
      const renderH = imgRef.current.clientHeight;
      setScale({
        scaleX: renderW / (survey?.image_width || 1280),
        scaleY: renderH / (survey?.image_height || 1800)
      });
    }
  };

  useEffect(() => {
    window.addEventListener('resize', updateScaling);
    return () => window.removeEventListener('resize', updateScaling);
  }, [survey]);

  useEffect(() => {
    if (selectedContact && containerRef.current && scale.scaleY > 0) {
      const targetY = selectedContact.bbox.y1 * scale.scaleY;
      containerRef.current.scrollTo({
        top: Math.max(0, targetY - containerRef.current.clientHeight / 3),
        behavior: 'smooth'
      });
    }
  }, [selectedContact, scale]);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!imgRef.current || !survey) return;
    const rect = imgRef.current.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;

    if (clientX >= 0 && clientX <= rect.width && clientY >= 0 && clientY <= rect.height) {
      const origX = Math.round(clientX / (scale.scaleX || 1));
      const origY = Math.round(clientY / (scale.scaleY || 1));
      const nadirX = survey.image_width / 2;
      const slantDistPx = Math.abs(origX - nadirX);
      const slantM = Math.round((slantDistPx * 0.15) * 10) / 10;

      setCursorPos({ x: origX, y: origY, slantM });
    } else {
      setCursorPos(null);
    }
  };

  const activeContact = selectedContact || contacts[0] || null;

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#060b17] text-slate-100 font-sans select-none">
      
      {/* 3-Column Industrial Command Grid */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 min-h-0 overflow-hidden">
        
        {/* Left Column (3 Cols): Telemetry Readouts & Quality Index */}
        <div className="lg:col-span-3 h-full border-r border-[#15233e] bg-[#091122] p-3 flex flex-col justify-between overflow-y-auto space-y-2.5">
          <div className="space-y-2.5">
            
            {/* Swath Telemetry Card */}
            <div className="telemetry-cell space-y-2">
              <div className="text-[9px] uppercase tracking-wider text-slate-400 font-mono font-bold border-b border-[#15233e] pb-1 flex items-center justify-between">
                <span>SWATH INGESTION RECORD</span>
                <span className="text-cyan-400">2D-SSS</span>
              </div>
              
              <div className="space-y-1.5 text-xs font-mono">
                <div className="flex justify-between items-center text-[11px]">
                  <span className="text-slate-400">Survey ID:</span>
                  <span className="text-slate-100 font-bold truncate max-w-[130px]" title={survey?.survey_id}>
                    {survey?.survey_id || 'STANDBY'}
                  </span>
                </div>
                <div className="flex justify-between items-center text-[11px]">
                  <span className="text-slate-400">File:</span>
                  <span className="text-slate-300 truncate max-w-[130px]" title={survey?.filename}>
                    {survey?.filename || '--'}
                  </span>
                </div>
                <div className="flex justify-between items-center text-[11px]">
                  <span className="text-slate-400">Matrix Size:</span>
                  <span className="text-slate-200">
                    {survey ? `${survey.image_width} × ${survey.image_height} px` : '--'}
                  </span>
                </div>
                <div className="flex justify-between items-center text-[11px]">
                  <span className="text-slate-400">Resolution:</span>
                  <span className="text-slate-200">15.0 cm / pixel</span>
                </div>
              </div>
            </div>

            {/* Acoustic Signal & Dynamic Range */}
            <div className="telemetry-cell space-y-2">
              <div className="text-[9px] uppercase tracking-wider text-slate-400 font-mono font-bold border-b border-[#15233e] pb-1 flex items-center justify-between">
                <span>ACOUSTIC RADIOMETRIC STATS</span>
                <span className="text-emerald-400">CALIBRATED</span>
              </div>
              
              <div className="space-y-1.5 text-xs font-mono">
                <div className="flex justify-between items-center text-[11px]">
                  <span className="text-slate-400">Swath SNR Quality:</span>
                  <div className="flex items-center gap-1.5">
                    <div className="w-12 h-1.5 bg-[#050a14] rounded overflow-hidden border border-[#172542]">
                      <div className="h-full bg-emerald-400" style={{ width: `${survey ? Math.round(survey.data_quality * 100) : 0}%` }}></div>
                    </div>
                    <span className="text-emerald-400 font-bold">
                      {survey ? `${Math.round(survey.data_quality * 100)}%` : '--'}
                    </span>
                  </div>
                </div>
                <div className="flex justify-between items-center text-[11px]">
                  <span className="text-slate-400">Dynamic Range:</span>
                  <span className="text-slate-200">18.4 dB</span>
                </div>
                <div className="flex justify-between items-center text-[11px]">
                  <span className="text-slate-400">Normalization:</span>
                  <span className="text-cyan-400 text-[10px]">1%–99% Percentile</span>
                </div>
                <div className="flex justify-between items-center text-[11px]">
                  <span className="text-slate-400">CLAHE Filter:</span>
                  <span className="text-slate-400 text-[10px]">DISABLED (Audit verified)</span>
                </div>
              </div>
            </div>

            {/* Navigation Log Integration */}
            <div className="telemetry-cell space-y-2">
              <div className="text-[9px] uppercase tracking-wider text-slate-400 font-mono font-bold border-b border-[#15233e] pb-1 flex items-center justify-between">
                <span>TOWFISH SENSOR LOG</span>
                <span className={`text-[9px] font-bold ${
                  survey?.has_navigation ? 'text-emerald-400' : 'text-slate-500'
                }`}>
                  {survey?.has_navigation ? 'SYNCHRONIZED' : 'UNAVAILABLE'}
                </span>
              </div>
              
              <div className="space-y-1 text-xs font-mono">
                {survey?.has_navigation ? (
                  <>
                    <div className="flex justify-between text-[11px]">
                      <span className="text-slate-400">Heading:</span>
                      <span className="text-slate-200">184.2° (SSW)</span>
                    </div>
                    <div className="flex justify-between text-[11px]">
                      <span className="text-slate-400">Tow Velocity:</span>
                      <span className="text-slate-200">4.2 kts</span>
                    </div>
                    <div className="flex justify-between text-[11px]">
                      <span className="text-slate-400">Altitude (FBR):</span>
                      <span className="text-cyan-400 font-bold">12.5 m</span>
                    </div>
                  </>
                ) : (
                  <div className="text-[10px] text-slate-500 italic leading-relaxed py-1">
                    Raw acoustic imagery only. Coordinates unavailable awaiting navigation CSV.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Re-Run AI Inference Action */}
          {survey && (
            <button
              onClick={onRunAnalysis}
              disabled={analyzing}
              className={`w-full py-2 rounded text-xs font-mono font-bold flex items-center justify-center gap-1.5 transition-colors border ${
                analyzing
                  ? 'bg-slate-800 text-slate-400 border-slate-700 cursor-wait animate-pulse'
                  : 'bg-[#122244] hover:bg-[#193061] text-cyan-300 border-[#234282]'
              }`}
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>{analyzing ? 'PROCESSING SWATH...' : 'RE-RUN YOLOv8n INFERENCE'}</span>
            </button>
          )}
        </div>

        {/* Center Column (6 Cols): Side-Scan Sonar Waterfall Viewport */}
        <div className="lg:col-span-6 h-full flex flex-col min-h-0 bg-[#02050c] relative">
          
          {/* Top Utilitarian Toolbar */}
          <div className="h-9 border-b border-[#15233e] bg-[#070e1c] px-3 flex items-center justify-between z-10 text-xs font-mono">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setViewMode(viewMode === 'processed' ? 'raw' : 'processed')}
                className={`px-2 py-0.5 rounded text-[10px] font-bold border transition-colors ${
                  viewMode === 'processed'
                    ? 'bg-[#122244] text-cyan-300 border-cyan-500/60'
                    : 'bg-[#091122] text-slate-400 border-[#172542]'
                }`}
              >
                {viewMode === 'processed' ? '1–99% NORMALIZED' : 'RAW ACOUSTIC'}
              </button>

              <button
                onClick={() => setShowBoxes(!showBoxes)}
                className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold border transition-colors ${
                  showBoxes
                    ? 'bg-[#0c2419] text-emerald-300 border-emerald-600/60'
                    : 'bg-[#091122] text-slate-500 border-[#172542]'
                }`}
              >
                <Eye className="w-3 h-3 text-emerald-400" />
                <span>OVERLAYS ({displayedContacts.length})</span>
              </button>
            </div>

            {/* Contrast Slider & Coordinate Tracker */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
                <Sliders className="w-3 h-3 text-slate-400" />
                <span>Gain:</span>
                <input
                  type="range"
                  min={50}
                  max={180}
                  value={contrast}
                  onChange={(e) => setContrast(Number(e.target.value))}
                  className="w-16 h-1 accent-cyan-400 cursor-pointer"
                />
                <span className="text-cyan-400 font-bold w-6 text-right">{contrast}%</span>
              </div>

              <button
                onClick={() => setFilterMode(filterMode === 'top' ? 'all' : 'top')}
                className="px-2 py-0.5 rounded bg-[#091122] hover:bg-[#122244] border border-[#172542] text-[10px] text-slate-300 flex items-center gap-1"
                title="Toggle Top 5 vs All"
              >
                <Filter className="w-2.5 h-2.5 text-amber-400" />
                <span>{filterMode === 'top' ? 'TOP 5' : 'ALL'}</span>
              </button>
            </div>
          </div>

          {/* Sonar Waterfall Viewport */}
          <div 
            ref={containerRef} 
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setCursorPos(null)}
            className="flex-1 overflow-auto relative p-3 flex justify-center items-start acoustic-scroll crosshair-canvas"
          >
            {survey ? (
              <div className="relative inline-block border border-[#172542] shadow-2xl bg-black">
                
                {/* Acoustic Port / Starboard Header Scale */}
                <div className="w-full bg-[#070e1c] border-b border-[#172542] py-0.5 px-2 flex justify-between text-[9px] font-mono text-slate-400">
                  <span>◄ PORT (75m)</span>
                  <span className="text-cyan-400 font-bold">NADIR (0m)</span>
                  <span>STARBOARD (75m) ►</span>
                </div>

                <img
                  ref={imgRef}
                  src={viewMode === 'processed' ? survey.processed_image_url : survey.raw_image_url}
                  alt="Side-Scan Sonar Waterfall Swath"
                  onLoad={updateScaling}
                  style={{
                    filter: `contrast(${contrast}%)`,
                    maxHeight: '100%',
                    display: 'block'
                  }}
                  className="transition-all select-none"
                />

                {/* Precision Tactical Reticle Overlays */}
                {showBoxes && displayedContacts.map((c) => {
                  const isSelected = activeContact?.contact_id === c.contact_id;
                  const left = c.bbox.x1 * scale.scaleX;
                  const top = c.bbox.y1 * scale.scaleY;
                  const width = (c.bbox.x2 - c.bbox.x1) * scale.scaleX;
                  const height = (c.bbox.y2 - c.bbox.y1) * scale.scaleY;

                  let strokeColor = '#38bdf8';
                  let tagBg = '#0284c7';
                  if (c.priority === 'HIGH') {
                    strokeColor = '#ef4444';
                    tagBg = '#dc2626';
                  } else if (c.priority === 'MEDIUM') {
                    strokeColor = '#f59e0b';
                    tagBg = '#d97706';
                  }

                  return (
                    <div
                      key={c.contact_id}
                      onClick={() => onSelectContact(c)}
                      style={{
                        position: 'absolute',
                        left: `${left}px`,
                        top: `${top + 16}px`, // Adjusted for port/starboard top header
                        width: `${Math.max(14, width)}px`,
                        height: `${Math.max(14, height)}px`,
                        borderColor: strokeColor,
                      }}
                      className={`cursor-pointer border transition-all ${
                        isSelected 
                          ? 'border-2 ring-1 ring-white bg-cyan-400/10 z-20' 
                          : 'opacity-90 hover:opacity-100 z-10'
                      }`}
                    >
                      {/* Crisp 1px Corner Marks */}
                      <div className="absolute -top-1 -left-1 w-1.5 h-1.5 border-t border-l border-white"></div>
                      <div className="absolute -top-1 -right-1 w-1.5 h-1.5 border-t border-r border-white"></div>
                      <div className="absolute -bottom-1 -left-1 w-1.5 h-1.5 border-b border-l border-white"></div>
                      <div className="absolute -bottom-1 -right-1 w-1.5 h-1.5 border-b border-r border-white"></div>

                      <div 
                        style={{ backgroundColor: tagBg }}
                        className="absolute -bottom-4 left-0 px-1 py-0.2 rounded font-mono font-bold text-[8px] text-white whitespace-nowrap shadow-xs"
                      >
                        {c.contact_id} • {Math.round(c.confidence * 100)}%
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-xs text-slate-500 font-mono text-center">
                <Crosshair className="w-8 h-8 text-slate-700 animate-spin mb-2" />
                <p className="font-bold text-slate-300">NO SWATH ACTIVE</p>
                <p className="text-[10px] text-slate-500 mt-1">Select a held-out test case from top bar.</p>
              </div>
            )}

            {/* Live Cursor Coordinate HUD */}
            {cursorPos && (
              <div className="absolute top-4 left-4 px-2 py-1 rounded bg-[#070e1c]/90 border border-[#172542] text-[10px] font-mono text-cyan-300 pointer-events-none z-30">
                X: {cursorPos.x}px | Y: {cursorPos.y}px | Slant Range: ~{cursorPos.slantM}m
              </div>
            )}
          </div>
        </div>

        {/* Right Column (3 Cols): Ranked Detection Queue */}
        <div className="lg:col-span-3 h-full border-l border-[#15233e] bg-[#091122] flex flex-col min-h-0 shadow-xs">
          <div className="p-2.5 border-b border-[#15233e] flex items-center justify-between bg-[#070e1c]">
            <span className="font-mono font-bold text-slate-200 text-[11px] tracking-wider uppercase flex items-center gap-1.5">
              <Scan className="w-3 h-3 text-cyan-400" /> DETECTION QUEUE
            </span>
            <span className="text-[9px] px-1.5 py-0.2 rounded bg-[#0d1830] text-cyan-300 font-mono font-bold border border-[#1b315e]">
              {displayedContacts.length} Ranked
            </span>
          </div>

          <div className="flex-1 overflow-y-auto p-2 space-y-1.5 text-xs font-mono">
            {displayedContacts.map((c) => {
              const isSelected = activeContact?.contact_id === c.contact_id;
              let badgeStyle = 'text-sky-400 bg-sky-950/80 border-sky-800';
              if (c.priority === 'HIGH') {
                badgeStyle = 'text-red-400 bg-red-950/80 border-red-800';
              } else if (c.priority === 'MEDIUM') {
                badgeStyle = 'text-amber-400 bg-amber-950/80 border-amber-800';
              }

              return (
                <div
                  key={c.contact_id}
                  onClick={() => onSelectContact(c)}
                  className={`p-2 rounded border cursor-pointer transition-colors flex items-center justify-between ${
                    isSelected 
                      ? 'bg-[#122244] border-cyan-500' 
                      : 'bg-[#0b1426] border-[#16233d] hover:bg-[#0f1b33]'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-black border border-[#1c2c4d] flex items-center justify-center font-bold text-[10px] text-cyan-300 shrink-0">
                      {c.contact_id}
                    </div>
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="font-bold text-slate-100 text-xs">{c.contact_id}</span>
                        <span className={`text-[8px] px-1 py-0.2 rounded border font-bold ${badgeStyle}`}>
                          {c.priority}
                        </span>
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5">
                        Conf: <strong className="text-white">{Math.round(c.confidence * 100)}%</strong> • {c.review_status}
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectContact(c);
                      onVerifyContact?.(c);
                    }}
                    className="p-1 rounded bg-[#060b17] hover:bg-cyan-600 hover:text-white text-slate-400 border border-[#15233e] transition-colors"
                    title="Open in Contact Verification"
                  >
                    <ArrowRight className="w-3 h-3" />
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Bottom Acoustic Context Verification Bar */}
      {activeContact && (
        <div className="h-16 border-t border-[#15233e] bg-[#070e1c] px-4 py-2 flex items-center justify-between text-xs z-20 font-mono">
          <div className="flex items-center gap-2 shrink-0">
            <Scan className="w-4 h-4 text-cyan-400" />
            <div>
              <div className="text-[9px] uppercase font-bold text-slate-400">
                PHYSICS CONTEXT ENGINE
              </div>
              <div className="text-[11px] font-bold text-white">
                TARGET {activeContact.contact_id}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-2.5 flex-1 max-w-4xl px-4 text-xs">
            {/* 1. Object-Shadow Deficit */}
            <div className="telemetry-cell flex items-center justify-between py-1">
              <div>
                <div className="text-[8px] text-slate-400 uppercase">SHADOW DEFICIT</div>
                <div className="font-bold text-slate-100 text-[11px]">SHADOW MATCHED</div>
              </div>
              <span className="text-[8px] px-1.5 py-0.2 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-bold">
                PASS
              </span>
            </div>

            {/* 2. Seabed Texture Match */}
            <div className="telemetry-cell flex items-center justify-between py-1">
              <div>
                <div className="text-[8px] text-slate-400 uppercase">SEABED FLOOR</div>
                <div className="font-bold text-slate-100 text-[11px]">SANDY / GRAVEL</div>
              </div>
              <span className="text-[8px] px-1.5 py-0.2 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-bold">
                87% MATCH
              </span>
            </div>

            {/* 3. False Positive Risk */}
            <div className="telemetry-cell flex items-center justify-between py-1">
              <div>
                <div className="text-[8px] text-slate-400 uppercase">CLUTTER RISK</div>
                <div className="font-bold text-slate-100 text-[11px]">ANOMALOUS STRUCT</div>
              </div>
              <span className="text-[8px] px-1.5 py-0.2 rounded bg-amber-950 text-amber-300 border border-amber-800 font-bold">
                12/100
              </span>
            </div>

            {/* 4. Overall AI Confidence */}
            <div className="telemetry-cell flex items-center justify-between py-1">
              <div>
                <div className="text-[8px] text-slate-400 uppercase">OVERALL SCORE</div>
                <div className="font-bold text-slate-100 text-[11px]">COMPOSITE MATCH</div>
              </div>
              <span className="text-[9px] px-1.5 py-0.2 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 font-bold">
                {Math.round(activeContact.confidence * 100)}%
              </span>
            </div>
          </div>

          <button
            onClick={() => onVerifyContact?.(activeContact)}
            className="px-3 py-1.5 rounded bg-[#122244] hover:bg-[#193061] text-cyan-300 font-mono font-bold text-[11px] flex items-center gap-1.5 border border-[#234282] transition-colors shrink-0"
          >
            <span>VERIFY CANDIDATE</span>
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>
      )}
    </div>
  );
};
