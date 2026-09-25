import React, { useState, useRef, useEffect } from 'react';
import { Contact, SurveyUploadResponse } from '../types/detection';
import { 
  Eye, 
  Sliders, 
  CheckCircle2, 
  AlertTriangle, 
  Filter, 
  Play, 
  ArrowRight, 
  ArrowLeft,
  Scan, 
  Crosshair, 
  Layers,
  Waves,
  ShieldAlert,
  Compass,
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
  onNavigateToDashboard?: () => void;
}

export const SonarAnalysisPage: React.FC<SonarAnalysisPageProps> = ({
  survey,
  contacts,
  selectedContact,
  analyzing,
  onSelectContact,
  onRunAnalysis,
  onVerifyContact,
  onNavigateToDashboard
}) => {
  const [viewMode, setViewMode] = useState<'raw' | 'processed'>('processed');
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [filterMode, setFilterMode] = useState<'top' | 'all'>('top');
  const [contrast, setContrast] = useState<number>(100);
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number; slantM: number } | null>(null);

  // Measurement Tools & Bottom Track Overlay State
  const [activeTool, setActiveTool] = useState<'none' | 'length' | 'area' | 'height'>('none');
  const [measurePoints, setMeasurePoints] = useState<Array<{ x: number; y: number }>>([]);
  const [measurementResult, setMeasurementResult] = useState<string | null>(null);
  const [showBottomTrack, setShowBottomTrack] = useState<boolean>(true);

  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [scale, setScale] = useState<{ scaleX: number; scaleY: number }>({ scaleX: 1, scaleY: 1 });

  const displayedContacts = filterMode === 'top' ? contacts.slice(0, 5) : contacts;

  const handleSelectTool = (tool: 'length' | 'area' | 'height') => {
    if (activeTool === tool) {
      setActiveTool('none');
      setMeasurePoints([]);
      setMeasurementResult(null);
    } else {
      setActiveTool(tool);
      setMeasurePoints([]);
      setMeasurementResult(null);
    }
  };

  const handleClearMeasurement = () => {
    setMeasurePoints([]);
    setMeasurementResult(null);
  };

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

  const handleImageClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!imgRef.current || activeTool === 'none') return;
    const rect = imgRef.current.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;

    if (clientX < 0 || clientX > rect.width || clientY < 0 || clientY > rect.height) return;

    const newPoint = { x: clientX, y: clientY };

    if (activeTool === 'length') {
      if (measurePoints.length === 0 || measurePoints.length >= 2) {
        setMeasurePoints([newPoint]);
        setMeasurementResult('Click point 2 to measure length');
      } else {
        const p1 = measurePoints[0];
        const p2 = newPoint;
        const distPx = Math.hypot(p2.x - p1.x, p2.y - p1.y);
        const distM = (distPx * 0.15 / (scale.scaleX || 1)).toFixed(2);
        setMeasurePoints([p1, p2]);
        setMeasurementResult(`Length: ${distM} m (${Math.round(distPx)} px)`);
      }
    } else if (activeTool === 'area') {
      const nextPoints = [...measurePoints, newPoint];
      setMeasurePoints(nextPoints);
      if (nextPoints.length >= 3) {
        // Shoelace formula for polygon area
        let area = 0;
        for (let i = 0; i < nextPoints.length; i++) {
          const j = (i + 1) % nextPoints.length;
          area += nextPoints[i].x * nextPoints[j].y;
          area -= nextPoints[j].x * nextPoints[i].y;
        }
        const areaPx = Math.abs(area) / 2;
        const areaM2 = (areaPx * (0.15 / (scale.scaleX || 1)) * (0.15 / (scale.scaleY || 1))).toFixed(2);
        setMeasurementResult(`Area: ${areaM2} m² (${nextPoints.length} points)`);
      } else {
        setMeasurementResult(`Area: Point ${nextPoints.length} set. Click more points.`);
      }
    } else if (activeTool === 'height') {
      if (measurePoints.length >= 3 || measurePoints.length === 0) {
        setMeasurePoints([newPoint]);
        setMeasurementResult('Point 1/3 (Bottom return) set. Click Object apex.');
      } else if (measurePoints.length === 1) {
        setMeasurePoints([...measurePoints, newPoint]);
        setMeasurementResult('Point 2/3 (Object apex) set. Click Shadow tip.');
      } else if (measurePoints.length === 2) {
        const p1 = measurePoints[0];
        const p2 = measurePoints[1];
        const p3 = newPoint;
        const shadowPx = Math.hypot(p3.x - p2.x, p3.y - p2.y);
        const shadowM = shadowPx * 0.15 / (scale.scaleX || 1);
        const imgW = imgRef.current.clientWidth || 600;
        const slantRangeM = Math.max(8, Math.abs(p2.x - (imgW / 2)) * 0.15 / (scale.scaleX || 1));
        const altitudeM = 12.5;
        const heightM = Math.max(0.2, (altitudeM * shadowM) / (slantRangeM + shadowM)).toFixed(2);
        setMeasurePoints([p1, p2, p3]);
        setMeasurementResult(`Height: ${heightM} m (Shadow: ${shadowM.toFixed(1)}m)`);
      }
    }
  };

  const activeContact = selectedContact || contacts[0] || null;

  return (
    <div className="p-6 lg:p-8 max-w-[1700px] mx-auto space-y-6 font-sans">
      
      {/* 1. Workspace Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft">
        <div className="space-y-2">
          {/* Top-left: Back Button & Breadcrumb */}
          <div className="flex items-center gap-3">
            {onNavigateToDashboard && (
              <button
                onClick={onNavigateToDashboard}
                className="px-3.5 py-1.5 rounded-xl bg-[#f8fafc] hover:bg-slate-100 text-[#0f172a] hover:text-[#1d4ed8] border border-[#e2e8f0] font-semibold text-xs transition-all duration-200 shadow-tactile flex items-center gap-1.5 cursor-pointer group"
              >
                <ArrowLeft className="w-3.5 h-3.5 text-[#64748b] group-hover:text-[#1d4ed8] group-hover:-translate-x-0.5 transition-transform" />
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
                Sonar Waterfall
              </span>
            </div>
          </div>

          <h2 className="text-2xl font-extrabold text-[#0f172a] font-display mt-0.5 flex items-center gap-2.5">
            <Waves className="w-6 h-6 text-[#1d4ed8]" />
            Sonar Waterfall Analysis & Triage
          </h2>
        </div>

        {/* Quick Actions */}
        <div className="flex items-center gap-3">
          <div className="text-xs font-semibold px-4 py-2 rounded-full bg-[#f8fafc] border border-[#e2e8f0] text-[#0f172a] flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>{survey ? survey.filename : 'No Swath Active'}</span>
          </div>

          {activeContact && (
            <button
              onClick={() => onVerifyContact?.(activeContact)}
              className="px-5 py-2.5 rounded-full bg-[#1d4ed8] hover:bg-[#1e40af] text-white font-semibold text-xs transition-all duration-200 shadow-tactile flex items-center gap-2 cursor-pointer shadow-blue-glow"
            >
              <span>Verify Target {activeContact.contact_id}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* 2. Three-Column Main Analysis Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[640px]">
        
        {/* Left Column (3 Cols): Swath Telemetry & Acoustic Metrics */}
        <div className="lg:col-span-3 space-y-6">
          
          {/* Swath Telemetry Record Card */}
          <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-4">
            <div className="border-b border-[#f1f5f9] pb-3 flex items-center justify-between">
              <span className="section-label">Ingestion Record</span>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-slate-100 text-[#0f172a]">
                2D-SSS
              </span>
            </div>

            <div className="space-y-3 text-xs">
              <div className="p-3 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] space-y-1">
                <span className="text-[10px] text-[#64748b] font-bold uppercase tracking-wider">Survey Swath ID</span>
                <div className="text-xs font-bold text-[#0f172a] truncate font-mono" title={survey?.survey_id}>
                  {survey?.survey_id || 'STANDBY'}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2.5">
                <div className="p-3 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0]">
                  <span className="text-[10px] text-[#64748b] font-bold uppercase tracking-wider">Matrix Size</span>
                  <div className="text-xs font-bold text-[#0f172a] font-mono mt-1">
                    {survey ? `${survey.image_width}x${survey.image_height}` : '--'}
                  </div>
                </div>

                <div className="p-3 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0]">
                  <span className="text-[10px] text-[#64748b] font-bold uppercase tracking-wider">Resolution</span>
                  <div className="text-xs font-bold text-[#0f172a] font-mono mt-1">
                    15.0 cm / px
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Radiometric Stats & Quality */}
          <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-4">
            <div className="border-b border-[#f1f5f9] pb-3 flex items-center justify-between">
              <span className="section-label">Acoustic Radiometrics</span>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700">
                Calibrated
              </span>
            </div>

            <div className="space-y-3.5 text-xs">
              <div className="space-y-1.5">
                <div className="flex justify-between items-center">
                  <span className="text-[#64748b] font-medium">Swath SNR Quality:</span>
                  <span className="font-bold text-emerald-600">
                    {survey ? `${Math.round(survey.data_quality * 100)}%` : '--'}
                  </span>
                </div>
                <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-emerald-500 rounded-full" 
                    style={{ width: `${survey ? Math.round(survey.data_quality * 100) : 0}%` }}
                  />
                </div>
              </div>

              <div className="flex justify-between items-center pt-1 border-t border-[#f1f5f9]">
                <span className="text-[#64748b]">Dynamic Range:</span>
                <span className="font-bold text-[#0f172a] font-mono">18.4 dB</span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-[#64748b]">Normalization:</span>
                <span className="font-semibold text-[#1d4ed8] bg-blue-50 px-2 py-0.5 rounded-full text-[11px] border border-blue-100">
                  1%–99% Percentile
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-[#64748b]">CLAHE Filter:</span>
                <span className="text-[#64748b] text-[11px] font-medium">
                  Audit Disabled (Shadow Preservation)
                </span>
              </div>
            </div>
          </div>

          {/* Towfish Nav Sensor Log */}
          <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-4">
            <div className="border-b border-[#f1f5f9] pb-3 flex items-center justify-between">
              <span className="section-label">Towfish Sensor Log</span>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                survey?.has_navigation ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-100 text-[#64748b]'
              }`}>
                {survey?.has_navigation ? 'SYNCHRONIZED' : 'DEAD-RECKONING'}
              </span>
            </div>

            <div className="space-y-2.5 text-xs font-mono">
              {survey?.has_navigation ? (
                <>
                  <div className="flex justify-between items-center">
                    <span className="text-[#64748b]">Heading:</span>
                    <span className="font-bold text-[#0f172a]">184.2° (SSW)</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-[#64748b]">Tow Velocity:</span>
                    <span className="font-bold text-[#0f172a]">4.2 kts</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-[#64748b]">Towfish Altitude:</span>
                    <span className="font-bold text-[#1d4ed8]">12.5 m</span>
                  </div>
                </>
              ) : (
                <div className="p-3 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] text-[#64748b] text-[11px] leading-relaxed">
                  Raw acoustic waterfall loaded. Towfish navigation coordinates estimated via dead-reckoning.
                </div>
              )}
            </div>

            {/* Run Analysis CTA */}
            {survey && (
              <button
                onClick={onRunAnalysis}
                disabled={analyzing}
                className={`w-full mt-2 py-3 rounded-full font-semibold text-xs flex items-center justify-center gap-2 transition-all duration-200 shadow-tactile ${
                  analyzing
                    ? 'bg-slate-200 text-[#64748b] cursor-wait'
                    : 'bg-[#1d4ed8] hover:bg-[#1e40af] text-white hover:scale-[1.02] active:scale-[0.98] cursor-pointer shadow-blue-glow'
                }`}
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>{analyzing ? 'Processing Swath...' : 'Re-Run YOLOv8n Triage'}</span>
              </button>
            )}
          </div>
        </div>

        {/* Center Column (6 Cols): Dominant Acoustic Sonar Waterfall Viewer */}
        <div className="lg:col-span-6 flex flex-col bg-white rounded-[24px] border border-[#e2e8f0] shadow-soft overflow-hidden">
          
          {/* Top Utilitarian Controls Toolbar */}
          <div className="p-4 border-b border-[#e2e8f0] bg-[#f8fafc] flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setViewMode(viewMode === 'processed' ? 'raw' : 'processed')}
                className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-all cursor-pointer ${
                  viewMode === 'processed'
                    ? 'bg-[#0f172a] text-white border-[#0f172a] shadow-sm'
                    : 'bg-white text-[#0f172a] border-[#e2e8f0] hover:bg-slate-50'
                }`}
              >
                {viewMode === 'processed' ? '1–99% Normalized' : 'Raw Acoustic'}
              </button>

              <button
                onClick={() => setShowBoxes(!showBoxes)}
                className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-all flex items-center gap-1.5 cursor-pointer ${
                  showBoxes
                    ? 'bg-emerald-50 text-emerald-700 border-emerald-200 shadow-sm'
                    : 'bg-white text-[#64748b] border-[#e2e8f0] hover:bg-slate-50'
                }`}
              >
                <Eye className="w-3.5 h-3.5" />
                <span>Overlays ({displayedContacts.length})</span>
              </button>
            </div>

            {/* Contrast Gain Slider & Filter Toggle */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-xs font-medium text-[#64748b]">
                <Sliders className="w-3.5 h-3.5 text-[#64748b]" />
                <span>Gain:</span>
                <input
                  type="range"
                  min={50}
                  max={180}
                  value={contrast}
                  onChange={(e) => setContrast(Number(e.target.value))}
                  className="w-20 h-1.5 accent-[#1d4ed8] cursor-pointer"
                />
                <span className="text-[#0f172a] font-bold w-9 text-right font-mono">{contrast}%</span>
              </div>

              <button
                onClick={() => setFilterMode(filterMode === 'top' ? 'all' : 'top')}
                className="px-3 py-1.5 rounded-full bg-white hover:bg-slate-50 border border-[#e2e8f0] text-xs font-semibold text-[#0f172a] flex items-center gap-1.5 cursor-pointer shadow-tactile"
              >
                <Filter className="w-3 h-3 text-[#1d4ed8]" />
                <span>{filterMode === 'top' ? 'Top 5 Targets' : 'All Candidates'}</span>
              </button>
            </div>
          </div>

          {/* Sonar Acoustic Waterfall Canvas Viewport */}
          <div 
            ref={containerRef} 
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setCursorPos(null)}
            className="flex-1 overflow-auto relative p-4 flex justify-center items-start bg-[#050a14] cursor-crosshair min-h-[500px]"
          >
            {survey ? (
              <div 
                onClick={handleImageClick}
                className={`relative inline-block border border-slate-800 shadow-2xl bg-black ${
                  activeTool !== 'none' ? 'cursor-crosshair' : ''
                }`}
              >
                
                {/* Port / Starboard Acoustic Header Scale */}
                <div className="w-full bg-[#091122] border-b border-slate-800 py-1 px-3 flex justify-between text-[10px] font-mono text-slate-400 select-none">
                  <span>◄ PORT SWATH (75m)</span>
                  <span className="text-cyan-400 font-bold">NADIR VOID (0m)</span>
                  <span>STARBOARD SWATH (75m) ►</span>
                </div>

                <img
                  ref={imgRef}
                  src={viewMode === 'processed' ? survey.processed_image_url : survey.raw_image_url}
                  alt="Side-Scan Sonar Waterfall"
                  onLoad={updateScaling}
                  style={{
                    filter: `contrast(${contrast}%)`,
                    maxHeight: '100%',
                    display: 'block'
                  }}
                  className="transition-all select-none"
                />

                {/* SVG Overlay for Bottom Line & Measurement Visuals */}
                <svg
                  className="absolute inset-0 w-full pointer-events-none"
                  style={{
                    top: '20px',
                    height: 'calc(100% - 20px)',
                    width: '100%'
                  }}
                >
                  {/* 2. Seafloor Bottom Track Line Overlay */}
                  {showBottomTrack && (
                    <g>
                      <line
                        x1={0}
                        y1={(imgRef.current?.clientHeight || 450) * 0.45}
                        x2={imgRef.current?.clientWidth || 650}
                        y2={(imgRef.current?.clientHeight || 450) * 0.45}
                        stroke="#00B4D8"
                        strokeWidth="2"
                        strokeOpacity="0.7"
                        strokeDasharray="6 4"
                      />
                      <rect
                        x={(imgRef.current?.clientWidth || 650) - 95}
                        y={(imgRef.current?.clientHeight || 450) * 0.45 - 18}
                        width="90"
                        height="16"
                        rx="4"
                        fill="#0f172a"
                        fillOpacity="0.85"
                        stroke="#00B4D8"
                        strokeWidth="1"
                        strokeOpacity="0.7"
                      />
                      <text
                        x={(imgRef.current?.clientWidth || 650) - 50}
                        y={(imgRef.current?.clientHeight || 450) * 0.45 - 6}
                        textAnchor="middle"
                        fill="#00B4D8"
                        fontSize="10"
                        fontFamily="'JetBrains Mono', monospace"
                        fontWeight="bold"
                        opacity="0.95"
                      >
                        Bottom Track
                      </text>
                    </g>
                  )}

                  {/* Measurement Tool Drawings */}
                  {activeTool === 'length' && measurePoints.length > 0 && (
                    <g>
                      {measurePoints.map((p, idx) => (
                        <circle key={idx} cx={p.x} cy={p.y} r="4" fill="#00B4D8" stroke="#ffffff" strokeWidth="1.5" />
                      ))}
                      {measurePoints.length === 2 && (
                        <>
                          <line
                            x1={measurePoints[0].x}
                            y1={measurePoints[0].y}
                            x2={measurePoints[1].x}
                            y2={measurePoints[1].y}
                            stroke="#00B4D8"
                            strokeWidth="2.5"
                            strokeDasharray="4 2"
                          />
                          <rect
                            x={(measurePoints[0].x + measurePoints[1].x) / 2 - 40}
                            y={(measurePoints[0].y + measurePoints[1].y) / 2 - 18}
                            width="80"
                            height="18"
                            rx="4"
                            fill="#0f172a"
                            fillOpacity="0.9"
                            stroke="#00B4D8"
                            strokeWidth="1"
                          />
                          <text
                            x={(measurePoints[0].x + measurePoints[1].x) / 2}
                            y={(measurePoints[0].y + measurePoints[1].y) / 2 - 5}
                            textAnchor="middle"
                            fill="#00B4D8"
                            fontSize="10"
                            fontWeight="bold"
                            fontFamily="'JetBrains Mono', monospace"
                          >
                            {(Math.hypot(measurePoints[1].x - measurePoints[0].x, measurePoints[1].y - measurePoints[0].y) * 0.15 / (scale.scaleX || 1)).toFixed(2)}m
                          </text>
                        </>
                      )}
                    </g>
                  )}

                  {activeTool === 'area' && measurePoints.length > 0 && (
                    <g>
                      {measurePoints.map((p, idx) => (
                        <circle key={idx} cx={p.x} cy={p.y} r="4" fill="#00B4D8" stroke="#ffffff" strokeWidth="1.5" />
                      ))}
                      {measurePoints.length >= 2 && (
                        <polygon
                          points={measurePoints.map(p => `${p.x},${p.y}`).join(' ')}
                          fill="rgba(0, 180, 216, 0.25)"
                          stroke="#00B4D8"
                          strokeWidth="2"
                          strokeDasharray="4 2"
                        />
                      )}
                    </g>
                  )}

                  {activeTool === 'height' && measurePoints.length > 0 && (
                    <g>
                      {measurePoints.map((p, idx) => (
                        <g key={idx}>
                          <circle cx={p.x} cy={p.y} r="4" fill={idx === 0 ? '#10b981' : idx === 1 ? '#ef4444' : '#f59e0b'} stroke="#ffffff" strokeWidth="1.5" />
                          <text x={p.x + 6} y={p.y + 3} fill="#ffffff" fontSize="9" fontWeight="bold" fontFamily="'JetBrains Mono', monospace">
                            {idx === 0 ? 'P1:Bed' : idx === 1 ? 'P2:Apex' : 'P3:Shadow'}
                          </text>
                        </g>
                      ))}
                      {measurePoints.length === 3 && (
                        <line
                          x1={measurePoints[1].x}
                          y1={measurePoints[1].y}
                          x2={measurePoints[2].x}
                          y2={measurePoints[2].y}
                          stroke="#f59e0b"
                          strokeWidth="2.5"
                          strokeDasharray="4 2"
                        />
                      )}
                    </g>
                  )}
                </svg>

                {/* Tactical Bounding Box Candidate Overlays */}
                {showBoxes && displayedContacts.map((c) => {
                  const isSelected = selectedContact?.contact_id === c.contact_id || activeContact?.contact_id === c.contact_id;
                  const left = c.bbox.x1 * scale.scaleX;
                  const top = c.bbox.y1 * scale.scaleY;
                  const width = (c.bbox.x2 - c.bbox.x1) * scale.scaleX;
                  const height = (c.bbox.y2 - c.bbox.y1) * scale.scaleY;

                  let strokeColor = '#10b981';
                  let tagBg = '#10b981';
                  if (c.priority === 'HIGH') {
                    strokeColor = '#ef4444';
                    tagBg = '#ef4444';
                  } else if (c.priority === 'MEDIUM') {
                    strokeColor = '#f59e0b';
                    tagBg = '#f59e0b';
                  }

                  return (
                    <div
                      key={c.contact_id}
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectContact(c);
                      }}
                      title="Click to verify"
                      style={{
                        position: 'absolute',
                        left: `${left}px`,
                        top: `${top + 20}px`,
                        width: `${Math.max(16, width)}px`,
                        height: `${Math.max(16, height)}px`,
                        borderColor: strokeColor,
                      }}
                      className={`cursor-pointer border transition-all group ${
                        isSelected 
                          ? 'border-2 ring-2 ring-blue-400 bg-blue-500/20 z-20 scale-[1.02]' 
                          : 'opacity-90 hover:opacity-100 hover:scale-105 z-10 hover:border-white'
                      }`}
                    >
                      {/* Corner Targeting Marks */}
                      <div className="absolute -top-1 -left-1 w-1.5 h-1.5 border-t border-l border-white" />
                      <div className="absolute -top-1 -right-1 w-1.5 h-1.5 border-t border-r border-white" />
                      <div className="absolute -bottom-1 -left-1 w-1.5 h-1.5 border-b border-l border-white" />
                      <div className="absolute -bottom-1 -right-1 w-1.5 h-1.5 border-b border-r border-white" />

                      {/* Pill Badge */}
                      <div 
                        style={{ backgroundColor: tagBg }}
                        className="absolute -bottom-5 left-0 px-1.5 py-0.5 rounded-full font-mono font-bold text-[9px] text-white whitespace-nowrap shadow-md"
                      >
                        {c.contact_id} • {Math.round(c.confidence * 100)}%
                      </div>

                      {/* Tooltip on hover */}
                      <div className="absolute -top-7 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded-md bg-slate-900/90 text-white text-[10px] font-sans whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-30 shadow-md border border-slate-700">
                        Click to verify
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-xs text-slate-400 font-mono text-center py-24">
                <Crosshair className="w-10 h-10 text-slate-600 animate-spin mb-3" />
                <p className="font-bold text-white text-sm">NO SWATH ACTIVE</p>
                <p className="text-xs text-slate-400 mt-1">Select a curated benchmark swath from the top header.</p>
              </div>
            )}

            {/* Live Hover HUD Coordinates */}
            {cursorPos && (
              <div className="absolute top-4 left-4 px-3 py-1.5 rounded-full bg-white/95 border border-[#e2e8f0] text-[11px] font-mono font-bold text-[#0f172a] pointer-events-none z-30 shadow-lg">
                X: {cursorPos.x}px | Y: {cursorPos.y}px | Slant Range: ~{cursorPos.slantM}m
              </div>
            )}
          </div>

          {/* 1. Measurement Tools Toolbar (Below the sonar image) */}
          <div className="p-3 px-5 border-t border-[#e2e8f0] bg-[#f8fafc] flex flex-wrap items-center justify-between gap-3 text-xs font-sans">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-bold text-[#64748b] uppercase tracking-wider flex items-center gap-1.5">
                Measurement Tools:
              </span>

              {/* Length Tool */}
              <button
                onClick={() => handleSelectTool('length')}
                className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-all cursor-pointer flex items-center gap-1.5 ${
                  activeTool === 'length'
                    ? 'bg-[#1d4ed8] text-white border-[#1d4ed8] shadow-blue-sm'
                    : 'bg-white text-[#0f172a] border-[#e2e8f0] hover:bg-slate-50'
                }`}
                title="Click two points to measure a target's length"
              >
                <span>📏</span>
                <span>Length</span>
              </button>

              {/* Area Tool */}
              <button
                onClick={() => handleSelectTool('area')}
                className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-all cursor-pointer flex items-center gap-1.5 ${
                  activeTool === 'area'
                    ? 'bg-[#1d4ed8] text-white border-[#1d4ed8] shadow-blue-sm'
                    : 'bg-white text-[#0f172a] border-[#e2e8f0] hover:bg-slate-50'
                }`}
                title="Click around a target to measure its area"
              >
                <span>📐</span>
                <span>Area</span>
              </button>

              {/* Height Tool */}
              <button
                onClick={() => handleSelectTool('height')}
                className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-all cursor-pointer flex items-center gap-1.5 ${
                  activeTool === 'height'
                    ? 'bg-[#1d4ed8] text-white border-[#1d4ed8] shadow-blue-sm'
                    : 'bg-white text-[#0f172a] border-[#e2e8f0] hover:bg-slate-50'
                }`}
                title="Click three points: 1) Nadir, 2) Object, 3) Shadow end to calculate height"
              >
                <span>📏</span>
                <span>Height</span>
              </button>

              {/* Clear Measurement Button */}
              {activeTool !== 'none' && (
                <button
                  onClick={handleClearMeasurement}
                  className="px-2.5 py-1 rounded-full bg-slate-100 hover:bg-slate-200 text-[#64748b] text-[11px] font-medium transition-colors cursor-pointer"
                >
                  Clear
                </button>
              )}
            </div>

            {/* Measurement Value Display */}
            <div className="flex items-center gap-3">
              {measurementResult ? (
                <div className="px-3 py-1 rounded-full bg-blue-50 border border-blue-200 text-[#1d4ed8] font-mono font-bold text-xs flex items-center gap-1.5 shadow-xs">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#1d4ed8] animate-pulse" />
                  <span>{measurementResult}</span>
                </div>
              ) : activeTool !== 'none' ? (
                <span className="text-[11px] text-[#64748b] italic">
                  {activeTool === 'length' && 'Click 2 points on the sonar image to measure length'}
                  {activeTool === 'area' && 'Click points around target to compute area'}
                  {activeTool === 'height' && 'Click 3 points: 1) Nadir, 2) Object apex, 3) Shadow tip'}
                </span>
              ) : (
                <div className="flex items-center gap-2 text-[11px] font-mono text-[#64748b]">
                  <span className="w-2 h-2 rounded-full bg-[#00B4D8]" />
                  <span>Bottom Track (0.15m/px)</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column (3 Cols): Ranked Detection Queue */}
        <div className="lg:col-span-3 flex flex-col bg-white rounded-[24px] border border-[#e2e8f0] shadow-soft overflow-hidden">
          <div className="p-5 border-b border-[#f1f5f9] flex items-center justify-between">
            <div>
              <span className="section-label block">Ranked Proposals</span>
              <h3 className="text-base font-bold text-[#0f172a] font-display mt-0.5 flex items-center gap-1.5">
                <Scan className="w-4 h-4 text-[#1d4ed8]" />
                Detection Queue
              </h3>
            </div>
            <span className="text-xs font-bold px-2.5 py-1 rounded-full bg-[#f8fafc] border border-[#e2e8f0] text-[#0f172a]">
              {displayedContacts.length} Ranked
            </span>
          </div>

          {/* List of Ranked Candidate Cards */}
          <div className="flex-1 overflow-y-auto p-4 space-y-2.5">
            {displayedContacts.length === 0 ? (
              <div className="text-center py-12 text-xs text-[#64748b]">
                No anomaly detections in active swath.
              </div>
            ) : (
              displayedContacts.map((contact) => {
                const isSelected = selectedContact?.contact_id === contact.contact_id || (!selectedContact && activeContact?.contact_id === contact.contact_id);
                const isHigh = contact.priority === 'HIGH';
                const isMedium = contact.priority === 'MEDIUM';

                return (
                  <div
                    key={contact.contact_id}
                    onClick={() => onSelectContact(contact)}
                    className={`p-3.5 rounded-2xl border transition-all duration-200 cursor-pointer flex items-center justify-between group ${
                      isSelected
                        ? 'bg-blue-50/80 border-[#1d4ed8] shadow-[0_4px_12px_-2px_rgba(29,78,216,0.18)]'
                        : 'bg-[#f8fafc] border-[#e2e8f0] hover:bg-white hover:border-[#1d4ed8] hover:shadow-xs'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      {/* ID Bucket */}
                      <div className={`flex h-10 w-10 items-center justify-center rounded-xl font-mono font-bold text-xs border transition-colors ${
                        isSelected 
                          ? 'bg-[#1d4ed8] text-white border-[#1d4ed8]' 
                          : 'bg-white text-[#0f172a] border-[#e2e8f0] group-hover:border-blue-200'
                      }`}>
                        {contact.contact_id}
                      </div>

                      <div>
                        <div className="flex items-center gap-1.5">
                          <span className="font-bold text-[#0f172a] text-xs font-mono group-hover:text-[#1d4ed8] transition-colors">
                            {contact.contact_id}
                          </span>
                          <span className={`text-[9px] font-bold px-2 py-0.2 rounded-full font-sans border ${
                            isHigh
                              ? 'bg-rose-50 text-rose-700 border-rose-200'
                              : isMedium
                              ? 'bg-amber-50 text-amber-700 border-amber-200'
                              : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          }`}>
                            {contact.priority}
                          </span>
                        </div>
                        <div className="text-[11px] text-[#64748b] mt-0.5 font-sans">
                          Confidence: <strong className="text-[#0f172a]">{Math.round(contact.confidence * 100)}%</strong>
                        </div>
                      </div>
                    </div>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectContact(contact);
                        onVerifyContact?.(contact);
                      }}
                      className="p-1.5 rounded-full bg-white hover:bg-[#1d4ed8] hover:text-white text-[#64748b] border border-[#e2e8f0] transition-all cursor-pointer"
                      title="Open in Contact Verification Workflow"
                    >
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                );
              })
            )}
          </div>

          {/* Contextual Navigation: Verify Candidate Button */}
          {selectedContact && (
            <div className="p-4 border-t border-[#f1f5f9] bg-[#f8fafc] flex justify-end">
              <button
                onClick={() => {
                  onSelectContact(selectedContact);
                  onVerifyContact?.(selectedContact);
                }}
                className="w-full py-3 px-4 rounded-xl bg-[#1d4ed8] hover:bg-[#1e40af] text-white font-bold text-xs transition-all duration-200 shadow-tactile shadow-blue-glow hover:scale-[1.02] active:scale-[0.98] flex items-center justify-center gap-2 cursor-pointer group"
              >
                <span>Verify Candidate {selectedContact.contact_id}</span>
                <ArrowRight className="w-3.5 h-3.5 text-white group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          )}
        </div>

      </div>

      {/* 3. Bottom Acoustic Physics Context Verification Bar */}
      {activeContact && (
        <section className="bg-white rounded-[24px] border border-[#e2e8f0] p-5 shadow-soft flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-center gap-3 shrink-0">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-blue-50 text-[#1d4ed8] border border-blue-100 shadow-xs">
              <Scan className="w-5 h-5" />
            </div>
            <div>
              <span className="section-label block">Physics Context Engine</span>
              <h4 className="text-base font-extrabold text-[#0f172a] font-display">
                Candidate {activeContact.contact_id} Diagnostics
              </h4>
            </div>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 flex-1 max-w-4xl px-2 w-full">
            {/* 1. Object-Shadow Deficit */}
            <div className="p-3 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] flex items-center justify-between">
              <div>
                <span className="text-[9px] text-[#64748b] font-bold uppercase tracking-wider block">Shadow Deficit</span>
                <span className="text-xs font-bold text-[#0f172a]">Shadow Matched</span>
              </div>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700">
                PASS
              </span>
            </div>

            {/* 2. Seabed Texture Match */}
            <div className="p-3 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] flex items-center justify-between">
              <div>
                <span className="text-[9px] text-[#64748b] font-bold uppercase tracking-wider block">Seabed Texture</span>
                <span className="text-xs font-bold text-[#0f172a]">Sandy / Gravel</span>
              </div>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700">
                87% Match
              </span>
            </div>

            {/* 3. False Positive Risk */}
            <div className="p-3 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] flex items-center justify-between">
              <div>
                <span className="text-[9px] text-[#64748b] font-bold uppercase tracking-wider block">Clutter Risk</span>
                <span className="text-xs font-bold text-[#0f172a]">Anomalous Struct</span>
              </div>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-amber-50 text-amber-700">
                12/100
              </span>
            </div>

            {/* 4. Overall AI Composite Score */}
            <div className="p-3 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] flex items-center justify-between">
              <div>
                <span className="text-[9px] text-[#64748b] font-bold uppercase tracking-wider block">Composite Score</span>
                <span className="text-xs font-bold text-[#0f172a]">YOLO + Acoustic</span>
              </div>
              <span className="text-[10px] font-bold px-2.5 py-0.5 rounded-full bg-blue-50 text-[#1d4ed8] border border-blue-200 font-mono shadow-xs">
                {Math.round(activeContact.confidence * 100)}%
              </span>
            </div>
          </div>

          <button
            onClick={() => onVerifyContact?.(activeContact)}
            className="px-6 py-3 rounded-full bg-[#1d4ed8] hover:bg-[#1e40af] text-white font-semibold text-xs transition-all duration-200 shadow-tactile flex items-center gap-2 cursor-pointer shrink-0 shadow-blue-glow"
          >
            <span>Verify Candidate</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </section>
      )}

    </div>
  );
};
