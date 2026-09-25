import React, { useState, useRef, useEffect } from 'react';
import { DetectionOverlay } from './DetectionOverlay';
import { PreprocessingComparison } from './PreprocessingComparison';
import { Contact, SurveyUploadResponse } from '../../types/detection';
import { ZoomIn, ZoomOut, RotateCcw, Crosshair } from 'lucide-react';

interface SonarViewerProps {
  survey: SurveyUploadResponse;
  contacts: Contact[];
  selectedContact: Contact | null;
  onSelectContact: (contact: Contact) => void;
}

export const SonarViewer: React.FC<SonarViewerProps> = ({
  survey,
  contacts,
  selectedContact,
  onSelectContact
}) => {
  const [viewMode, setViewMode] = useState<'raw' | 'processed' | 'split'>('raw');
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [zoom, setZoom] = useState<number>(1.0);
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [scale, setScale] = useState<{ scaleX: number; scaleY: number }>({ scaleX: 1, scaleY: 1 });

  // Update scaling ratio between natural image dimensions and rendered layout dimensions
  const updateScaling = () => {
    if (imgRef.current && imgRef.current.clientWidth > 0) {
      const renderW = imgRef.current.clientWidth;
      const renderH = imgRef.current.clientHeight;
      setScale({
        scaleX: renderW / (survey.image_width || 1280),
        scaleY: renderH / (survey.image_height || 1800)
      });
    }
  };

  useEffect(() => {
    window.addEventListener('resize', updateScaling);
    return () => window.removeEventListener('resize', updateScaling);
  }, [survey]);

  // Center on selected contact when changed
  useEffect(() => {
    if (selectedContact && containerRef.current && scale.scaleY > 0) {
      const targetY = selectedContact.bbox.y1 * scale.scaleY;
      containerRef.current.scrollTo({
        top: Math.max(0, targetY - containerRef.current.clientHeight / 3),
        behavior: 'smooth'
      });
    }
  }, [selectedContact, scale]);

  const imageUrl = viewMode === 'raw'
    ? survey.raw_image_url
    : (survey.processed_image_url || survey.raw_image_url);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;
    if (scale.scaleX > 0 && scale.scaleY > 0) {
      setCursorPos({
        x: Math.round(clientX / scale.scaleX),
        y: Math.round(clientY / scale.scaleY)
      });
    }
  };

  return (
    <div className="relative flex-1 h-full flex flex-col bg-[#050b14] border-r border-[#1a2f4c] overflow-hidden select-none">
      {/* Top Controls Overlay */}
      <div className="absolute top-3 left-3 right-3 flex items-center justify-between z-20 pointer-events-none">
        <div className="pointer-events-auto">
          <PreprocessingComparison
            viewMode={viewMode}
            onChangeViewMode={setViewMode}
            showOverlay={showOverlay}
            onToggleOverlay={setShowOverlay}
          />
        </div>

        {/* Zoom and Cursor Coordinates Readout */}
        <div className="flex items-center gap-2 pointer-events-auto">
          {cursorPos && (
            <div className="px-2.5 py-1 rounded bg-[#070e1a]/90 backdrop-blur border border-[#1a2f4c] text-[11px] font-mono text-cyan-300 flex items-center gap-1.5">
              <Crosshair className="w-3 h-3 text-cyan-400" />
              <span>X:{cursorPos.x} Y:{cursorPos.y}</span>
            </div>
          )}

          <div className="flex items-center bg-[#070e1a]/90 backdrop-blur border border-[#1a2f4c] rounded p-0.5">
            <button
              onClick={() => setZoom(z => Math.max(0.6, z - 0.2))}
              className="p-1 text-slate-400 hover:text-cyan-300 transition-colors"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="px-1.5 text-[10px] font-mono text-slate-300">{Math.round(zoom * 100)}%</span>
            <button
              onClick={() => setZoom(z => Math.min(2.5, z + 0.2))}
              className="p-1 text-slate-400 hover:text-cyan-300 transition-colors"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setZoom(1.0)}
              className="p-1 text-slate-400 hover:text-cyan-300 transition-colors border-l border-[#1a2f4c] ml-0.5"
              title="Reset Zoom"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Swath Channel Indicators */}
      <div className="h-6 bg-[#070e1a] border-b border-[#1a2f4c] flex items-center justify-between px-6 text-[10px] font-mono text-slate-400 z-10">
        <span className="text-cyan-400">◄ PORT SWATH (-50m)</span>
        <span className="text-amber-400/80 tracking-widest">NADIR / WATER COLUMN</span>
        <span className="text-cyan-400">STARBOARD SWATH (+50m) ►</span>
      </div>

      {/* Scrollable Sonar Swath Viewport */}
      <div
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setCursorPos(null)}
        className="flex-1 overflow-auto flex justify-center p-4 bg-[#03070d]"
      >
        <div
          className="relative transition-transform duration-100 origin-top"
          style={{ transform: `scale(${zoom})` }}
        >
          {/* Main Sonar Waterfall Image */}
          <img
            ref={imgRef}
            src={imageUrl}
            alt="Side-Scan Sonar Waterfall"
            onLoad={updateScaling}
            className="max-w-[720px] w-full h-auto block border border-[#1a2f4c] shadow-2xl"
          />

          {/* Central Nadir Line Indicator */}
          <div className="absolute top-0 bottom-0 left-1/2 -translate-x-1/2 w-0.5 border-l border-dashed border-cyan-500/30 pointer-events-none" />

          {/* Detections Overlay */}
          {showOverlay && (
            <DetectionOverlay
              contacts={contacts}
              selectedContact={selectedContact}
              onSelectContact={onSelectContact}
              scaleX={scale.scaleX}
              scaleY={scale.scaleY}
            />
          )}
        </div>
      </div>
    </div>
  );
};
