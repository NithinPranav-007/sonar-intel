import React from 'react';
import { SlidersHorizontal, Eye } from 'lucide-react';

interface PreprocessingComparisonProps {
  viewMode: 'raw' | 'processed' | 'split';
  onChangeViewMode: (mode: 'raw' | 'processed' | 'split') => void;
  showOverlay: boolean;
  onToggleOverlay: (show: boolean) => void;
}

export const PreprocessingComparison: React.FC<PreprocessingComparisonProps> = ({
  viewMode,
  onChangeViewMode,
  showOverlay,
  onToggleOverlay
}) => {
  return (
    <div className="flex items-center gap-2 p-1.5 rounded bg-[#070e1a]/90 backdrop-blur border border-[#1a2f4c] text-xs font-mono select-none">
      <div className="flex items-center gap-1 bg-[#0b1626] p-0.5 rounded border border-[#1a2f4c]">
        <button
          onClick={() => onChangeViewMode('raw')}
          className={`px-2 py-1 rounded transition-colors ${
            viewMode === 'raw'
              ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          RAW
        </button>
        <button
          onClick={() => onChangeViewMode('processed')}
          className={`px-2 py-1 rounded transition-colors ${
            viewMode === 'processed'
              ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          NORMALIZED (1-99%)
        </button>
      </div>

      <div className="h-4 w-px bg-[#1a2f4c]" />

      {/* Toggle Bounding Boxes */}
      <button
        onClick={() => onToggleOverlay(!showOverlay)}
        className={`flex items-center gap-1 px-2 py-1 rounded transition-colors ${
          showOverlay
            ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-700/50 font-semibold'
            : 'text-slate-500 hover:text-slate-300'
        }`}
      >
        <Eye className="w-3.5 h-3.5" />
        <span>BOUNDING BOXES</span>
      </button>
    </div>
  );
};
