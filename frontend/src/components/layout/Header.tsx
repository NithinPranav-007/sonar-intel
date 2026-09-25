import React, { useState, useRef, useEffect } from 'react';
import { 
  Sparkles, 
  Upload, 
  Play, 
  ChevronDown,
  Check
} from 'lucide-react';
import { SurveyUploadResponse } from '../../types/detection';

interface HeaderProps {
  survey: SurveyUploadResponse | null;
  analyzing: boolean;
  onRunAnalysis: () => void;
  onCustomUploadClick: () => void;
  onLoadDemoSample: (sampleId: string) => void;
  activeScreen: string;
}

export const Header: React.FC<HeaderProps> = ({
  survey,
  analyzing,
  onRunAnalysis,
  onCustomUploadClick,
  onLoadDemoSample,
  activeScreen,
}) => {
  const [dropdownOpen, setDropdownOpen] = useState<boolean>(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const demoSamples = [
    { 
      id: 'viator_04', 
      label: 'Viator-04', 
      badge: 'True Wreck', 
      desc: 'Shipwreck True Positive with acoustic shadow',
      badgeColor: 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
    },
    { 
      id: 'corsican_02', 
      label: 'Corsican-02', 
      badge: 'Verified', 
      desc: 'Held-out Target verification swath',
      badgeColor: 'bg-blue-50 text-blue-700 border border-blue-200' 
    },
    { 
      id: 'artificial_reef_02', 
      label: 'Artificial Reef', 
      badge: 'Clutter', 
      desc: 'Geological Clutter & natural seabed',
      badgeColor: 'bg-amber-50 text-amber-700 border border-amber-200' 
    },
    { 
      id: 'survey_001', 
      label: 'Survey-001', 
      badge: 'Nav Track', 
      desc: 'Towfish Nav Track with spatial trajectory',
      badgeColor: 'bg-indigo-50 text-indigo-700 border border-indigo-200' 
    },
  ];

  // Close dropdown on click outside or escape key
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setDropdownOpen(false);
      }
    };
    if (dropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      document.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [dropdownOpen]);

  const activeSample = demoSamples.find((sample) => 
    survey?.filename.toLowerCase().includes(sample.id.replace('_', ''))
  );

  const screenTitleMap: Record<string, string> = {
    'dashboard': 'Dashboard Overview',
    'sonar-analysis': 'Sonar Waterfall',
    'contact-verification': 'Contact Triage',
    'gis-mapping': 'GIS Mapping & Spatial',
    'ai-pipeline': 'Pipeline Monitor',
    'reports': 'Reports & Export',
  };

  const currentTitle = screenTitleMap[activeScreen] || 'Dashboard Overview';

  return (
    <header className="h-20 bg-white border-b border-[#e2e8f0] px-8 flex items-center justify-between sticky top-0 z-30 shadow-soft shrink-0">
      
      {/* Left: Breadcrumbs & Live Status */}
      <div className="flex items-center gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-bold text-[#64748b] uppercase tracking-wider font-sans">
              MISSION INTELLIGENCE
            </span>
            <span className="text-[#cbd5e1]">/</span>
            <span className="text-[11px] font-bold text-[#1d4ed8] uppercase tracking-wider font-sans">
              {currentTitle}
            </span>
          </div>
          <h1 className="text-xl font-extrabold text-[#0f172a] font-display tracking-tight flex items-center gap-2.5 mt-0.5">
            <span>{currentTitle}</span>
            <span className="inline-flex items-center gap-1.5 text-[11px] font-sans font-semibold px-2.5 py-0.5 rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              API Connected
            </span>
          </h1>
        </div>
      </div>

      {/* Right Actions: Benchmarks & CTAs */}
      <div className="flex items-center gap-3 sm:gap-4">
        
        {/* Benchmark Dropdown Selector */}
        <div className="relative" ref={dropdownRef}>
          <button
            type="button"
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className={`h-10 px-3.5 rounded-xl border text-xs font-semibold flex items-center gap-2 transition-all duration-200 shadow-tactile cursor-pointer ${
              dropdownOpen
                ? 'bg-blue-50 border-blue-300 text-[#1d4ed8]'
                : 'bg-[#f8fafc] hover:bg-slate-100 border-[#e2e8f0] text-[#0f172a]'
            }`}
            aria-expanded={dropdownOpen}
            aria-haspopup="true"
          >
            <div className="flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-[#1d4ed8]" />
              <span className="font-semibold font-sans">Benchmarks</span>
            </div>

            {activeSample && (
              <span className="px-2 py-0.5 rounded-md text-[10px] font-bold bg-[#1d4ed8] text-white shadow-xs max-w-[110px] truncate">
                {activeSample.label}
              </span>
            )}

            <ChevronDown 
              className={`w-3.5 h-3.5 text-[#64748b] transition-transform duration-200 ${
                dropdownOpen ? 'rotate-180 text-[#1d4ed8]' : ''
              }`} 
            />
          </button>

          {/* Dropdown Menu listing all benchmark options */}
          {dropdownOpen && (
            <div className="absolute right-0 top-full mt-2 w-80 bg-white rounded-2xl border border-[#e2e8f0] shadow-xl z-50 p-2 animate-in fade-in slide-in-from-top-1 duration-150">
              <div className="px-3 py-2 border-b border-[#f1f5f9] mb-1.5 flex items-center justify-between">
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-[#64748b]">
                    Curated Benchmarks
                  </span>
                  <p className="text-[11px] text-[#94a3b8]">
                    Select a held-out dataset
                  </p>
                </div>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-md bg-blue-50 text-[#1d4ed8] border border-blue-100">
                  {demoSamples.length} Swaths
                </span>
              </div>

              <div className="flex flex-col gap-1">
                {demoSamples.map((sample) => {
                  const isSelected = survey?.filename.toLowerCase().includes(sample.id.replace('_', ''));
                  return (
                    <button
                      key={sample.id}
                      onClick={() => {
                        onLoadDemoSample(sample.id);
                        setDropdownOpen(false);
                      }}
                      className={`w-full text-left p-2.5 rounded-xl text-xs font-semibold transition-all duration-150 flex items-center justify-between gap-3 cursor-pointer ${
                        isSelected
                          ? 'bg-blue-50/90 text-[#1d4ed8] border border-blue-200/80'
                          : 'hover:bg-[#f8fafc] text-[#0f172a] border border-transparent'
                      }`}
                    >
                      <div className="flex items-center gap-2.5 min-w-0">
                        <div className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 ${
                          isSelected ? 'bg-[#1d4ed8] text-white' : 'bg-slate-100 text-[#64748b]'
                        }`}>
                          <Sparkles className="w-3.5 h-3.5" />
                        </div>
                        <div className="min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span className="font-bold text-xs truncate">{sample.label}</span>
                            <span className={`text-[10px] px-1.5 py-0.5 rounded-md font-bold shrink-0 ${
                              sample.badgeColor
                            }`}>
                              {sample.badge}
                            </span>
                          </div>
                          <div className="text-[11px] font-normal text-[#64748b] truncate">
                            {sample.desc}
                          </div>
                        </div>
                      </div>

                      {isSelected && (
                        <div className="w-5 h-5 rounded-md bg-[#1d4ed8] text-white flex items-center justify-center shrink-0 shadow-xs">
                          <Check className="w-3 h-3 stroke-[3]" />
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Custom Upload CTA */}
        <button
          onClick={onCustomUploadClick}
          className="h-10 px-4 rounded-xl bg-white hover:bg-slate-50 text-[#0f172a] border border-[#e2e8f0] font-semibold text-xs flex items-center gap-2 transition-all duration-200 shadow-tactile cursor-pointer"
        >
          <Upload className="w-3.5 h-3.5 text-[#64748b]" />
          <span>Upload Swath</span>
        </button>

        {/* Primary Action Button: Run Inference */}
        <button
          onClick={onRunAnalysis}
          disabled={!survey || analyzing}
          className={`h-10 px-5 rounded-xl font-semibold text-xs flex items-center gap-2 transition-all duration-200 shadow-tactile ${
            analyzing
              ? 'bg-slate-200 text-[#64748b] cursor-wait'
              : survey
              ? 'bg-[#1d4ed8] hover:bg-[#1e40af] text-white hover:scale-[1.02] active:scale-[0.98] cursor-pointer shadow-blue-glow'
              : 'bg-slate-100 text-[#64748b] cursor-not-allowed'
          }`}
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          <span>{analyzing ? 'Inference Running...' : 'Run AI Detection'}</span>
        </button>
      </div>
    </header>
  );
};
