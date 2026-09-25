import React, { useState } from 'react';
import { Contact, NavWaypoint, SurveyUploadResponse } from '../types/detection';
import { MapView, DEFAULT_OCEAN_CANDIDATES, DEFAULT_NAV_TRACK } from '../components/map/MapView';
import { 
  Compass, 
  MapPin, 
  Filter, 
  FileDown, 
  Layers, 
  ArrowRight, 
  Waves, 
  Scan,
  ShieldCheck,
  ExternalLink,
  Search,
  Copy,
  FileSpreadsheet,
  CheckCircle2,
  Calendar,
  ChevronLeft,
  ChevronRight,
  SlidersHorizontal,
  X
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
  onNavigateToDashboard?: () => void;
}

export const GisMappingPage: React.FC<GisMappingPageProps> = ({
  survey,
  contacts,
  selectedContact,
  navTrack,
  onSelectContact,
  onNavigateToAnalysis,
  onNavigateToVerify,
  onExportGeoJSON,
  onNavigateToDashboard
}) => {
  const [filterMode, setFilterMode] = useState<'all' | 'high'>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [isSearchPanelOpen, setIsSearchPanelOpen] = useState<boolean>(true);
  const [isExportDropdownOpen, setIsExportDropdownOpen] = useState<boolean>(false);
  const [copyToast, setCopyToast] = useState<string | null>(null);

  // Filter Checkbox States
  const [priorityFilters, setPriorityFilters] = useState({
    all: true,
    high: true,
    medium: true,
    low: true
  });

  // Filter Bar Dropdown & Date States
  const [priorityDropdown, setPriorityDropdown] = useState<string>('all');
  const [statusDropdown, setStatusDropdown] = useState<string>('all');
  const [dateFrom, setDateFrom] = useState<string>('2026-09-01');
  const [dateTo, setDateTo] = useState<string>('2026-09-07');

  const handleCheckboxToggle = (key: 'all' | 'high' | 'medium' | 'low') => {
    if (key === 'all') {
      const nextVal = !priorityFilters.all;
      setPriorityFilters({ all: nextVal, high: nextVal, medium: nextVal, low: nextVal });
    } else {
      const updated = { ...priorityFilters, [key]: !priorityFilters[key] };
      updated.all = updated.high && updated.medium && updated.low;
      setPriorityFilters(updated);
    }
  };

  const sourceContacts = (contacts && contacts.length > 0) ? contacts : DEFAULT_OCEAN_CANDIDATES;
  const sourceNavTrack = (navTrack && navTrack.length > 0) ? navTrack : DEFAULT_NAV_TRACK;

  const filteredContacts = sourceContacts.filter(c => {
    // 1. Header Filter Pills
    if (filterMode === 'high' && c.priority !== 'HIGH') return false;

    // 2. Search Query (ID, Priority, Class)
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      const matchId = c.contact_id.toLowerCase().includes(q);
      const matchPriority = c.priority.toLowerCase().includes(q);
      const matchClass = c.class_name.toLowerCase().includes(q);
      if (!matchId && !matchPriority && !matchClass) return false;
    }

    // 3. Filter Checkboxes
    if (!priorityFilters.all) {
      if (c.priority === 'HIGH' && !priorityFilters.high) return false;
      if (c.priority === 'MEDIUM' && !priorityFilters.medium) return false;
      if (c.priority === 'LOW' && !priorityFilters.low) return false;
    }

    // 4. Dropdown Priority Filter
    if (priorityDropdown !== 'all' && c.priority !== priorityDropdown) return false;

    // 5. Dropdown Status Filter
    if (statusDropdown !== 'all') {
      if (statusDropdown === 'CONFIRMED' && c.review_status !== 'CONFIRMED') return false;
      if (statusDropdown === 'FALSE_POSITIVE' && c.review_status !== 'FALSE_POSITIVE') return false;
      if (statusDropdown === 'UNCERTAIN' && c.review_status !== 'UNCERTAIN') return false;
      if (statusDropdown === 'AI_CANDIDATE' && c.review_status !== 'AI_CANDIDATE') return false;
    }

    return true;
  });

  const activeContact = selectedContact || filteredContacts[0] || null;

  // Export KML/KMZ Handler
  const handleExportKML = () => {
    const kmlHeader = `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Sonar-Intel Survey Contacts</name>
    <description>WGS-84 Geodetic Fix Points for Side-Scan Sonar Contacts</description>
    ${contacts.filter(c => c.latitude != null && c.longitude != null).map(c => `
    <Placemark>
      <name>${c.contact_id} - ${c.class_name}</name>
      <description>Priority: ${c.priority}, Status: ${c.review_status}, Confidence: ${(c.confidence * 100).toFixed(1)}%</description>
      <Point>
        <coordinates>${c.longitude},${c.latitude},0</coordinates>
      </Point>
    </Placemark>`).join('')}
  </Document>
</kml>`;
    const blob = new Blob([kmlHeader], { type: 'application/vnd.google-earth.kml+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sonar_contacts_${survey?.survey_id || 'geospatial'}.kml`;
    a.click();
    URL.revokeObjectURL(url);
    setCopyToast('KML/KMZ file exported successfully!');
    setIsExportDropdownOpen(false);
    setTimeout(() => setCopyToast(null), 3000);
  };

  // Copy Coordinates Handler
  const handleCopyCoordinates = () => {
    if (activeContact?.latitude && activeContact?.longitude) {
      const text = `Candidate ${activeContact.contact_id}: LAT ${activeContact.latitude.toFixed(6)}° N, LON ${activeContact.longitude.toFixed(6)}° E (WGS-84)`;
      navigator.clipboard.writeText(text);
      setCopyToast(`Coordinates for ${activeContact.contact_id} copied!`);
      setIsExportDropdownOpen(false);
      setTimeout(() => setCopyToast(null), 3000);
    } else {
      setCopyToast('Target coordinates unavailable or awaiting nav log.');
      setIsExportDropdownOpen(false);
      setTimeout(() => setCopyToast(null), 3000);
    }
  };

  // Export Report Handler
  const handleExportReport = () => {
    onExportGeoJSON();
    setCopyToast('Survey GeoJSON report download started.');
    setIsExportDropdownOpen(false);
    setTimeout(() => setCopyToast(null), 3000);
  };

  return (
    <div className="p-6 lg:p-8 max-w-[1700px] mx-auto space-y-6 font-sans">
      
      {/* 1. Header Toolbar */}
      <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        <div className="space-y-2">
          {/* Top-Left: Back Button & Breadcrumb */}
          <div className="flex items-center gap-3">
            {onNavigateToDashboard && (
              <button
                onClick={onNavigateToDashboard}
                className="px-3.5 py-1.5 rounded-xl bg-[#f8fafc] hover:bg-slate-100 text-[#0f172a] hover:text-[#1d4ed8] border border-[#e2e8f0] font-semibold text-xs transition-all duration-200 shadow-tactile flex items-center gap-1.5 cursor-pointer group"
              >
                <ChevronLeft className="w-3.5 h-3.5 text-[#64748b] group-hover:text-[#1d4ed8] group-hover:-translate-x-0.5 transition-transform" />
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
                GIS Mapping & Spatial
              </span>
            </div>
          </div>

          <h2 className="text-2xl font-extrabold text-[#0f172a] font-display flex items-center gap-2.5">
            <Compass className="w-6 h-6 text-[#1d4ed8]" />
            GIS Mapping & Spatial Cleanup Planning
          </h2>
        </div>

        {/* Spatial Filter Buttons & Grouped Export Dropdown */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5 p-1 rounded-full bg-[#f8fafc] border border-[#e2e8f0] shadow-tactile">
            <button
              onClick={() => setFilterMode('all')}
              className={`px-3.5 py-1.5 rounded-full text-xs font-bold transition-all cursor-pointer ${
                filterMode === 'all'
                  ? 'bg-[#1d4ed8] text-white shadow-blue-sm'
                  : 'text-[#64748b] hover:text-[#0f172a]'
              }`}
            >
              All Areas ({contacts.length})
            </button>

            <button
              onClick={() => setFilterMode('high')}
              className={`px-3.5 py-1.5 rounded-full text-xs font-bold transition-all cursor-pointer ${
                filterMode === 'high'
                  ? 'bg-rose-500 text-white shadow-sm'
                  : 'text-[#64748b] hover:text-[#0f172a]'
              }`}
            >
              High Priority
            </button>
          </div>

          {/* Grouped Export Dropdown */}
          <div className="relative">
            <button
              onClick={() => setIsExportDropdownOpen(!isExportDropdownOpen)}
              className="px-4 py-2 rounded-full bg-[#1d4ed8] hover:bg-[#1e40af] text-white font-semibold text-xs transition-all duration-200 shadow-tactile flex items-center gap-2 cursor-pointer shadow-blue-glow"
            >
              <FileDown className="w-3.5 h-3.5" />
              <span>Export</span>
              <span className="text-[10px] opacity-80">▼</span>
            </button>

            {isExportDropdownOpen && (
              <div className="absolute right-0 top-full mt-2 w-56 bg-white rounded-2xl border border-[#e2e8f0] shadow-xl z-50 p-2 space-y-1 animate-in fade-in duration-150">
                <button
                  onClick={() => {
                    onExportGeoJSON();
                    setIsExportDropdownOpen(false);
                  }}
                  className="w-full text-left px-3 py-2 rounded-xl text-xs font-semibold text-[#0f172a] hover:bg-blue-50 hover:text-[#1d4ed8] flex items-center gap-2 transition-colors cursor-pointer"
                >
                  <FileDown className="w-3.5 h-3.5 text-[#1d4ed8]" />
                  <span>Export GeoJSON</span>
                </button>
                <button
                  onClick={handleExportKML}
                  className="w-full text-left px-3 py-2 rounded-xl text-xs font-semibold text-[#0f172a] hover:bg-blue-50 hover:text-[#1d4ed8] flex items-center gap-2 transition-colors cursor-pointer"
                >
                  <FileDown className="w-3.5 h-3.5 text-indigo-600" />
                  <span>Export KML / KMZ</span>
                </button>
                <button
                  onClick={handleExportReport}
                  className="w-full text-left px-3 py-2 rounded-xl text-xs font-semibold text-[#0f172a] hover:bg-blue-50 hover:text-[#1d4ed8] flex items-center gap-2 transition-colors cursor-pointer"
                >
                  <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-600" />
                  <span>Export Report</span>
                </button>
                <button
                  onClick={handleCopyCoordinates}
                  className="w-full text-left px-3 py-2 rounded-xl text-xs font-semibold text-[#0f172a] hover:bg-blue-50 hover:text-[#1d4ed8] flex items-center gap-2 transition-colors cursor-pointer border-t border-slate-100 mt-1 pt-2"
                >
                  <Copy className="w-3.5 h-3.5 text-[#64748b]" />
                  <span>Copy Coordinates</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Toast Notice */}
      {copyToast && (
        <div className="p-3.5 rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold flex items-center justify-between shadow-soft">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            <span>{copyToast}</span>
          </div>
          <button onClick={() => setCopyToast(null)} className="text-emerald-700 hover:text-emerald-900 font-bold text-xs cursor-pointer">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* 2. Main Map Canvas + Left Search Panel + Right Telemetry Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[640px]">
        
        {/* Left Side: Candidate Search Panel (Collapsible) */}
        {isSearchPanelOpen ? (
          <div className="lg:col-span-3 bg-white rounded-[24px] border border-[#e2e8f0] p-5 shadow-soft space-y-4 flex flex-col justify-between max-h-[720px] overflow-hidden font-sans">
            <div className="space-y-4 overflow-y-auto pr-1">
              
              {/* Panel Header */}
              <div className="flex items-center justify-between border-b border-[#f1f5f9] pb-3">
                <div className="flex items-center gap-2">
                  <Search className="w-4 h-4 text-[#1d4ed8]" />
                  <h3 className="text-sm font-bold text-[#0f172a] font-display">
                    Candidate Search
                  </h3>
                </div>
                <button
                  onClick={() => setIsSearchPanelOpen(false)}
                  title="Collapse Panel"
                  className="p-1 rounded-lg hover:bg-slate-100 text-[#64748b] transition-colors cursor-pointer"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
              </div>

              {/* Search Input */}
              <div className="relative">
                <Search className="w-3.5 h-3.5 text-[#64748b] absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search by ID or priority..."
                  className="w-full pl-9 pr-3 py-2 bg-[#f8fafc] border border-[#e2e8f0] rounded-xl text-xs text-[#0f172a] placeholder:text-[#64748b] focus:outline-none focus:border-blue-600 focus:ring-1 focus:ring-blue-600 transition-all"
                />
              </div>

              {/* Filter Checkboxes */}
              <div className="space-y-1.5 pt-1">
                <div className="text-[10px] font-bold text-[#64748b] uppercase tracking-wider">
                  Priority Filter Checkboxes
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <label className="flex items-center gap-2 p-1.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0] cursor-pointer hover:bg-slate-100 transition-colors">
                    <input
                      type="checkbox"
                      checked={priorityFilters.all}
                      onChange={() => handleCheckboxToggle('all')}
                      className="rounded text-[#1d4ed8] focus:ring-0 cursor-pointer"
                    />
                    <span className="font-semibold text-[#0f172a]">All ({contacts.length})</span>
                  </label>
                  <label className="flex items-center gap-2 p-1.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0] cursor-pointer hover:bg-slate-100 transition-colors">
                    <input
                      type="checkbox"
                      checked={priorityFilters.high}
                      onChange={() => handleCheckboxToggle('high')}
                      className="rounded text-rose-600 focus:ring-0 cursor-pointer"
                    />
                    <span className="font-semibold text-rose-700 flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-rose-500" /> High
                    </span>
                  </label>
                  <label className="flex items-center gap-2 p-1.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0] cursor-pointer hover:bg-slate-100 transition-colors">
                    <input
                      type="checkbox"
                      checked={priorityFilters.medium}
                      onChange={() => handleCheckboxToggle('medium')}
                      className="rounded text-amber-600 focus:ring-0 cursor-pointer"
                    />
                    <span className="font-semibold text-amber-700 flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-500" /> Med
                    </span>
                  </label>
                  <label className="flex items-center gap-2 p-1.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0] cursor-pointer hover:bg-slate-100 transition-colors">
                    <input
                      type="checkbox"
                      checked={priorityFilters.low}
                      onChange={() => handleCheckboxToggle('low')}
                      className="rounded text-yellow-600 focus:ring-0 cursor-pointer"
                    />
                    <span className="font-semibold text-yellow-700 flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-yellow-400" /> Low
                    </span>
                  </label>
                </div>
              </div>

              {/* Filter Bar (Below Search) */}
              <div className="space-y-2 p-3 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold text-[#64748b] uppercase tracking-wider flex items-center gap-1">
                    <SlidersHorizontal className="w-3 h-3 text-[#1d4ed8]" />
                    Filter Bar
                  </span>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] text-[#64748b]">Priority:</span>
                    <select
                      value={priorityDropdown}
                      onChange={(e) => setPriorityDropdown(e.target.value)}
                      className="px-2 py-1 bg-white border border-[#e2e8f0] rounded-lg text-xs font-semibold text-[#0f172a] focus:outline-none"
                    >
                      <option value="all">All ▼</option>
                      <option value="HIGH">High Priority</option>
                      <option value="MEDIUM">Medium Priority</option>
                      <option value="LOW">Low Priority</option>
                    </select>
                  </div>

                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] text-[#64748b]">Status:</span>
                    <select
                      value={statusDropdown}
                      onChange={(e) => setStatusDropdown(e.target.value)}
                      className="px-2 py-1 bg-white border border-[#e2e8f0] rounded-lg text-xs font-semibold text-[#0f172a] focus:outline-none"
                    >
                      <option value="all">All ▼</option>
                      <option value="CONFIRMED">Confirmed</option>
                      <option value="AI_CANDIDATE">AI Candidate</option>
                      <option value="UNCERTAIN">Uncertain</option>
                      <option value="FALSE_POSITIVE">False Positive</option>
                    </select>
                  </div>

                  <div className="flex flex-col gap-1 pt-1 border-t border-slate-200/60">
                    <span className="text-[10px] font-bold text-[#64748b] flex items-center gap-1">
                      <Calendar className="w-3 h-3 text-[#64748b]" />
                      Date Range:
                    </span>
                    <div className="grid grid-cols-2 gap-1.5">
                      <input
                        type="date"
                        value={dateFrom}
                        onChange={(e) => setDateFrom(e.target.value)}
                        className="p-1 bg-white border border-[#e2e8f0] rounded text-[11px] font-mono text-[#0f172a]"
                      />
                      <input
                        type="date"
                        value={dateTo}
                        onChange={(e) => setDateTo(e.target.value)}
                        className="p-1 bg-white border border-[#e2e8f0] rounded text-[11px] font-mono text-[#0f172a]"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Scrollable Candidate List */}
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center justify-between text-[10px] font-bold text-[#64748b] uppercase tracking-wider">
                  <span>Candidate Queue</span>
                  <span>{filteredContacts.length} Targets</span>
                </div>

                <div className="space-y-1.5 max-h-[220px] overflow-y-auto pr-1">
                  {filteredContacts.map((c) => {
                    const isSelected = activeContact?.contact_id === c.contact_id;
                    const isHigh = c.priority === 'HIGH';
                    const isMed = c.priority === 'MEDIUM';

                    return (
                      <div
                        key={c.contact_id}
                        onClick={() => onSelectContact(c)}
                        className={`p-2.5 rounded-xl border transition-all cursor-pointer flex items-center justify-between group ${
                          isSelected
                            ? 'bg-blue-50 border-blue-300 shadow-blue-sm scale-[1.01]'
                            : 'bg-[#f8fafc] border-[#e2e8f0] hover:border-[#1d4ed8] hover:bg-blue-50/30 hover:shadow-xs'
                        }`}
                      >
                        <div className="flex items-center gap-2.5">
                          {/* Priority Dot */}
                          <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${
                            c.review_status === 'CONFIRMED'
                              ? 'bg-emerald-500 shadow-xs'
                              : isHigh
                              ? 'bg-rose-500 shadow-xs'
                              : isMed
                              ? 'bg-amber-500 shadow-xs'
                              : 'bg-yellow-400'
                          }`} />

                          <div>
                            <div className="font-mono font-bold text-xs text-[#0f172a] group-hover:text-[#1d4ed8] transition-colors">{c.contact_id}</div>
                            <div className="text-[10px] text-[#64748b] font-medium">
                              {c.priority} • <span className="font-bold text-[#1d4ed8]">{Math.round(c.confidence * 100)}%</span>
                            </div>
                          </div>
                        </div>

                        {/* Navigate Button */}
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectContact(c);
                          }}
                          title="Navigate & Center on Map"
                          className="p-1.5 rounded-lg bg-white border border-[#e2e8f0] text-[#1d4ed8] hover:bg-blue-600 hover:text-white transition-colors cursor-pointer shadow-xs"
                        >
                          <ArrowRight className="w-3 h-3" />
                        </button>
                      </div>
                    );
                  })}

                  {filteredContacts.length === 0 && (
                    <div className="p-4 text-center text-xs text-[#64748b] bg-[#f8fafc] rounded-xl border border-dashed border-[#e2e8f0]">
                      No matching targets found
                    </div>
                  )}
                </div>
              </div>

            </div>

            <div className="pt-2 border-t border-[#f1f5f9] text-[11px] text-[#64748b] text-center font-mono">
              PostGIS Spatial Filter Active
            </div>
          </div>
        ) : (
          <div className="lg:col-span-1 bg-white rounded-[24px] border border-[#e2e8f0] p-3 shadow-soft flex flex-col items-center justify-between">
            <button
              onClick={() => setIsSearchPanelOpen(true)}
              title="Expand Candidate Search Panel"
              className="p-2.5 rounded-xl bg-blue-50 text-[#1d4ed8] hover:bg-blue-100 transition-colors cursor-pointer flex flex-col items-center gap-1"
            >
              <Search className="w-4 h-4" />
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
            <div className="rotate-90 text-[10px] font-bold text-[#64748b] uppercase tracking-wider whitespace-nowrap">
              Search Panel
            </div>
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          </div>
        )}
        
        {/* Vector Nautical Map Canvas (Center Column) */}
        <div className={`${isSearchPanelOpen ? 'lg:col-span-5' : 'lg:col-span-7'} bg-white rounded-[24px] border border-[#e2e8f0] shadow-soft overflow-hidden flex flex-col transition-all duration-300`}>
          <div className="p-4 border-b border-[#e2e8f0] bg-[#f8fafc] flex items-center justify-between text-xs font-sans">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="font-bold text-[#0f172a]">MapLibre GL Nautical GIS</span>
              <span className="text-[#64748b] hidden sm:inline">• Bathymetric Ocean Contours & PostGIS Trajectory</span>
            </div>
            <span className="text-xs font-bold px-3 py-1 rounded-full bg-white border border-[#e2e8f0] text-[#64748b]">
              Datum: WGS-84 (EPSG:4326)
            </span>
          </div>

          <div className="flex-1 w-full min-h-[550px] relative bg-[#050a14]">
            <MapView
              contacts={filteredContacts}
              selectedContact={activeContact}
              navTrack={sourceNavTrack}
              onSelectContact={onSelectContact}
            />
          </div>
        </div>

        {/* Spatial Target Telemetry Drawer (Right Column: 4 Cols) */}
        <div className="lg:col-span-4 bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-5 flex flex-col justify-between font-sans">
          {activeContact ? (
            <div className="space-y-5">
              <div className="border-b border-[#f1f5f9] pb-3 flex items-center justify-between">
                <div>
                  <span className="section-label block">Target Spatial Pin</span>
                  <h3 className="text-base font-bold text-[#0f172a] font-display mt-0.5 flex items-center gap-1.5">
                    <MapPin className="w-4 h-4 text-[#1d4ed8]" />
                    Candidate {activeContact.contact_id}
                  </h3>
                </div>
                <span className={`text-xs font-bold px-3 py-1 rounded-full border ${
                  activeContact.priority === 'HIGH'
                    ? 'bg-rose-50 text-rose-700 border-rose-200'
                    : activeContact.priority === 'MEDIUM'
                    ? 'bg-amber-50 text-amber-700 border-amber-200'
                    : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                }`}>
                  {activeContact.priority} PRIORITY
                </span>
              </div>

              {/* Geographic Coordinates Card */}
              <div className="p-4 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] space-y-2">
                <div className="flex items-center justify-between">
                  <span className="section-label block">WGS-84 Geodetic Fix</span>
                  <button
                    onClick={handleCopyCoordinates}
                    title="Copy Fix Coordinates"
                    className="text-[10px] font-bold text-[#1d4ed8] hover:underline flex items-center gap-1 cursor-pointer"
                  >
                    <Copy className="w-3 h-3" />
                    <span>Copy</span>
                  </button>
                </div>
                {activeContact.latitude && activeContact.longitude ? (
                  <div className="space-y-1 font-mono text-xs font-bold text-emerald-700">
                    <div>LAT: {activeContact.latitude.toFixed(6)}° N</div>
                    <div>LON: {activeContact.longitude.toFixed(6)}° E</div>
                  </div>
                ) : (
                  <div className="text-xs text-amber-700 font-medium italic">
                    Coordinates unavailable (Awaiting towfish navigation log)
                  </div>
                )}
                <div className="text-[11px] text-[#64748b] pt-2 border-t border-[#e2e8f0] flex justify-between">
                  <span>Provenance:</span>
                  <span className="font-bold text-[#0f172a]">{activeContact.localization_status}</span>
                </div>
              </div>

              {/* Physical Diagnostics Card */}
              <div className="p-4 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] space-y-2.5 text-xs">
                <span className="section-label block">Target Diagnostics</span>
                <div className="flex justify-between items-center">
                  <span className="text-[#64748b]">Class:</span>
                  <span className="font-bold text-[#0f172a] font-mono">{activeContact.class_name}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[#64748b]">AI Confidence:</span>
                  <span className="font-bold text-[#0f172a] font-mono">{Math.round(activeContact.confidence * 100)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[#64748b]">Triage Status:</span>
                  <span className={`font-bold font-mono ${
                    activeContact.review_status === 'CONFIRMED'
                      ? 'text-emerald-700'
                      : activeContact.review_status === 'FALSE_POSITIVE'
                      ? 'text-rose-700'
                      : activeContact.review_status === 'UNCERTAIN'
                      ? 'text-amber-700'
                      : 'text-[#1d4ed8]'
                  }`}>{activeContact.review_status}</span>
                </div>
              </div>

              {/* Navigation Action Buttons */}
              <div className="space-y-2.5 pt-2 border-t border-[#f1f5f9]">
                <button
                  onClick={onNavigateToAnalysis}
                  className="w-full py-3 rounded-full bg-[#f8fafc] hover:bg-blue-50/50 hover:border-blue-200 border border-[#e2e8f0] text-[#0f172a] font-semibold text-xs transition-colors flex items-center justify-center gap-2 shadow-tactile cursor-pointer"
                >
                  <Waves className="w-4 h-4 text-[#1d4ed8]" />
                  <span>Inspect in Sonar Waterfall</span>
                </button>

                <button
                  onClick={onNavigateToVerify}
                  className="w-full py-3 rounded-full bg-[#1d4ed8] hover:bg-[#1e40af] text-white font-semibold text-xs transition-colors flex items-center justify-center gap-2 shadow-tactile cursor-pointer shadow-blue-glow"
                >
                  <span>Verify & Classify Target</span>
                  <ArrowRight className="w-4 h-4" />
                </button>

                {onNavigateToDashboard && (
                  <button
                    onClick={onNavigateToDashboard}
                    className="w-full py-2.5 rounded-full bg-[#f8fafc] hover:bg-slate-100 border border-[#e2e8f0] text-[#64748b] hover:text-[#0f172a] font-semibold text-xs transition-colors flex items-center justify-center gap-1.5 cursor-pointer shadow-tactile"
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                    <span>Back to Dashboard Overview</span>
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="text-center py-16 space-y-3">
              <MapPin className="w-8 h-8 text-[#64748b] mx-auto" />
              <div className="text-sm font-bold text-[#0f172a]">No Spatial Target Selected</div>
              <p className="text-xs text-[#64748b]">Click any spatial pin on the nautical map to view telemetry.</p>
              {onNavigateToDashboard && (
                <button
                  onClick={onNavigateToDashboard}
                  className="mt-4 px-4 py-2 rounded-full bg-[#f8fafc] hover:bg-slate-100 border border-[#e2e8f0] text-[#0f172a] font-semibold text-xs transition-colors inline-flex items-center gap-1.5 cursor-pointer"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                  <span>Back to Dashboard Overview</span>
                </button>
              )}
            </div>
          )}
        </div>

      </div>

    </div>
  );
};
