import React, { useState, useEffect } from 'react';
import { Contact, SurveyUploadResponse, ReviewStatus } from '../types/detection';
import { 
  CheckCircle2, 
  XCircle, 
  HelpCircle, 
  MapPin, 
  Save, 
  Clock, 
  Scan, 
  ShieldCheck, 
  Compass, 
  ArrowRight,
  ArrowLeft,
  Sparkles,
  Layers,
  ChevronRight,
  ChevronLeft,
  SkipForward,
  Camera
} from 'lucide-react';

interface ContactVerificationPageProps {
  survey: SurveyUploadResponse | null;
  contacts: Contact[];
  selectedContact: Contact | null;
  onSelectContact: (contact: Contact) => void;
  onSubmitReview: (contactId: string, status: ReviewStatus, note?: string) => Promise<void>;
  onNavigateToMap: () => void;
  onNavigateToAnalysis?: () => void;
  onNavigateToDashboard?: () => void;
}

export const ContactVerificationPage: React.FC<ContactVerificationPageProps> = ({
  survey,
  contacts,
  selectedContact,
  onSelectContact,
  onSubmitReview,
  onNavigateToMap,
  onNavigateToAnalysis,
  onNavigateToDashboard
}) => {
  const activeContact = selectedContact || contacts[0] || null;
  const [operatorNote, setOperatorNote] = useState<string>('');
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [lastSavedTime, setLastSavedTime] = useState<string>(() => {
    const now = new Date();
    return now.toTimeString().split(' ')[0];
  });

  const currentIndex = contacts.findIndex(c => c.contact_id === activeContact?.contact_id);
  const currentNum = currentIndex >= 0 ? currentIndex + 1 : 1;
  const totalNum = contacts.length || 1;
  const progressPercent = Math.round((currentNum / totalNum) * 100);

  const handleSkipNext = () => {
    if (contacts.length === 0) return;
    const nextIndex = (currentIndex + 1) % contacts.length;
    onSelectContact(contacts[nextIndex]);
  };

  const handlePrevCandidate = () => {
    if (contacts.length === 0) return;
    const prevIndex = (currentIndex - 1 + contacts.length) % contacts.length;
    onSelectContact(contacts[prevIndex]);
  };

  const handleNextCandidate = () => {
    if (contacts.length === 0) return;
    const nextIndex = (currentIndex + 1) % contacts.length;
    onSelectContact(contacts[nextIndex]);
  };

  const handleAction = async (status: ReviewStatus) => {
    if (!activeContact) return;
    setSubmitting(true);
    setSaveMessage(null);
    try {
      await onSubmitReview(activeContact.contact_id, status, operatorNote);
      setSaveMessage(`Target ${activeContact.contact_id} classification updated to ${status.replace('_', ' ')}.`);
      const now = new Date();
      setLastSavedTime(now.toTimeString().split(' ')[0]);
      setOperatorNote('');
    } catch (err) {
      console.error('Review submission error:', err);
    } finally {
      setSubmitting(false);
    }
  };

  const handleSaveAndContinue = async () => {
    if (!activeContact) return;
    await handleAction(activeContact.review_status || 'CONFIRMED');
    handleSkipNext();
  };

  // Keyboard shortcut listener: 1 = Confirm Debris, 2 = False Alarm, 3 = Needs Review
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const targetTag = (e.target as HTMLElement)?.tagName?.toLowerCase();
      if (targetTag === 'input' || targetTag === 'textarea') return;

      if (e.key === '1') {
        e.preventDefault();
        handleAction('CONFIRMED');
      } else if (e.key === '2') {
        e.preventDefault();
        handleAction('FALSE_POSITIVE');
      } else if (e.key === '3') {
        e.preventDefault();
        handleAction('UNCERTAIN');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeContact, operatorNote, submitting]);

  if (!activeContact) {
    return (
      <div className="p-12 text-center max-w-lg mx-auto bg-white rounded-[24px] border border-[#e2e8f0] shadow-soft my-12 space-y-4">
        <div className="w-16 h-16 rounded-full bg-sky-50 text-sky-600 flex items-center justify-center mx-auto border border-sky-100">
          <Scan className="w-8 h-8" />
        </div>
        <h3 className="text-xl font-bold text-[#0f172a] font-display">No Candidate Contact Selected</h3>
        <p className="text-xs text-[#64748b]">
          Select a survey swath or benchmark case from the top header to begin operator triage.
        </p>
      </div>
    );
  }

  const bboxWidth = activeContact.bbox.x2 - activeContact.bbox.x1;
  const bboxHeight = activeContact.bbox.y2 - activeContact.bbox.y1;
  const isHigh = activeContact.priority === 'HIGH';
  const isMedium = activeContact.priority === 'MEDIUM';

  return (
    <div className="p-6 lg:p-8 max-w-[1600px] mx-auto space-y-6 font-sans">
      
      {/* 1. Header & Contact Selector Pill Bar */}
      <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
        <div className="space-y-2">
          {/* Top-left: Back Button & Breadcrumb */}
          <div className="flex items-center gap-3">
            {onNavigateToAnalysis && (
              <button
                onClick={onNavigateToAnalysis}
                className="px-3.5 py-1.5 rounded-xl bg-[#f8fafc] hover:bg-slate-100 text-[#0f172a] hover:text-[#1d4ed8] border border-[#e2e8f0] font-semibold text-xs transition-all duration-200 shadow-tactile flex items-center gap-1.5 cursor-pointer group"
              >
                <ArrowLeft className="w-3.5 h-3.5 text-[#64748b] group-hover:text-[#1d4ed8] group-hover:-translate-x-0.5 transition-transform" />
                <span>Back to Sonar Waterfall</span>
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
              <button
                onClick={onNavigateToAnalysis}
                className="text-[#64748b] hover:text-[#1d4ed8] hover:underline cursor-pointer transition-colors"
              >
                Sonar Waterfall
              </button>
              <span className="text-[#cbd5e1]">&gt;</span>
              <span className="text-[#1d4ed8] font-bold">
                Contact Triage
              </span>
            </div>
          </div>

          <h2 className="text-2xl font-extrabold text-[#0f172a] font-display flex items-center gap-2.5">
            <ShieldCheck className="w-6 h-6 text-[#1d4ed8]" />
            Contact Triage: Target {activeContact.contact_id} Verification & Audit
          </h2>
        </div>

        {/* Right side: Candidate Pills & Next Candidate Button */}
        <div className="flex items-center gap-2.5">
          {/* Contact Selector Pills */}
          <div className="flex items-center gap-1.5 overflow-x-auto p-1.5 rounded-full bg-[#f8fafc] border border-[#e2e8f0] shadow-tactile max-w-full">
            {contacts.map((c) => {
              const isSelected = activeContact.contact_id === c.contact_id;
              return (
                <button
                  key={c.contact_id}
                  onClick={() => onSelectContact(c)}
                  className={`px-3.5 py-1.5 rounded-full text-xs font-bold transition-all duration-200 cursor-pointer flex items-center gap-1.5 ${
                    isSelected
                      ? 'bg-[#1d4ed8] text-white shadow-blue-sm scale-[1.02]'
                      : 'text-[#0f172a] hover:bg-slate-200/70'
                  }`}
                >
                  <span>{c.contact_id}</span>
                  {c.priority === 'HIGH' && (
                    <span className={`w-1.5 h-1.5 rounded-full ${isSelected ? 'bg-white' : 'bg-rose-500'}`} />
                  )}
                </button>
              );
            })}
          </div>

          {/* Next Candidate Button */}
          <button
            onClick={handleNextCandidate}
            title="Next Candidate"
            className="px-3.5 py-2 rounded-full bg-[#f8fafc] hover:bg-slate-100 border border-[#e2e8f0] text-[#0f172a] text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer shadow-tactile shrink-0"
          >
            <span>Next Candidate</span>
            <ChevronRight className="w-3.5 h-3.5 text-[#64748b]" />
          </button>
        </div>
      </div>

      {/* Success Notification Alert */}
      {saveMessage && (
        <div className="p-4 rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold flex items-center justify-between shadow-soft">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            <span>{saveMessage}</span>
          </div>
          <button 
            onClick={() => setSaveMessage(null)} 
            className="text-emerald-700 hover:text-emerald-900 font-bold underline cursor-pointer text-[11px]"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* 2. Main Two-Column Triage Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column (5 Cols): Acoustic Target Optical Crop */}
        <div className="lg:col-span-5 bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-5 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-[#f1f5f9] pb-3">
              <div>
                <span className="section-label block">Optical Backscatter Crop</span>
                <h3 className="text-base font-bold text-[#0f172a] font-display mt-0.5 flex items-center gap-1.5">
                  <Scan className="w-4 h-4 text-[#1d4ed8]" />
                  Acoustic Signature Crop ({activeContact.contact_id})
                </h3>
              </div>
              <span className="text-xs font-mono font-bold px-2.5 py-1 rounded-full bg-[#f8fafc] border border-[#e2e8f0] text-[#0f172a]">
                {bboxWidth} × {bboxHeight} px
              </span>
            </div>

            {/* High-Resolution Optical Crop Container */}
            <div className="aspect-4/3 rounded-2xl bg-[#050a14] border border-slate-800 relative overflow-hidden shadow-xl flex items-center justify-center group">
              {survey ? (
                <img
                  src={survey.processed_image_url || survey.raw_image_url}
                  alt="Acoustic Target Crop"
                  className="w-full h-full object-cover scale-[1.8] filter contrast-125"
                  style={{
                    objectPosition: `${(activeContact.bbox.x1 / (survey.image_width || 1280)) * 100}% ${(activeContact.bbox.y1 / (survey.image_height || 1800)) * 100}%`
                  }}
                />
              ) : (
                <div className="text-slate-500 font-mono text-xs">No Acoustic Image Available</div>
              )}

              {/* Acoustic Confidence Bar Overlay on Image (Top-Left) */}
              <div className="absolute top-3 left-3 bg-slate-900/85 backdrop-blur-md border border-slate-700/70 rounded-lg px-2.5 py-1.5 shadow-lg flex items-center gap-2 z-10 pointer-events-none">
                <div className="flex flex-col gap-1">
                  <div className="flex items-center justify-between gap-3 text-[10px] font-bold text-white font-mono">
                    <span className="text-slate-300">CONFIDENCE</span>
                    <span className="text-cyan-300">{Math.round(activeContact.confidence * 100)}%</span>
                  </div>
                  <div className="w-20 bg-slate-700 h-1.5 rounded-full overflow-hidden">
                    <div 
                      className="bg-gradient-to-r from-cyan-400 to-blue-500 h-full rounded-full transition-all duration-300"
                      style={{ width: `${Math.round(activeContact.confidence * 100)}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Targeting Reticle & ID Tag Overlay */}
              <div className="absolute inset-5 border-2 border-cyan-400/90 rounded-sm pointer-events-none shadow-2xl">
                <div className={`absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-0.5 text-white font-mono font-bold text-[10px] rounded-full shadow-md whitespace-nowrap ${
                  isHigh ? 'bg-rose-600' : 'bg-[#1d4ed8]'
                }`}>
                  {activeContact.contact_id} • {Math.round(activeContact.confidence * 100)}% CONF
                </div>
                {/* Crosshairs */}
                <div className="absolute top-1/2 left-0 right-0 h-px bg-cyan-400/40" />
                <div className="absolute top-0 bottom-0 left-1/2 w-px bg-cyan-400/40" />
              </div>
            </div>

            {/* View Full Sonar Image Link */}
            <div className="flex justify-center pt-1">
              <button
                type="button"
                onClick={onNavigateToAnalysis}
                className="text-xs font-bold text-[#1d4ed8] hover:text-[#1e40af] hover:underline flex items-center gap-1.5 cursor-pointer py-1 px-3 rounded-full hover:bg-blue-50 transition-colors"
              >
                <Camera className="w-3.5 h-3.5" />
                <span>View Full Sonar Image</span>
              </button>
            </div>
          </div>

          {/* Physical Acoustic Characteristics Card */}
          <div className="p-4 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] space-y-2.5 text-xs">
            <div className="flex justify-between items-center">
              <span className="text-[#64748b] font-medium">Acoustic Shadow Deficit:</span>
              <span className="font-bold text-emerald-700 bg-emerald-50 px-2.5 py-0.5 rounded-full border border-emerald-100">
                MATCHED (High-Deficit Void)
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#64748b] font-medium">Seabed Backscatter Floor:</span>
              <span className="font-bold text-[#0f172a]">Sandy / Gravel Sediment</span>
            </div>
            <div className="flex justify-between items-center pt-2 border-t border-[#e2e8f0]">
              <span className="text-[#64748b] font-medium">Slant Bounding Box:</span>
              <span className="font-mono font-bold text-[#0f172a]">
                [{activeContact.bbox.x1}, {activeContact.bbox.y1}, {activeContact.bbox.x2}, {activeContact.bbox.y2}]
              </span>
            </div>
          </div>
        </div>

        {/* Right Column (7 Cols): Candidate Telemetry & Triage Buttons */}
        <div className="lg:col-span-7 space-y-6">
          
          {/* Candidate Telemetry Grid */}
          <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-4">
            <div className="border-b border-[#f1f5f9] pb-3 flex items-center justify-between">
              <div>
                <span className="section-label block">Target Telemetry</span>
                <h3 className="text-base font-bold text-[#0f172a] font-display mt-0.5">
                  Physical & Spatial Properties
                </h3>
              </div>
              <span className="text-xs font-semibold px-3 py-1 rounded-full bg-slate-100 text-[#64748b]">
                WGS-84 Datum
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="p-4 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0]">
                <span className="text-[10px] text-[#64748b] font-bold uppercase tracking-wider block">AI Confidence</span>
                <div className="text-2xl font-extrabold text-[#0f172a] font-display mt-1">
                  {Math.round(activeContact.confidence * 100)}%
                </div>
                <div className="text-[11px] text-[#1d4ed8] font-medium mt-0.5">yolov8n-baseline</div>
              </div>

              <div className="p-4 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0]">
                <span className="text-[10px] text-[#64748b] font-bold uppercase tracking-wider block">Priority Tier</span>
                <div className={`text-2xl font-extrabold font-display mt-1 ${
                  isHigh ? 'text-rose-600' : isMedium ? 'text-amber-600' : 'text-emerald-600'
                }`}>
                  {activeContact.priority}
                </div>
                <div className="text-[11px] text-[#64748b] font-medium mt-0.5">Triage Level</div>
              </div>

              <div className="p-4 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0]">
                <span className="text-[10px] text-[#64748b] font-bold uppercase tracking-wider block">Slant Range</span>
                <div className="text-2xl font-extrabold text-[#0f172a] font-display mt-1">
                  24.6 m
                </div>
                <div className="text-[11px] text-[#64748b] font-medium mt-0.5">Towfish Offset</div>
              </div>

              <div className="p-4 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0]">
                <span className="text-[10px] text-[#64748b] font-bold uppercase tracking-wider block">Localization</span>
                <div className="text-sm font-extrabold text-[#0f172a] font-mono mt-2">
                  {activeContact.localization_status}
                </div>
                <div className="text-[11px] text-[#64748b] font-medium mt-0.5">GPS Nav Log</div>
              </div>
            </div>
          </div>

          {/* One-Click Operator Triage Actions Card */}
          <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-5">
            <div className="border-b border-[#f1f5f9] pb-3 flex items-center justify-between">
              <div>
                <span className="section-label block">Classification Action</span>
                <h3 className="text-base font-bold text-[#0f172a] font-display mt-0.5">
                  One-Click Operator Triage Decisions
                </h3>
              </div>
              <span className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${
                activeContact.review_status === 'CONFIRMED'
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  : activeContact.review_status === 'FALSE_POSITIVE'
                  ? 'bg-rose-50 text-rose-700 border-rose-200'
                  : activeContact.review_status === 'UNCERTAIN'
                  ? 'bg-amber-50 text-amber-700 border-amber-200'
                  : 'bg-blue-50 text-[#1d4ed8] border-blue-200'
              }`}>
                Current: {activeContact.review_status.replace('_', ' ')}
              </span>
            </div>

            {/* 3 Decision Action Cards with Keyboard Shortcut Hints */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {/* Confirm Contact */}
              <button
                onClick={() => handleAction('CONFIRMED')}
                disabled={submitting}
                className="p-4 rounded-2xl bg-emerald-50 hover:bg-emerald-100/80 border border-emerald-200 text-emerald-900 transition-all duration-200 flex flex-col items-center text-center gap-2 cursor-pointer shadow-tactile hover:-translate-y-0.5"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-emerald-600 shadow-xs">
                  <CheckCircle2 className="w-5 h-5" />
                </div>
                <div>
                  <div className="font-bold text-xs uppercase tracking-wide">Confirm Debris</div>
                  <div className="text-[11px] text-emerald-700 mt-0.5">Validated Target</div>
                  <div className="text-[10px] font-mono font-semibold text-emerald-700/90 mt-1 bg-emerald-100/80 px-2 py-0.5 rounded-full inline-block border border-emerald-300/60">
                    Press 1
                  </div>
                </div>
              </button>

              {/* False Alarm / Clutter */}
              <button
                onClick={() => handleAction('FALSE_POSITIVE')}
                disabled={submitting}
                className="p-4 rounded-2xl bg-rose-50 hover:bg-rose-100/80 border border-rose-200 text-rose-900 transition-all duration-200 flex flex-col items-center text-center gap-2 cursor-pointer shadow-tactile hover:-translate-y-0.5"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-rose-600 shadow-xs">
                  <XCircle className="w-5 h-5" />
                </div>
                <div>
                  <div className="font-bold text-xs uppercase tracking-wide">False Alarm</div>
                  <div className="text-[11px] text-rose-700 mt-0.5">Geological Clutter</div>
                  <div className="text-[10px] font-mono font-semibold text-rose-700/90 mt-1 bg-rose-100/80 px-2 py-0.5 rounded-full inline-block border border-rose-300/60">
                    Press 2
                  </div>
                </div>
              </button>

              {/* Needs Review */}
              <button
                onClick={() => handleAction('UNCERTAIN')}
                disabled={submitting}
                className="p-4 rounded-2xl bg-amber-50 hover:bg-amber-100/80 border border-amber-200 text-amber-900 transition-all duration-200 flex flex-col items-center text-center gap-2 cursor-pointer shadow-tactile hover:-translate-y-0.5"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-amber-600 shadow-xs">
                  <HelpCircle className="w-5 h-5" />
                </div>
                <div>
                  <div className="font-bold text-xs uppercase tracking-wide">Needs Review</div>
                  <div className="text-[11px] text-amber-700 mt-0.5">Secondary ROV Pass</div>
                  <div className="text-[10px] font-mono font-semibold text-amber-700/90 mt-1 bg-amber-100/80 px-2 py-0.5 rounded-full inline-block border border-amber-300/60">
                    Press 3
                  </div>
                </div>
              </button>
            </div>

            {/* Operator Notes Input & Save Action */}
            <div className="space-y-3 pt-2">
              <label className="section-label block">
                Operator Observations & Hydrographic Log Notes
              </label>
              <textarea
                value={operatorNote}
                onChange={(e) => setOperatorNote(e.target.value)}
                placeholder="Enter acoustic signature observations, wreck structural integrity, or diver notes..."
                rows={3}
                className="w-full p-4 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] focus:border-blue-600 focus:ring-1 focus:ring-blue-600 text-[#0f172a] text-xs placeholder:text-[#64748b] focus:outline-none transition-all"
              />

              {/* Timestamp on Operator Notes */}
              <div className="flex items-center justify-between text-[11px] text-[#64748b] px-1">
                <span className="flex items-center gap-1 font-mono">
                  <Clock className="w-3.5 h-3.5 text-[#64748b]" />
                  Last saved: {lastSavedTime}
                </span>
                <span>{operatorNote.length} characters</span>
              </div>

              <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pt-1">
                {/* View on GIS Map Button */}
                <button
                  type="button"
                  onClick={onNavigateToMap}
                  className="px-5 py-2.5 rounded-full bg-[#1d4ed8] hover:bg-[#1e40af] text-white text-xs font-bold flex items-center justify-center gap-2 transition-all duration-200 shadow-tactile shadow-blue-glow cursor-pointer hover:scale-[1.02] active:scale-[0.98] group"
                >
                  <Compass className="w-4 h-4 text-white" />
                  <span>View on GIS Map</span>
                  <ArrowRight className="w-3.5 h-3.5 text-white group-hover:translate-x-1 transition-transform" />
                </button>

                <div className="flex items-center gap-2.5 justify-end">
                  {/* Save & Continue Button */}
                  <button
                    type="button"
                    onClick={handleSaveAndContinue}
                    disabled={submitting}
                    className="px-5 py-2.5 rounded-full bg-[#1d4ed8] hover:bg-[#1e40af] text-white text-xs font-bold flex items-center justify-center gap-2 transition-all duration-200 shadow-tactile shadow-blue-glow cursor-pointer hover:scale-[1.02] active:scale-[0.98]"
                  >
                    <Save className="w-3.5 h-3.5" />
                    <span>Save & Continue</span>
                  </button>

                  {/* Skip to Next Button */}
                  <button
                    type="button"
                    onClick={handleSkipNext}
                    className="px-4 py-2.5 rounded-full bg-[#f8fafc] hover:bg-slate-100 border border-[#e2e8f0] text-[#0f172a] text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors cursor-pointer shadow-tactile"
                  >
                    <SkipForward className="w-3.5 h-3.5 text-[#64748b]" />
                    <span>Skip to Next</span>
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Audit History Log Card */}
          <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-4">
            <div className="border-b border-[#f1f5f9] pb-3 flex items-center justify-between">
              <div>
                <span className="section-label block">Audit Trail</span>
                <h3 className="text-base font-bold text-[#0f172a] font-display mt-0.5 flex items-center gap-2">
                  <Clock className="w-4 h-4 text-[#64748b]" />
                  Verification Review History
                </h3>
              </div>
              <span className="text-xs font-semibold text-[#64748b]">
                Immutable Hydrographic Log
              </span>
            </div>

            <div className="space-y-2.5 text-xs">
              <div className="p-3.5 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-blue-900 font-bold text-xs shadow-xs">
                    CV
                  </div>
                  <div>
                    <span className="font-bold text-[#0f172a]">Dr. C. Vance (Lead Hydrographer)</span>
                    <div className="text-[11px] text-[#64748b]">
                      Status: <strong className="text-[#0f172a]">{activeContact.review_status.replace('_', ' ')}</strong>
                      {activeContact.review_note && ` • "${activeContact.review_note}"`}
                    </div>
                  </div>
                </div>
                <span className="text-[11px] text-[#64748b] font-mono">
                  Recorded UTC
                </span>
              </div>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
};

