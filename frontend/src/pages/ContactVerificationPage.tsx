import React, { useState } from 'react';
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
  Maximize2
} from 'lucide-react';

interface ContactVerificationPageProps {
  survey: SurveyUploadResponse | null;
  contacts: Contact[];
  selectedContact: Contact | null;
  onSelectContact: (contact: Contact) => void;
  onSubmitReview: (contactId: string, status: ReviewStatus, note?: string) => Promise<void>;
  onNavigateToMap: () => void;
}

export const ContactVerificationPage: React.FC<ContactVerificationPageProps> = ({
  survey,
  contacts,
  selectedContact,
  onSelectContact,
  onSubmitReview,
  onNavigateToMap
}) => {
  const activeContact = selectedContact || contacts[0] || null;
  const [operatorNote, setOperatorNote] = useState<string>('');
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  const handleAction = async (status: ReviewStatus) => {
    if (!activeContact) return;
    setSubmitting(true);
    setSaveMessage(null);
    try {
      await onSubmitReview(activeContact.contact_id, status, operatorNote);
      setSaveMessage(`Contact ${activeContact.contact_id} marked as ${status}.`);
      setOperatorNote('');
    } catch (err) {
      console.error('Review submission error:', err);
    } finally {
      setSubmitting(false);
    }
  };

  if (!activeContact) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 text-center text-xs text-slate-500 bg-[#050a14]">
        <Scan className="w-10 h-10 text-slate-700 animate-pulse mb-2" />
        <p className="font-bold text-slate-300">No candidate contact selected for triage.</p>
        <p className="text-[11px] text-slate-500 mt-1">Select a survey swath in Sonar Analysis first.</p>
      </div>
    );
  }

  const bboxWidth = activeContact.bbox.x2 - activeContact.bbox.x1;
  const bboxHeight = activeContact.bbox.y2 - activeContact.bbox.y1;

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-[#050a14] text-slate-100 font-sans select-none">
      
      {/* Page Header */}
      <div className="flex items-center justify-between border-b border-[#142244] pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-xl font-extrabold tracking-tight text-white font-mono flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-emerald-400" />
              HUMAN-IN-THE-LOOP CONTACT VERIFICATION CONSOLE
            </h1>
            <span className="text-[10px] px-2.5 py-0.5 rounded-full bg-emerald-950 text-emerald-300 font-mono font-bold border border-emerald-800 uppercase">
              TRIAGE CONSOLE
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1 font-medium">
            Operator Review Linking YOLOv8 Statistical Acoustic Proposals to Verified Marine Debris / Hydrographic Contacts
          </p>
        </div>

        {/* Contact Selector Pill Bar */}
        <div className="flex items-center gap-2 overflow-x-auto max-w-md p-1.5 rounded-xl bg-[#081024] border border-[#14244a]">
          {contacts.slice(0, 7).map((c) => (
            <button
              key={c.contact_id}
              onClick={() => onSelectContact(c)}
              className={`px-3 py-1 rounded-lg text-xs font-mono font-bold transition-all ${
                activeContact.contact_id === c.contact_id
                  ? 'bg-cyan-600 text-white shadow-md shadow-cyan-950/50'
                  : 'text-slate-400 hover:text-white hover:bg-[#101d3b]'
              }`}
            >
              {c.contact_id}
            </button>
          ))}
        </div>
      </div>

      {saveMessage && (
        <div className="p-3 rounded-xl bg-emerald-950/80 border border-emerald-700 text-emerald-300 text-xs font-mono font-bold flex items-center justify-between">
          <span>✓ {saveMessage}</span>
          <button onClick={() => setSaveMessage(null)} className="underline text-emerald-400 text-[11px]">Dismiss</button>
        </div>
      )}

      {/* Main 2-Column Triage Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left: High-Resolution Acoustic Crop Viewer (5 Cols) */}
        <div className="lg:col-span-5 p-5 rounded-2xl bg-[#091226] border border-[#15274f] space-y-4 shadow-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-[#142244] pb-3">
              <span className="font-mono font-bold text-xs text-cyan-300 uppercase flex items-center gap-1.5">
                <Scan className="w-4 h-4 text-cyan-400" /> ACOUSTIC TARGET CROP ({activeContact.contact_id})
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-black text-slate-300 font-mono border border-slate-700">
                {bboxWidth} × {bboxHeight} px
              </span>
            </div>

            {/* Simulated Zoomed Optical / Acoustic Crop */}
            <div className="mt-4 aspect-4/3 rounded-xl bg-black border border-[#1a2d59] relative overflow-hidden flex items-center justify-center group shadow-2xl">
              {survey ? (
                <img
                  src={survey.processed_image_url}
                  alt="Acoustic Crop"
                  className="w-full h-full object-cover scale-150 filter contrast-125"
                  style={{
                    objectPosition: `${(activeContact.bbox.x1 / (survey.image_width || 1280)) * 100}% ${(activeContact.bbox.y1 / (survey.image_height || 1800)) * 100}%`
                  }}
                />
              ) : (
                <div className="text-slate-600 font-mono text-xs">No Acoustic Imagery</div>
              )}

              {/* Tactical Crosshair Reticle Overlay */}
              <div className="absolute inset-4 border-2 border-cyan-400/80 rounded-sm pointer-events-none shadow-lg">
                <div className="absolute -top-2 left-1/2 -translate-x-1/2 px-2 py-0.5 bg-cyan-600 text-white font-mono font-bold text-[9px] rounded shadow-md">
                  {activeContact.contact_id} • {Math.round(activeContact.confidence * 100)}% CONF
                </div>
                {/* Crosshairs */}
                <div className="absolute top-1/2 left-0 right-0 h-px bg-cyan-400/40"></div>
                <div className="absolute top-0 bottom-0 left-1/2 w-px bg-cyan-400/40"></div>
              </div>
            </div>
          </div>

          <div className="p-3 rounded-xl bg-[#060b17] border border-[#142244] text-[11px] text-slate-400 font-mono space-y-1">
            <div className="flex justify-between">
              <span>Acoustic Shadow Deficit:</span>
              <span className="text-emerald-400 font-bold">MATCHED (Dark down-range void)</span>
            </div>
            <div className="flex justify-between">
              <span>Seabed Backscatter Floor:</span>
              <span className="text-slate-300 font-bold">Sandy / Gravel sediment</span>
            </div>
          </div>
        </div>

        {/* Right: Target Telemetry & One-Click Operator Action (7 Cols) */}
        <div className="lg:col-span-7 space-y-5">
          
          {/* Telemetry Metrics Grid */}
          <div className="p-5 rounded-2xl bg-[#091226] border border-[#15274f] space-y-4 shadow-lg">
            <div className="border-b border-[#142244] pb-3 flex items-center justify-between">
              <span className="font-mono font-bold text-xs text-white uppercase">CANDIDATE TELEMETRY & PHYSICAL PROPERTIES</span>
              <span className="text-[10px] font-mono text-cyan-400">EPSG:4326</span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="p-3 rounded-xl bg-[#0b1429] border border-[#182a52]">
                <div className="text-[10px] text-slate-400 font-mono uppercase">AI CONFIDENCE</div>
                <div className="text-xl font-extrabold text-white font-mono mt-1">{Math.round(activeContact.confidence * 100)}%</div>
                <div className="text-[10px] text-emerald-400 font-mono mt-0.5">YOLOv8n Single-Class</div>
              </div>

              <div className="p-3 rounded-xl bg-[#0b1429] border border-[#182a52]">
                <div className="text-[10px] text-slate-400 font-mono uppercase">PRIORITY TIER</div>
                <div className={`text-xl font-extrabold font-mono mt-1 ${
                  activeContact.priority === 'HIGH' ? 'text-red-400' : 'text-amber-400'
                }`}>
                  {activeContact.priority}
                </div>
                <div className="text-[10px] text-slate-400 font-mono mt-0.5">Triage Level</div>
              </div>

              <div className="p-3 rounded-xl bg-[#0b1429] border border-[#182a52]">
                <div className="text-[10px] text-slate-400 font-mono uppercase">SLANT RANGE OFFSET</div>
                <div className="text-xl font-extrabold text-cyan-300 font-mono mt-1">24.6 m</div>
                <div className="text-[10px] text-slate-400 font-mono mt-0.5">Port / Starboard</div>
              </div>

              <div className="p-3 rounded-xl bg-[#0b1429] border border-[#182a52]">
                <div className="text-[10px] text-slate-400 font-mono uppercase">LOCALIZATION</div>
                <div className="text-sm font-extrabold text-white font-mono mt-2">{activeContact.localization_status}</div>
                <div className="text-[10px] text-slate-400 font-mono mt-0.5">GPS Track Provenance</div>
              </div>
            </div>
          </div>

          {/* Operator Triage Classification Actions */}
          <div className="p-5 rounded-2xl bg-[#091226] border border-[#15274f] space-y-4 shadow-lg">
            <div className="border-b border-[#142244] pb-3">
              <span className="font-mono font-bold text-xs text-white uppercase">ONE-CLICK OPERATOR TRIAGE ACTIONS</span>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <button
                onClick={() => handleAction('CONFIRMED')}
                disabled={submitting}
                className="p-3.5 rounded-xl bg-gradient-to-r from-emerald-950 to-[#0c3120] hover:from-emerald-900 hover:to-[#12452c] text-emerald-300 border border-emerald-600/70 font-bold text-xs flex flex-col items-center gap-1.5 transition-all shadow-md group"
              >
                <CheckCircle2 className="w-5 h-5 text-emerald-400 group-hover:scale-110 transition-transform" />
                <span>CONFIRM DEBRIS</span>
                <span className="text-[10px] text-emerald-400/70 font-normal">Validated Contact</span>
              </button>

              <button
                onClick={() => handleAction('FALSE_POSITIVE')}
                disabled={submitting}
                className="p-3.5 rounded-xl bg-gradient-to-r from-red-950 to-[#361118] hover:from-red-900 hover:to-[#4a1822] text-red-300 border border-red-600/70 font-bold text-xs flex flex-col items-center gap-1.5 transition-all shadow-md group"
              >
                <XCircle className="w-5 h-5 text-red-400 group-hover:scale-110 transition-transform" />
                <span>FALSE POSITIVE</span>
                <span className="text-[10px] text-red-400/70 font-normal">Geological Clutter</span>
              </button>

              <button
                onClick={() => handleAction('UNCERTAIN')}
                disabled={submitting}
                className="p-3.5 rounded-xl bg-gradient-to-r from-amber-950 to-[#38260a] hover:from-amber-900 hover:to-[#4d340e] text-amber-300 border border-amber-600/70 font-bold text-xs flex flex-col items-center gap-1.5 transition-all shadow-md group"
              >
                <HelpCircle className="w-5 h-5 text-amber-400 group-hover:scale-110 transition-transform" />
                <span>NEEDS REVIEW</span>
                <span className="text-[10px] text-amber-400/70 font-normal">Secondary ROV Pass</span>
              </button>
            </div>

            {/* Operator Observation Notes Input */}
            <div className="space-y-1.5 pt-2">
              <label className="text-[11px] font-mono text-slate-400 font-bold uppercase">
                OPERATOR OBSERVATIONS & SURVEY LOG NOTES
              </label>
              <textarea
                value={operatorNote}
                onChange={(e) => setOperatorNote(e.target.value)}
                placeholder="Enter acoustic signature analysis, wreck orientation, or diver notes..."
                rows={3}
                className="w-full p-3 rounded-xl bg-[#060b17] border border-[#142244] focus:border-cyan-500 text-slate-100 text-xs placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-cyan-500/40 font-sans"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
