import React, { useState } from 'react';
import { Contact, ReviewStatus } from '../../types/detection';
import { Check, X, HelpCircle, MessageSquare } from 'lucide-react';

interface ReviewActionsProps {
  contact: Contact;
  onSubmitReview: (contactId: string, status: ReviewStatus, note?: string) => Promise<void>;
}

export const ReviewActions: React.FC<ReviewActionsProps> = ({ contact, onSubmitReview }) => {
  const [note, setNote] = useState<string>(contact.review_note || '');
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [showNoteInput, setShowNoteInput] = useState<boolean>(Boolean(contact.review_note));

  const handleAction = async (status: ReviewStatus) => {
    setSubmitting(true);
    try {
      await onSubmitReview(contact.contact_id, status, note.trim() || undefined);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-3 font-mono text-xs select-none">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-bold text-slate-300">HUMAN TRIAGE VERDICT</span>
        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
          contact.review_status === 'CONFIRMED'
            ? 'bg-emerald-950 text-emerald-300 border border-emerald-700'
            : contact.review_status === 'FALSE_POSITIVE'
            ? 'bg-red-950 text-red-300 border border-red-700'
            : contact.review_status === 'UNCERTAIN'
            ? 'bg-amber-950 text-amber-300 border border-amber-700'
            : 'bg-cyan-950 text-cyan-300 border border-cyan-800'
        }`}>
          {contact.review_status.replace('_', ' ')}
        </span>
      </div>

      {/* Review Triage Action Buttons */}
      <div className="grid grid-cols-3 gap-2">
        <button
          onClick={() => handleAction('CONFIRMED')}
          disabled={submitting}
          className={`py-2 px-2 rounded font-bold flex items-center justify-center gap-1.5 transition-all ${
            contact.review_status === 'CONFIRMED'
              ? 'bg-emerald-600 text-slate-950 shadow-[0_0_12px_#10b981]'
              : 'bg-emerald-950/80 hover:bg-emerald-900 border border-emerald-700/60 text-emerald-300'
          }`}
        >
          <Check className="w-3.5 h-3.5" />
          <span>CONFIRM</span>
        </button>

        <button
          onClick={() => handleAction('FALSE_POSITIVE')}
          disabled={submitting}
          className={`py-2 px-2 rounded font-bold flex items-center justify-center gap-1.5 transition-all ${
            contact.review_status === 'FALSE_POSITIVE'
              ? 'bg-red-600 text-slate-950 shadow-[0_0_12px_#ef4444]'
              : 'bg-red-950/80 hover:bg-red-900 border border-red-700/60 text-red-300'
          }`}
        >
          <X className="w-3.5 h-3.5" />
          <span>FALSE POS</span>
        </button>

        <button
          onClick={() => handleAction('UNCERTAIN')}
          disabled={submitting}
          className={`py-2 px-2 rounded font-bold flex items-center justify-center gap-1.5 transition-all ${
            contact.review_status === 'UNCERTAIN'
              ? 'bg-amber-600 text-slate-950 shadow-[0_0_12px_#f59e0b]'
              : 'bg-amber-950/80 hover:bg-amber-900 border border-amber-700/60 text-amber-300'
          }`}
        >
          <HelpCircle className="w-3.5 h-3.5" />
          <span>UNCERTAIN</span>
        </button>
      </div>

      {/* Reviewer Note Toggle and Input */}
      <div>
        {!showNoteInput ? (
          <button
            onClick={() => setShowNoteInput(true)}
            className="text-[11px] text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-sans"
          >
            <MessageSquare className="w-3 h-3" />
            <span>+ Add surveyor remark / note</span>
          </button>
        ) : (
          <div className="space-y-1.5 font-sans">
            <textarea
              rows={2}
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="e.g., Distinct 3.2m acoustic shadow cast down-range; linear edge suggests lost crate/cargo."
              className="w-full bg-[#050b14] border border-[#1a2f4c] rounded p-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-sans"
            />
            <div className="flex justify-end">
              <button
                onClick={() => handleAction(contact.review_status !== 'AI_CANDIDATE' ? contact.review_status : 'CONFIRMED')}
                disabled={submitting}
                className="px-2.5 py-1 bg-[#132845] hover:bg-[#1a3861] border border-cyan-800 text-cyan-200 rounded text-[11px] font-mono"
              >
                Save Remark
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
