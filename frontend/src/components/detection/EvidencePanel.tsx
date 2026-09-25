import React from 'react';
import { Contact } from '../../types/detection';
import { CheckCircle2, AlertCircle, HelpCircle } from 'lucide-react';

interface EvidencePanelProps {
  contact: Contact;
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({ contact }) => {
  const hasStrongContrast = contact.context_score >= 0.65;
  const hasStrongShadow = contact.shadow_evidence >= 0.60;
  const hasGoodQuality = contact.data_quality >= 0.70;
  const isHighConfidence = contact.confidence >= 0.70;

  return (
    <div className="p-3 rounded bg-[#070e1a] border border-[#1a2f4c] space-y-2 select-none">
      <div className="flex items-center justify-between border-b border-[#14233a] pb-1.5">
        <span className="text-[11px] font-bold font-mono tracking-wider text-cyan-400 uppercase">
          WHY FLAGGED?
        </span>
        <span className="text-[10px] text-slate-500 font-mono">ACOUSTIC EVIDENCE</span>
      </div>

      <div className="space-y-1.5 text-xs font-sans">
        <div className="flex items-center gap-2">
          {isHighConfidence ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          ) : (
            <HelpCircle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          )}
          <span className={isHighConfidence ? 'text-slate-200' : 'text-slate-400'}>
            Object-like acoustic highlight return ({(contact.confidence * 100).toFixed(0)}% confidence)
          </span>
        </div>

        <div className="flex items-center gap-2">
          {hasStrongContrast ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          ) : (
            <HelpCircle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          )}
          <span className={hasStrongContrast ? 'text-slate-200' : 'text-slate-400'}>
            Strong local backscatter contrast against seabed
          </span>
        </div>

        <div className="flex items-center gap-2">
          {hasStrongShadow ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          ) : (
            <HelpCircle className="w-3.5 h-3.5 text-slate-400 shrink-0" />
          )}
          <span className={hasStrongShadow ? 'text-slate-200' : 'text-slate-400'}>
            {hasStrongShadow
              ? `Supporting down-range acoustic shadow (${(contact.shadow_evidence * 100).toFixed(0)}% deficit)`
              : 'Subtle acoustic shadow (typical for low-profile / buried targets)'}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {hasGoodQuality ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          ) : (
            <AlertCircle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          )}
          <span className={hasGoodQuality ? 'text-slate-200' : 'text-slate-400'}>
            Swath SNR and dynamic range within operational threshold
          </span>
        </div>
      </div>
    </div>
  );
};
