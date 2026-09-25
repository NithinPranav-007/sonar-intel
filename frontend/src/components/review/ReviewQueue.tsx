import React from 'react';
import { Contact } from '../../types/detection';
import { ListFilter } from 'lucide-react';

interface ReviewQueueProps {
  contacts: Contact[];
  selectedContact: Contact | null;
  onSelectContact: (contact: Contact) => void;
}

export const ReviewQueue: React.FC<ReviewQueueProps> = ({
  contacts,
  selectedContact,
  onSelectContact
}) => {
  return (
    <div className="h-44 border-t border-[#1a2f4c] bg-[#070e1a] flex flex-col select-none">
      <div className="px-3 py-1.5 border-b border-[#14233a] flex items-center justify-between text-[11px] font-mono text-slate-400">
        <div className="flex items-center gap-1.5">
          <ListFilter className="w-3.5 h-3.5 text-cyan-400" />
          <span className="font-bold text-slate-200">CONTACT QUEUE</span>
        </div>
        <span>{contacts.length} CANDIDATES</span>
      </div>

      <div className="flex-1 overflow-x-auto flex items-center p-2 gap-2">
        {contacts.map((c) => {
          const isSelected = selectedContact?.contact_id === c.contact_id;
          let priorityBorder = 'border-[#1a2f4c]';
          let priorityBg = 'bg-[#0b1626]';
          if (c.priority === 'HIGH') {
            priorityBorder = isSelected ? 'border-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]' : 'border-red-900/60';
            priorityBg = isSelected ? 'bg-red-950/40' : 'bg-[#0c1626]';
          } else if (c.priority === 'MEDIUM') {
            priorityBorder = isSelected ? 'border-amber-500 shadow-[0_0_10px_rgba(245,158,11,0.5)]' : 'border-amber-900/60';
            priorityBg = isSelected ? 'bg-amber-950/40' : 'bg-[#0c1626]';
          } else {
            priorityBorder = isSelected ? 'border-cyan-400 shadow-[0_0_10px_rgba(6,182,212,0.5)]' : 'border-[#1a2f4c]';
          }

          return (
            <div
              key={c.contact_id}
              onClick={() => onSelectContact(c)}
              className={`w-44 h-full p-2 rounded border cursor-pointer flex flex-col justify-between transition-all shrink-0 font-mono text-[11px] ${priorityBorder} ${priorityBg}`}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-100">{c.contact_id}</span>
                <span className={`text-[9px] px-1.5 py-0.2 rounded font-bold ${
                  c.priority === 'HIGH' ? 'text-red-400 bg-red-950' :
                  c.priority === 'MEDIUM' ? 'text-amber-400 bg-amber-950' : 'text-sky-400 bg-sky-950'
                }`}>
                  {c.priority}
                </span>
              </div>

              <div className="space-y-0.5 text-[10px] text-slate-400">
                <div className="flex justify-between">
                  <span>AI Conf:</span>
                  <span className="text-cyan-300 font-bold">{Math.round(c.confidence * 100)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Shadow:</span>
                  <span className="text-amber-300">{Math.round(c.shadow_evidence * 100)}%</span>
                </div>
              </div>

              <div className="pt-1 border-t border-[#14233a] flex items-center justify-between text-[9px]">
                <span className="text-slate-500">{c.localization_status}</span>
                <span className={
                  c.review_status === 'CONFIRMED' ? 'text-emerald-400 font-bold' :
                  c.review_status === 'FALSE_POSITIVE' ? 'text-red-400' :
                  c.review_status === 'UNCERTAIN' ? 'text-amber-400' : 'text-slate-400'
                }>
                  {c.review_status.replace('_', ' ')}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
