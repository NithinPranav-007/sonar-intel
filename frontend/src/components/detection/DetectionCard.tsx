import React from 'react';
import { ConfidencePanel } from './ConfidencePanel';
import { EvidencePanel } from './EvidencePanel';
import { ReviewActions } from '../review/ReviewActions';
import { Contact, ReviewStatus } from '../../types/detection';

interface DetectionCardProps {
  contact: Contact | null;
  onSubmitReview: (contactId: string, status: ReviewStatus, note?: string) => Promise<void>;
}

export const DetectionCard: React.FC<DetectionCardProps> = ({ contact, onSubmitReview }) => {
  if (!contact) {
    return (
      <div className="h-full flex items-center justify-center p-6 text-center text-slate-500 text-xs font-mono select-none">
        <div>
          <div className="w-10 h-10 mx-auto mb-2 rounded-full border border-dashed border-slate-700 flex items-center justify-center text-slate-600">
            ?
          </div>
          <p>No contact selected.</p>
          <p className="text-[11px] text-slate-600 mt-1">Select a candidate on the sonar swath or map to inspect evidence.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col justify-between p-4 space-y-4 bg-[#0a1322] border-t border-[#1a2f4c] overflow-y-auto">
      {/* 1. Metrics & Priority */}
      <ConfidencePanel contact={contact} />

      {/* 2. Acoustic Evidence: WHY FLAGGED? */}
      <EvidencePanel contact={contact} />

      {/* 3. Human Triage Actions */}
      <ReviewActions contact={contact} onSubmitReview={onSubmitReview} />
    </div>
  );
};
