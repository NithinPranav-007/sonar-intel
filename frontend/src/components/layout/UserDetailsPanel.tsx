import React, { useEffect } from 'react';
import { 
  X, 
  Mail, 
  Building2, 
  Activity, 
  ShieldCheck, 
  User, 
  Settings, 
  LogOut,
  Radio
} from 'lucide-react';

interface UserDetailsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  surveyFilename?: string;
}

export const UserDetailsPanel: React.FC<UserDetailsPanelProps> = ({
  isOpen,
  onClose,
  surveyFilename = 'viator_04_test_wreck.png',
}) => {
  // Close on Escape key press
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
    }
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  return (
    <>
      {/* Translucent Backdrop Overlay (Click outside to close) */}
      <div 
        className={`fixed inset-0 bg-black/50 backdrop-blur-xs z-50 transition-opacity duration-300 ${
          isOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        }`}
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Slide-in Drawer Panel from the Right */}
      <aside
        className={`fixed right-0 top-0 bottom-0 z-50 w-[320px] h-full bg-[#0f172a] text-slate-100 border-l border-slate-800 shadow-[-10px_0_30px_rgba(0,0,0,0.4)] p-6 overflow-y-auto flex flex-col justify-between transition-transform duration-300 ease-in-out font-sans ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
        role="dialog"
        aria-modal="true"
        aria-label="User Profile Details"
      >
        <div className="space-y-6">
          {/* Top Row: Header & Close Button (X) */}
          <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
            <div className="flex items-center gap-2">
              <User className="w-4 h-4 text-blue-400" />
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Operator Profile
              </span>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-xl hover:bg-slate-800 text-slate-400 hover:text-white transition-colors cursor-pointer"
              title="Close Panel"
              aria-label="Close user profile"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* 1. User Avatar + Name */}
          <div className="flex flex-col items-center text-center space-y-3 pt-1">
            {/* Large 60px Avatar with Green Online Status Dot */}
            <div className="relative">
              <div className="w-[60px] h-[60px] rounded-full bg-blue-600/25 border-2 border-blue-500 text-blue-400 font-extrabold text-xl flex items-center justify-center shadow-lg font-display">
                CV
              </div>
              <span className="absolute bottom-0 right-0 w-3.5 h-3.5 rounded-full bg-emerald-500 border-2 border-[#0f172a] ring-2 ring-emerald-400/30" />
            </div>

            <div>
              <h3 className="text-base font-bold text-white tracking-tight font-display">
                Dr. Clara Vance
              </h3>
              <p className="text-xs text-slate-400 font-medium">
                Lead Hydrographer
              </p>
              <div className="inline-flex items-center gap-1.5 mt-2 px-2.5 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 text-[11px] font-semibold">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span>Online</span>
              </div>
            </div>
          </div>

          <div className="h-px bg-slate-800" />

          {/* 2. Contact Information */}
          <div className="space-y-3">
            <h4 className="text-sm font-semibold text-white tracking-tight flex items-center gap-1.5">
              <Mail className="w-4 h-4 text-blue-400" />
              <span>Contact Information</span>
            </h4>
            <div className="space-y-2.5 text-left">
              <div>
                <span className="text-xs text-slate-400 block mb-0.5">Email</span>
                <span className="text-sm text-slate-100 font-mono font-medium select-all">
                  c.vance@sonar-intel.gov.in
                </span>
              </div>
              <div>
                <span className="text-xs text-slate-400 block mb-0.5">Organization</span>
                <span className="text-sm text-slate-100 font-medium">
                  National Centre for Coastal Research
                </span>
              </div>
            </div>
          </div>

          <div className="h-px bg-slate-800" />

          {/* 3. Activity Summary */}
          <div className="space-y-3">
            <h4 className="text-sm font-semibold text-white tracking-tight flex items-center gap-1.5">
              <Activity className="w-4 h-4 text-blue-400" />
              <span>Activity Summary</span>
            </h4>
            <div className="space-y-2.5 text-left">
              <div>
                <span className="text-xs text-slate-400 block mb-0.5">Current Survey</span>
                <div className="flex items-center gap-1.5 text-sm text-slate-100 font-mono truncate">
                  <Radio className="w-3.5 h-3.5 text-blue-400 shrink-0 animate-pulse" />
                  <span className="truncate" title={surveyFilename}>{surveyFilename}</span>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2 pt-1">
                <div className="p-2 rounded-xl bg-slate-800/60 border border-slate-700/60">
                  <span className="text-[10px] text-slate-400 font-bold uppercase block">Reviewed</span>
                  <span className="text-sm font-extrabold font-mono text-white">74</span>
                </div>
                <div className="p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                  <span className="text-[10px] text-emerald-400 font-bold uppercase block">Confirmed</span>
                  <span className="text-sm font-extrabold font-mono text-emerald-400">8</span>
                </div>
              </div>
              <div>
                <span className="text-xs text-slate-400 block mb-0.5">Last Active</span>
                <span className="text-sm text-slate-100 font-medium">2 minutes ago</span>
              </div>
            </div>
          </div>

          <div className="h-px bg-slate-800" />

          {/* 4. Account Details */}
          <div className="space-y-3">
            <h4 className="text-sm font-semibold text-white tracking-tight flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-blue-400" />
              <span>Account Details</span>
            </h4>
            <div className="space-y-2.5 text-left">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">Member Since:</span>
                <span className="text-sm text-slate-100 font-medium">2024</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">Role:</span>
                <span className="text-sm text-slate-100 font-medium">Lead Hydrographer</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">Access Level:</span>
                <span className="px-2 py-0.5 rounded-md bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-xs font-semibold">
                  Full Access
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* 5. Action Buttons (at bottom) */}
        <div className="pt-6 space-y-2 border-t border-slate-800 mt-6">
          <button
            type="button"
            className="w-full py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-semibold text-xs border border-slate-700 transition-colors flex items-center justify-center gap-2 cursor-pointer shadow-xs"
          >
            <User className="w-3.5 h-3.5" />
            <span>View Full Profile</span>
          </button>

          <button
            type="button"
            className="w-full py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-semibold text-xs border border-slate-700 transition-colors flex items-center justify-center gap-2 cursor-pointer shadow-xs"
          >
            <Settings className="w-3.5 h-3.5" />
            <span>Settings</span>
          </button>

          <button
            type="button"
            className="w-full py-2.5 px-4 rounded-xl bg-rose-600/20 hover:bg-rose-600 text-rose-400 hover:text-white font-semibold text-xs border border-rose-500/30 transition-colors flex items-center justify-center gap-2 cursor-pointer"
          >
            <LogOut className="w-3.5 h-3.5" />
            <span>Sign Out</span>
          </button>
        </div>
      </aside>
    </>
  );
};
