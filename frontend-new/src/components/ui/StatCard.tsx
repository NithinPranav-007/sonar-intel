import React from 'react';
import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  icon: LucideIcon;
  trend?: {
    value: string;
    isUp?: boolean;
    isNeutral?: boolean;
  };
  onClick?: () => void;
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  subtext,
  icon: Icon,
  trend,
  onClick,
}) => {
  return (
    <div
      onClick={onClick}
      className={`group relative overflow-hidden bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft hover:shadow-card-hover transition-all duration-300 ${
        onClick ? 'cursor-pointer hover:-translate-y-1' : ''
      }`}
    >
      {/* Absolute Decorative Glow Gradient */}
      <div 
        className="absolute -right-8 -bottom-8 w-36 h-36 bg-[#1d4ed8]/5 rounded-full blur-3xl pointer-events-none transition-all duration-500 group-hover:bg-[#1d4ed8]/20 group-hover:scale-125" 
      />

      {/* Top Header: Label & Icon Bucket */}
      <div className="relative z-10 flex items-start justify-between gap-4">
        <div className="space-y-1 min-w-0">
          <span className="section-label block truncate">
            {label}
          </span>
          <div className="text-3xl font-extrabold tracking-tight text-[#0f172a] font-display">
            {value}
          </div>
        </div>

        {/* Interactive Icon Bucket (Majestic Blue Theme) */}
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-blue-50/70 border border-blue-100 text-[#1d4ed8] transition-all duration-300 group-hover:bg-[#1d4ed8] group-hover:text-white group-hover:border-[#1d4ed8] group-hover:shadow-[0_8px_20px_-3px_rgba(29,78,216,0.4)]">
          <Icon className="w-5 h-5 transition-transform duration-300 group-hover:scale-110" />
        </div>
      </div>

      {/* Footer / Trend Tracker */}
      <div className="relative z-10 mt-5 flex items-center justify-between pt-3.5 border-t border-[#f1f5f9] text-xs">
        {trend ? (
          <span
            className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs font-semibold transition-colors ${
              trend.isNeutral
                ? 'bg-slate-100 text-slate-700'
                : trend.isUp
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                : 'bg-rose-50 text-rose-700 border border-rose-200'
            }`}
          >
            {trend.value}
          </span>
        ) : (
          <span className="text-xs text-[#64748b] font-medium">Calibrated</span>
        )}

        {subtext && (
          <span className="text-xs font-medium text-[#64748b] truncate max-w-[180px]" title={subtext}>
            {subtext}
          </span>
        )}
      </div>
    </div>
  );
};
