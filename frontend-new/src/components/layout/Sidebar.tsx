import React from 'react';
import { 
  LayoutDashboard, 
  Waves, 
  CheckCircle2, 
  Compass, 
  Cpu, 
  FileText, 
  Radio, 
  ChevronRight,
} from 'lucide-react';
import { ActiveScreen } from './MainLayout';

interface SidebarProps {
  activeScreen: ActiveScreen;
  onSelectScreen: (screen: ActiveScreen) => void;
  surveyFilename?: string;
  totalContactsCount?: number;
  highPriorityCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeScreen,
  onSelectScreen,
  surveyFilename,
  totalContactsCount = 0,
  highPriorityCount = 0
}) => {
  const navItems = [
    { 
      id: 'dashboard' as ActiveScreen, 
      label: 'Dashboard Overview', 
      icon: LayoutDashboard,
      badge: 'LIVE',
      isLive: true
    },
    { 
      id: 'sonar-analysis' as ActiveScreen, 
      label: 'Sonar Waterfall', 
      icon: Waves,
      badge: totalContactsCount > 0 ? `${totalContactsCount}` : undefined
    },
    { 
      id: 'contact-verification' as ActiveScreen, 
      label: 'Contact Triage', 
      icon: CheckCircle2,
      badge: highPriorityCount > 0 ? `${highPriorityCount} High` : undefined,
      badgeAlert: highPriorityCount > 0
    },
    { 
      id: 'gis-mapping' as ActiveScreen, 
      label: 'GIS Mapping & Spatial', 
      icon: Compass 
    },
    { 
      id: 'ai-pipeline' as ActiveScreen, 
      label: 'Pipeline Monitor', 
      icon: Cpu 
    },
    { 
      id: 'reports' as ActiveScreen, 
      label: 'Reports & Export', 
      icon: FileText 
    },
  ];

  return (
    <aside 
      className="group fixed left-0 top-0 bottom-0 z-50 bg-white border-r border-[#e2e8f0] shadow-[0_10px_35px_-5px_rgba(15,23,42,0.06),0_0_20px_rgba(29,78,216,0.06)] hover:shadow-[0_20px_50px_-5px_rgba(29,78,216,0.18)] flex flex-col justify-between transition-all duration-300 ease-[cubic-bezier(0.2,0.8,0.2,1)] overflow-hidden"
      style={{ width: '80px' }}
      onMouseEnter={(e) => (e.currentTarget.style.width = '280px')}
      onMouseLeave={(e) => (e.currentTarget.style.width = '80px')}
    >
      {/* Top Branding Header */}
      <div>
        <div className="h-20 flex items-center px-4 border-b border-[#e2e8f0]">
          {/* Logo Mark Bucket */}
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-[#1d4ed8] text-white shadow-[0_8px_20px_-3px_rgba(29,78,216,0.45)] transition-transform group-hover:scale-105">
            <Radio className="w-6 h-6 animate-pulse" />
          </div>

          {/* Expanded Brand Name & Tag */}
          <div className="ml-3.5 opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap overflow-hidden">
            <div className="flex items-center gap-1.5">
              <span className="font-display text-base font-extrabold tracking-tight text-[#0f172a]">
                SONAR-INTEL
              </span>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-md bg-blue-100 text-blue-800 shadow-xs">
                v2.0
              </span>
            </div>
            <p className="text-[11px] font-medium text-[#64748b]">
              Hydrographic Anomaly Triage
            </p>
          </div>
        </div>

        {/* Section Label */}
        <div className="px-5 pt-5 pb-2">
          <span className="text-[10px] font-bold tracking-wider text-[#64748b] uppercase opacity-0 group-hover:opacity-100 transition-opacity duration-200 block whitespace-nowrap font-sans">
            Workspaces
          </span>
        </div>

        {/* Navigation Items */}
        <nav className="px-3 space-y-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeScreen === item.id;

            return (
              <button
                key={item.id}
                onClick={() => onSelectScreen(item.id)}
                className={`relative w-full h-12 rounded-2xl flex items-center transition-all duration-200 font-sans cursor-pointer ${
                  isActive
                    ? 'bg-blue-50/90 text-blue-700 font-semibold shadow-[0_4px_14px_-2px_rgba(29,78,216,0.22)] border border-blue-200/80'
                    : 'text-[#64748b] hover:text-[#0f172a] hover:bg-slate-100/70 font-medium'
                }`}
                title={item.label}
              >
                {/* Active Left-Aligned Vertical Indicator Bar */}
                {isActive && (
                  <span className="absolute left-0 top-3 bottom-3 w-1.5 rounded-r-md bg-[#1d4ed8] shadow-[0_0_10px_rgba(29,78,216,0.7)]" />
                )}

                {/* Centered Icon Container */}
                <div className="relative w-14 shrink-0 flex items-center justify-center">
                  <Icon className={`w-5 h-5 transition-colors ${isActive ? 'text-[#1d4ed8]' : 'text-current'}`} />
                  
                  {/* Collapsed mini indicator badge when sidebar is not hovered */}
                  {item.isLive && (
                    <span className="absolute top-1 right-2.5 group-hover:hidden w-2 h-2 rounded-full bg-emerald-500 border border-white ring-2 ring-emerald-400/30 animate-pulse" />
                  )}
                  {item.badgeAlert && (
                    <span className="absolute top-1 right-2.5 group-hover:hidden w-2 h-2 rounded-full bg-rose-500 border border-white ring-2 ring-rose-400/30 animate-pulse" />
                  )}
                </div>

                {/* Expanded Label & Badges */}
                <div className="flex-1 min-w-0 flex items-center justify-between pr-3 pl-1 opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap overflow-hidden">
                  <span className="text-sm tracking-tight text-left truncate font-medium">{item.label}</span>

                  {item.isLive ? (
                    <span className="shrink-0 inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[10px] font-extrabold tracking-wider uppercase bg-emerald-50 text-emerald-700 border border-emerald-200 shadow-xs">
                      <span className="relative flex h-1.5 w-1.5">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500"></span>
                      </span>
                      LIVE
                    </span>
                  ) : item.badge ? (
                    <span 
                      className={`shrink-0 text-[10px] font-bold px-2 py-0.5 rounded-md font-sans ml-2 ${
                        item.badgeAlert
                          ? 'bg-rose-500 text-white shadow-xs'
                          : isActive
                          ? 'bg-blue-100 text-blue-800 shadow-xs'
                          : 'bg-slate-100 text-[#64748b]'
                      }`}
                    >
                      {item.badge}
                    </span>
                  ) : null}
                </div>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Area: Active Swath Context & Operator Profile */}
      <div className="p-3 border-t border-[#e2e8f0] space-y-3">
        {/* Active Swath Badge (Visible only on expansion) */}
        {surveyFilename && (
          <div className="hidden group-hover:block p-3 rounded-2xl bg-blue-50/50 border border-blue-100 shadow-[0_4px_14px_-2px_rgba(29,78,216,0.12)] transition-all">
            <div className="flex items-center justify-between text-[10px] font-bold text-[#64748b] uppercase tracking-wider mb-1">
              <span className="flex items-center gap-1 text-[#1d4ed8]">
                <span className="w-1.5 h-1.5 rounded-full bg-[#1d4ed8] animate-ping" />
                Active Swath
              </span>
              <span className="text-[10px] text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded-md font-semibold">
                READY
              </span>
            </div>
            <div className="text-xs font-semibold text-[#0f172a] truncate font-sans" title={surveyFilename}>
              {surveyFilename}
            </div>
          </div>
        )}

        {/* User Profile Avatar Card */}
        <div className="flex items-center h-12 px-1 rounded-2xl hover:bg-slate-100/70 transition-colors cursor-pointer">
          {/* Avatar with live status indicator */}
          <div className="w-14 shrink-0 flex items-center justify-center">
            <div className="relative flex h-10 w-10 items-center justify-center rounded-full bg-blue-50 border border-blue-200 text-[#1d4ed8] font-bold text-xs shadow-xs">
              CV
              <span className="absolute bottom-0 right-0 w-2.5 h-2.5 rounded-full bg-emerald-500 border-2 border-white" />
            </div>
          </div>

          {/* User Details (Expanded) */}
          <div className="ml-1 opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap overflow-hidden flex-1">
            <div className="text-xs font-bold text-[#0f172a] leading-tight">Dr. C. Vance</div>
            <div className="text-[11px] text-[#64748b]">Lead Hydrographer</div>
          </div>

          <ChevronRight className="w-4 h-4 text-[#64748b] opacity-0 group-hover:opacity-100 transition-opacity duration-300 pr-2 shrink-0" />
        </div>
      </div>
    </aside>
  );
};
