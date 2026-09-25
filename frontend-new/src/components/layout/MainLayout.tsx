import React from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { SurveyUploadResponse, Contact, SurveySummary } from '../../types/detection';

export type ActiveScreen = 
  | 'dashboard' 
  | 'sonar-analysis' 
  | 'contact-verification' 
  | 'gis-mapping' 
  | 'ai-pipeline' 
  | 'reports';

interface MainLayoutProps {
  children: React.ReactNode;
  activeScreen: ActiveScreen;
  onSelectScreen: (screen: ActiveScreen) => void;
  survey: SurveyUploadResponse | null;
  summary: SurveySummary | null;
  contacts: Contact[];
  analyzing: boolean;
  onLoadDemoSample: (sampleId: string) => void;
  onCustomUploadClick: () => void;
  onRunAnalysis: () => void;
}

export const MainLayout: React.FC<MainLayoutProps> = ({
  children,
  activeScreen,
  onSelectScreen,
  survey,
  contacts,
  analyzing,
  onLoadDemoSample,
  onCustomUploadClick,
  onRunAnalysis
}) => {
  const highPriorityCount = contacts.filter(c => c.priority === 'HIGH').length;

  return (
    <div className="flex h-screen w-screen bg-[#fcfcfc] text-[#1f1f1f] overflow-hidden select-none font-sans">
      
      {/* Expandable Placely Sidebar (80px collapsed, 260px hover) */}
      <Sidebar
        activeScreen={activeScreen}
        onSelectScreen={onSelectScreen}
        surveyFilename={survey?.filename}
        totalContactsCount={contacts.length}
        highPriorityCount={highPriorityCount}
      />

      {/* Main App Content Area strictly offset by 80px sidebar gutter */}
      <div 
        className="flex-1 flex flex-col h-screen overflow-hidden bg-[#fcfcfc]"
        style={{ marginLeft: '80px', width: 'calc(100vw - 80px)' }}
      >
        {/* Top Header */}
        <Header
          survey={survey}
          analyzing={analyzing}
          onRunAnalysis={onRunAnalysis}
          onCustomUploadClick={onCustomUploadClick}
          onLoadDemoSample={onLoadDemoSample}
          activeScreen={activeScreen}
        />

        {/* Scrollable Workspace Content */}
        <main className="flex-1 overflow-y-auto bg-[#fcfcfc] pb-12">
          {children}
        </main>
      </div>
    </div>
  );
};
