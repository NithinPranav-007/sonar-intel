import React from 'react';
import { Header } from './Header';
import { Sidebar, ActiveScreen } from './Sidebar';
import { SurveyUploadResponse, SurveySummary, Contact } from '../../types/detection';

interface MainLayoutProps {
  activeScreen: ActiveScreen;
  onSelectScreen: (screen: ActiveScreen) => void;
  survey: SurveyUploadResponse | null;
  summary: SurveySummary | null;
  contacts: Contact[];
  selectedContact?: Contact | null;
  analyzing: boolean;
  onLoadDemoSample: (sampleId: string) => void;
  onCustomUploadClick?: () => void;
  onRunAnalysis: () => void;
  onSelectContact?: (contact: Contact) => void;
  children: React.ReactNode;
}

export const MainLayout: React.FC<MainLayoutProps> = ({
  activeScreen,
  onSelectScreen,
  survey,
  contacts,
  selectedContact,
  analyzing,
  onLoadDemoSample,
  onCustomUploadClick,
  onRunAnalysis,
  onSelectContact,
  children
}) => {
  return (
    <div className="flex flex-col h-screen w-screen bg-[#050a14] text-slate-100 overflow-hidden font-sans">
      <Header 
        surveyId={survey?.survey_id} 
        onLoadDemoSample={onLoadDemoSample}
        onCustomUploadClick={onCustomUploadClick}
      />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          activeScreen={activeScreen}
          onSelectScreen={onSelectScreen}
          survey={survey}
          contacts={contacts}
          selectedContact={selectedContact}
          analyzing={analyzing}
          onRunAnalysis={onRunAnalysis}
          onSelectContact={onSelectContact}
        />
        <main className="flex-1 flex flex-col overflow-hidden bg-[#060b17] relative">
          {children}
        </main>
      </div>
    </div>
  );
};
