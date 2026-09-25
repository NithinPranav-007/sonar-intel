import React, { useState, useEffect } from 'react';
import { MainLayout } from '../components/layout/MainLayout';
import { ActiveScreen } from '../components/layout/Sidebar';
import { SurveyUpload } from '../components/upload/SurveyUpload';
import { useSurvey } from '../hooks/useSurvey';
import { Contact } from '../types/detection';

// The 6 Authoritative Application Pages
import { DashboardPage } from './DashboardPage';
import { SonarAnalysisPage } from './SonarAnalysisPage';
import { ContactVerificationPage } from './ContactVerificationPage';
import { GisMappingPage } from './GisMappingPage';
import { AiPipelinePage } from './AiPipelinePage';
import { ReportsPage } from './ReportsPage';

export const Dashboard: React.FC = () => {
  const [activeScreen, setActiveScreen] = useState<ActiveScreen>('dashboard');
  const [showUploadModal, setShowUploadModal] = useState<boolean>(false);

  const {
    survey,
    contacts,
    selectedContact,
    navTrack,
    summary,
    loading,
    analyzing,
    error,
    setSelectedContact,
    loadSurvey,
    runAnalysis,
    submitReview,
    loadCuratedSample,
    setError
  } = useSurvey();

  // On first load, automatically ingest the primary held-out benchmark true-positive (Viator-04)
  useEffect(() => {
    if (!survey && !loading) {
      loadCuratedSample('viator_04');
    }
  }, []);

  const handleExportGeoJSON = () => {
    if (!survey) return;
    window.open(`/api/surveys/${survey.survey_id}/geojson`, '_blank');
  };

  const handleExportCSV = () => {
    if (!survey) return;
    window.open(`/api/surveys/${survey.survey_id}/csv`, '_blank');
  };

  const handleVerifyContact = (contact: Contact) => {
    setSelectedContact(contact);
    setActiveScreen('contact-verification');
  };

  return (
    <MainLayout
      activeScreen={activeScreen}
      onSelectScreen={setActiveScreen}
      survey={survey}
      summary={summary}
      contacts={contacts}
      selectedContact={selectedContact}
      analyzing={analyzing}
      onLoadDemoSample={loadCuratedSample}
      onCustomUploadClick={() => setShowUploadModal(true)}
      onRunAnalysis={() => runAnalysis(0.20)}
      onSelectContact={setSelectedContact}
    >
      {/* System alert banner */}
      {error && (
        <div className="bg-red-50 border-b border-red-200 text-red-800 px-4 py-2 text-xs flex items-center justify-between z-30 font-sans">
          <span className="font-medium">[System Notice] {error}</span>
          <button onClick={() => setError(null)} className="underline ml-4 text-[11px] text-red-600 hover:text-red-900">Dismiss</button>
        </div>
      )}

      {/* Screen 1: Dashboard Overview */}
      {activeScreen === 'dashboard' && (
        <DashboardPage
          survey={survey}
          contacts={contacts}
          onSelectScreen={setActiveScreen}
          onSelectContact={(c) => {
            setSelectedContact(c);
            setActiveScreen('sonar-analysis');
          }}
        />
      )}

      {/* Screen 2: Sonar Analysis Workspace */}
      {activeScreen === 'sonar-analysis' && (
        <SonarAnalysisPage
          survey={survey}
          contacts={contacts}
          selectedContact={selectedContact}
          analyzing={analyzing}
          onSelectContact={setSelectedContact}
          onRunAnalysis={() => runAnalysis(0.20)}
          onVerifyContact={handleVerifyContact}
        />
      )}

      {/* Screen 3: Contact Verification Workflow */}
      {activeScreen === 'contact-verification' && (
        <ContactVerificationPage
          survey={survey}
          contacts={contacts}
          selectedContact={selectedContact}
          onSelectContact={setSelectedContact}
          onSubmitReview={submitReview}
          onNavigateToMap={() => setActiveScreen('gis-mapping')}
        />
      )}

      {/* Screen 4: GIS Mapping & Cleanup Planning */}
      {activeScreen === 'gis-mapping' && (
        <GisMappingPage
          survey={survey}
          contacts={contacts}
          selectedContact={selectedContact}
          navTrack={navTrack}
          onSelectContact={setSelectedContact}
          onNavigateToAnalysis={() => setActiveScreen('sonar-analysis')}
          onNavigateToVerify={() => setActiveScreen('contact-verification')}
          onExportGeoJSON={handleExportGeoJSON}
        />
      )}

      {/* Screen 5: AI Deep Learning Pipeline Monitor */}
      {activeScreen === 'ai-pipeline' && (
        <AiPipelinePage />
      )}

      {/* Screen 6: Reports & Export Central */}
      {activeScreen === 'reports' && (
        <ReportsPage
          survey={survey}
          contacts={contacts}
        />
      )}

      {/* Modal: Custom Sonar Upload */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs z-50 flex items-center justify-center p-4">
          <div className="bg-white border border-slate-200 rounded-lg max-w-xl w-full p-6 space-y-4 shadow-xl text-xs font-sans text-slate-900">
            <div className="flex justify-between items-center border-b border-slate-100 pb-2">
              <span className="font-bold text-slate-900 uppercase">UPLOAD CUSTOM SSS SWATH</span>
              <button 
                onClick={() => setShowUploadModal(false)}
                className="text-slate-400 hover:text-slate-700 text-lg leading-none"
              >
                &times;
              </button>
            </div>
            <SurveyUpload
              onUploadSuccess={(newSurvey) => {
                loadSurvey(newSurvey);
                setShowUploadModal(false);
                setActiveScreen('sonar-analysis');
              }}
            />
          </div>
        </div>
      )}
    </MainLayout>
  );
};
