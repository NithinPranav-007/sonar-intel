import React, { useState, useEffect, useRef } from 'react';
import { MainLayout, ActiveScreen } from '../components/layout/MainLayout';
import { useSurvey } from '../hooks/useSurvey';
import { Contact } from '../types/detection';
import { DashboardPage } from './DashboardPage';
import { SonarAnalysisPage } from './SonarAnalysisPage';
import { ContactVerificationPage } from './ContactVerificationPage';
import { GisMappingPage } from './GisMappingPage';
import { AiPipelinePage } from './AiPipelinePage';
import { ReportsPage } from './ReportsPage';
import { Upload, AlertCircle, X, FileText, CheckCircle2, Play, Navigation } from 'lucide-react';

export const Dashboard: React.FC = () => {
  const [activeScreen, setActiveScreen] = useState<ActiveScreen>('dashboard');
  const [showUploadModal, setShowUploadModal] = useState<boolean>(false);
  const [selectedSonarFile, setSelectedSonarFile] = useState<File | null>(null);
  const [selectedNavFile, setSelectedNavFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const sonarInputRef = useRef<HTMLInputElement>(null);
  const navInputRef = useRef<HTMLInputElement>(null);

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
    uploadSurvey,
    runAnalysis,
    submitReview,
    loadCuratedSample,
    setError
  } = useSurvey();

  // Ingest primary held-out benchmark true-positive (Viator-04) on first load
  useEffect(() => {
    if (!survey && !loading) {
      loadCuratedSample('viator_04');
    }
  }, []);

  const handleExportGeoJSON = () => {
    if (!survey) return;
    window.open(`/api/surveys/${survey.survey_id}/geojson`, '_blank');
  };

  const handleVerifyContact = (contact: Contact) => {
    setSelectedContact(contact);
    setActiveScreen('contact-verification');
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.name.match(/\.(png|jpg|jpeg|tif|tiff)$/i)) {
        setSelectedSonarFile(file);
      } else if (file.name.match(/\.(csv|txt|nav)$/i)) {
        setSelectedNavFile(file);
      }
    }
  };

  const handlePerformUpload = async () => {
    if (!selectedSonarFile) return;
    setIsUploading(true);
    try {
      await uploadSurvey(selectedSonarFile, selectedNavFile || undefined);
      setShowUploadModal(false);
      setSelectedSonarFile(null);
      setSelectedNavFile(null);
      setActiveScreen('sonar-analysis');
    } catch (err) {
      // Error handled by useSurvey hook
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <MainLayout
      activeScreen={activeScreen}
      onSelectScreen={setActiveScreen}
      survey={survey}
      summary={summary}
      contacts={contacts}
      analyzing={analyzing}
      onLoadDemoSample={loadCuratedSample}
      onCustomUploadClick={() => setShowUploadModal(true)}
      onRunAnalysis={() => runAnalysis(0.20)}
    >
      {/* System Error / Alert Toast */}
      {error && (
        <div className="mx-8 mt-4 p-4 rounded-2xl bg-rose-50 border border-rose-200 text-rose-700 flex items-center justify-between shadow-soft">
          <div className="flex items-center gap-2.5 text-xs font-semibold">
            <AlertCircle className="w-4 h-4 text-rose-600" />
            <span>[System Notice] {error}</span>
          </div>
          <button 
            onClick={() => setError(null)} 
            className="p-1 rounded-full hover:bg-rose-100 text-rose-700 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Screen 1: Dashboard Overview (Placely Timespent Theme) */}
      {activeScreen === 'dashboard' && (
        <DashboardPage
          survey={survey}
          contacts={contacts}
          navTrack={navTrack}
          onSelectScreen={setActiveScreen}
          onSelectContact={(c) => {
            setSelectedContact(c);
            setActiveScreen('sonar-analysis');
          }}
          onCustomUploadClick={() => setShowUploadModal(true)}
        />
      )}

      {/* Screen 2: Sonar Analysis Workspace (Placely Timespent Theme) */}
      {activeScreen === 'sonar-analysis' && (
        <SonarAnalysisPage
          survey={survey}
          contacts={contacts}
          selectedContact={selectedContact}
          analyzing={analyzing}
          onSelectContact={setSelectedContact}
          onRunAnalysis={() => runAnalysis(0.20)}
          onVerifyContact={handleVerifyContact}
          onNavigateToDashboard={() => setActiveScreen('dashboard')}
        />
      )}

      {/* Screen 3: Contact Verification Workflow (Placely Timespent Theme) */}
      {activeScreen === 'contact-verification' && (
        <ContactVerificationPage
          survey={survey}
          contacts={contacts}
          selectedContact={selectedContact}
          onSelectContact={setSelectedContact}
          onSubmitReview={submitReview}
          onNavigateToMap={() => setActiveScreen('gis-mapping')}
          onNavigateToAnalysis={() => setActiveScreen('sonar-analysis')}
          onNavigateToDashboard={() => setActiveScreen('dashboard')}
        />
      )}

      {/* Screen 4: GIS Spatial Mapping & Nautical Chart (Placely Timespent Theme) */}
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
          onNavigateToDashboard={() => setActiveScreen('dashboard')}
        />
      )}

      {/* Screen 5: AI Deep Learning Pipeline Monitor (Placely Timespent Theme) */}
      {activeScreen === 'ai-pipeline' && (
        <AiPipelinePage 
          onNavigateToDashboard={() => setActiveScreen('dashboard')}
        />
      )}

      {/* Screen 6: Reports & Export Central (Placely Timespent Theme) */}
      {activeScreen === 'reports' && (
        <ReportsPage
          survey={survey}
          contacts={contacts}
          onNavigateToDashboard={() => setActiveScreen('dashboard')}
          onSelectContact={setSelectedContact}
          onNavigateToVerify={() => setActiveScreen('contact-verification')}
        />
      )}

      {/* Interactive Upload Swath Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-[28px] border border-[#e2e8f0] max-w-xl w-full p-7 space-y-5 shadow-2xl font-sans">
            
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div>
                <h4 className="text-lg font-extrabold text-[#0f172a] font-display">
                  Ingest Side-Scan Sonar Swath
                </h4>
                <p className="text-xs text-[#64748b] mt-0.5">
                  Upload raw SSS imagery and optional towfish navigation log for PostGIS georeferencing.
                </p>
              </div>
              <button 
                onClick={() => setShowUploadModal(false)}
                className="p-2 rounded-full hover:bg-slate-100 text-[#64748b] transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Hidden file inputs */}
            <input 
              ref={sonarInputRef} 
              type="file" 
              accept=".png,.jpg,.jpeg,.tif,.tiff" 
              className="hidden" 
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  setSelectedSonarFile(e.target.files[0]);
                }
              }} 
            />
            <input 
              ref={navInputRef} 
              type="file" 
              accept=".csv,.txt,.nav" 
              className="hidden" 
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  setSelectedNavFile(e.target.files[0]);
                }
              }} 
            />

            {/* Primary Sonar Drag & Drop Zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => sonarInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-6 text-center space-y-2 transition-all cursor-pointer ${
                isDragging 
                  ? 'border-[#1d4ed8] bg-blue-50/60 shadow-blue-sm' 
                  : selectedSonarFile 
                  ? 'border-emerald-400 bg-emerald-50/40' 
                  : 'border-[#e2e8f0] hover:border-blue-400 bg-[#f8fafc]'
              }`}
            >
              {selectedSonarFile ? (
                <div className="flex items-center justify-center gap-3">
                  <CheckCircle2 className="w-8 h-8 text-emerald-500 shrink-0" />
                  <div className="text-left">
                    <div className="text-sm font-bold text-[#0f172a] truncate max-w-xs">{selectedSonarFile.name}</div>
                    <div className="text-xs text-[#64748b]">
                      {(selectedSonarFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for AI Preprocessing
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  <Upload className="w-8 h-8 text-[#1d4ed8] mx-auto" />
                  <div className="text-sm font-bold text-[#0f172a]">Click to select or drag Sonar Waterfall Swath</div>
                  <div className="text-xs text-[#64748b]">Supports standard SSS GeoTIFF, PNG, JPG</div>
                </>
              )}
            </div>

            {/* Optional Navigation Track File Section */}
            <div 
              onClick={() => navInputRef.current?.click()}
              className={`p-4 rounded-xl border border-dashed transition-all cursor-pointer flex items-center justify-between ${
                selectedNavFile 
                  ? 'border-emerald-300 bg-emerald-50/30' 
                  : 'border-[#e2e8f0] bg-[#f8fafc] hover:border-slate-300'
              }`}
            >
              <div className="flex items-center gap-3">
                <Navigation className={`w-5 h-5 ${selectedNavFile ? 'text-emerald-500' : 'text-[#64748b]'}`} />
                <div>
                  <div className="text-xs font-bold text-[#0f172a]">
                    {selectedNavFile ? selectedNavFile.name : 'Attach Towfish Navigation Log (Optional)'}
                  </div>
                  <div className="text-[11px] text-[#64748b]">
                    {selectedNavFile ? `${(selectedNavFile.size / 1024).toFixed(1)} KB • Coordinates linked` : 'CSV containing ping_id, latitude, longitude, heading'}
                  </div>
                </div>
              </div>
              <button 
                type="button"
                className="text-xs font-bold text-[#1d4ed8] hover:underline cursor-pointer"
              >
                {selectedNavFile ? 'Change' : 'Browse'}
              </button>
            </div>

            {/* Actions */}
            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => {
                  setShowUploadModal(false);
                  setSelectedSonarFile(null);
                  setSelectedNavFile(null);
                }}
                disabled={isUploading}
                className="px-5 py-2.5 rounded-full border border-[#e2e8f0] text-[#0f172a] text-xs font-semibold hover:bg-slate-50 transition-colors cursor-pointer"
              >
                Cancel
              </button>

              <button
                type="button"
                onClick={handlePerformUpload}
                disabled={!selectedSonarFile || isUploading}
                className={`px-6 py-2.5 rounded-full text-xs font-bold flex items-center gap-2 transition-all shadow-tactile ${
                  !selectedSonarFile || isUploading
                    ? 'bg-slate-200 text-[#64748b] cursor-not-allowed'
                    : 'bg-[#1d4ed8] hover:bg-[#1e40af] text-white hover:scale-[1.02] active:scale-[0.98] cursor-pointer shadow-blue-glow'
                }`}
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>{isUploading ? 'Ingesting & Analyzing...' : 'Ingest & Run Pipeline'}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </MainLayout>
  );
};
