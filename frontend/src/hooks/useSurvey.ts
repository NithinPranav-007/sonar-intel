import { useState, useCallback } from 'react';
import { Contact, SurveyUploadResponse, SurveySummary, NavWaypoint, ReviewStatus } from '../types/detection';
import { apiService } from '../services/api';

export function useSurvey() {
  const [survey, setSurvey] = useState<SurveyUploadResponse | null>(null);
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [selectedContact, setSelectedContact] = useState<Contact | null>(null);
  const [navTrack, setNavTrack] = useState<NavWaypoint[]>([]);
  const [summary, setSummary] = useState<SurveySummary | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const loadSurvey = useCallback(async (surveyData: SurveyUploadResponse) => {
    setSurvey(surveyData);
    setError(null);
    try {
      const track = await apiService.getSurveyTrack(surveyData.survey_id);
      setNavTrack(track);
      const existingContacts = await apiService.getSurveyContacts(surveyData.survey_id);
      setContacts(existingContacts);
      if (existingContacts.length > 0) {
        setSelectedContact(existingContacts[0]);
        const sum = await apiService.getSurveySummary(surveyData.survey_id);
        setSummary(sum);
      } else {
        setSelectedContact(null);
        setSummary(null);
      }
    } catch (err: any) {
      console.warn('Could not fetch existing survey details:', err);
    }
  }, []);

  const runAnalysis = useCallback(async (confidenceThreshold = 0.25) => {
    if (!survey) return;
    setAnalyzing(true);
    setError(null);
    try {
      const result = await apiService.analyzeSurvey(survey.survey_id, confidenceThreshold);
      setContacts(result.contacts);
      if (result.contacts.length > 0) {
        setSelectedContact(result.contacts[0]);
      }
      const sum = await apiService.getSurveySummary(survey.survey_id);
      setSummary(sum);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Analysis failed. Check server logs.');
    } finally {
      setAnalyzing(false);
    }
  }, [survey]);

  const submitReview = useCallback(async (contactId: string, status: ReviewStatus, note?: string) => {
    try {
      const updated = await apiService.submitReview(contactId, status, note);
      setContacts(prev => prev.map(c => c.contact_id === contactId ? updated : c));
      if (selectedContact?.contact_id === contactId) {
        setSelectedContact(updated);
      }
      if (survey) {
        const sum = await apiService.getSurveySummary(survey.survey_id);
        setSummary(sum);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to submit review.');
    }
  }, [selectedContact, survey]);

  const loadCuratedSample = useCallback(async (sampleId: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.loadDemoSample(sampleId);
      setSurvey(data.survey);
      setContacts(data.contacts);
      if (data.contacts && data.contacts.length > 0) {
        setSelectedContact(data.contacts[0]);
      } else {
        setSelectedContact(null);
      }
      try {
        const sum = await apiService.getSurveySummary(data.survey.survey_id);
        setSummary(sum);
        const track = await apiService.getSurveyTrack(data.survey.survey_id);
        setNavTrack(track);
      } catch (e) {
        // Nav track or summary might be partial
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || `Failed to load demo sample '${sampleId}'.`);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadDemoSurvey = useCallback(async () => {
    return loadCuratedSample('survey_001');
  }, [loadCuratedSample]);

  const uploadSurvey = useCallback(async (sonarFile: File, navFile?: File) => {
    setLoading(true);
    setError(null);
    try {
      const surveyData = await apiService.uploadSurvey(sonarFile, navFile);
      setSurvey(surveyData);
      // Run automatic analysis immediately after upload
      setAnalyzing(true);
      try {
        const result = await apiService.analyzeSurvey(surveyData.survey_id, 0.20);
        setContacts(result.contacts);
        if (result.contacts.length > 0) {
          setSelectedContact(result.contacts[0]);
        }
        const sum = await apiService.getSurveySummary(surveyData.survey_id);
        setSummary(sum);
        const track = await apiService.getSurveyTrack(surveyData.survey_id);
        setNavTrack(track);
      } finally {
        setAnalyzing(false);
      }
      return surveyData;
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload acoustic swath file.');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
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
    loadDemoSurvey,
    loadCuratedSample,
    setError
  };
}
