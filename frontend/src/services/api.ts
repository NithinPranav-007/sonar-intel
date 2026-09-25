import axios from 'axios';
import { Contact, SurveyUploadResponse, SurveySummary, NavWaypoint, ReviewStatus } from '../types/detection';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  async checkHealth() {
    try {
      const response = await client.get('/api/health');
      return response.data;
    } catch (err) {
      console.warn('API health check error:', err);
      return { status: 'offline' };
    }
  },

  async uploadSurvey(sonarFile: File, navFile?: File): Promise<SurveyUploadResponse> {
    const formData = new FormData();
    formData.append('sonar_file', sonarFile);
    if (navFile) {
      formData.append('nav_file', navFile);
    }
    const response = await client.post<SurveyUploadResponse>('/api/surveys/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async analyzeSurvey(surveyId: string, confidenceThreshold = 0.25): Promise<{ contacts: Contact[]; execution_time_ms: number }> {
    const response = await client.post<{ contacts: Contact[]; execution_time_ms: number }>(
      `/api/surveys/${surveyId}/analyze`,
      { confidence_threshold: confidenceThreshold }
    );
    return response.data;
  },

  async getSurveyContacts(surveyId: string): Promise<Contact[]> {
    const response = await client.get<Contact[]>(`/api/surveys/${surveyId}/contacts`);
    return response.data;
  },

  async getContact(contactId: string): Promise<Contact> {
    const response = await client.get<Contact>(`/api/contacts/${contactId}`);
    return response.data;
  },

  async submitReview(contactId: string, reviewStatus: ReviewStatus, reviewNote?: string): Promise<Contact> {
    const response = await client.post<Contact>(`/api/contacts/${contactId}/review`, {
      review_status: reviewStatus,
      review_note: reviewNote || null,
    });
    return response.data;
  },

  async getSurveySummary(surveyId: string): Promise<SurveySummary> {
    const response = await client.get<SurveySummary>(`/api/surveys/${surveyId}/summary`);
    return response.data;
  },

  async getSurveyGeoJSON(surveyId: string): Promise<any> {
    const response = await client.get(`/api/surveys/${surveyId}/geojson`);
    return response.data;
  },

  async getSurveyTrack(surveyId: string): Promise<NavWaypoint[]> {
    try {
      const response = await client.get<NavWaypoint[]>(`/api/surveys/${surveyId}/track`);
      return response.data;
    } catch {
      return [];
    }
  },

  getRawImageUrl(surveyId: string): string {
    return `/api/surveys/${surveyId}/image/raw`;
  },

  getProcessedImageUrl(surveyId: string): string {
    return `/api/surveys/${surveyId}/image/processed`;
  },

  async getDemoSamples(): Promise<any[]> {
    const response = await client.get('/api/demo/samples');
    return response.data;
  },

  async loadDemoSample(sampleId: string): Promise<{ survey: SurveyUploadResponse; contacts: Contact[]; sample_info: any }> {
    const response = await client.post(`/api/demo/load/${sampleId}`);
    return response.data;
  },

  async getDashboardStats(): Promise<any> {
    const response = await client.get('/api/dashboard/stats');
    return response.data;
  },

  async getPipelineInfo(): Promise<any> {
    const response = await client.get('/api/pipeline/info');
    return response.data;
  },
};

