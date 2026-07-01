import axios from 'axios';
import { Pipeline, PipelineConfig, PipelineAnalytics, Run, Document, SimilarChunk } from '../types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const pipelineService = {
  list: async (): Promise<Pipeline[]> => {
    const res = await api.get<Pipeline[]>('/pipelines');
    return res.data;
  },
  get: async (id: string): Promise<Pipeline> => {
    const res = await api.get<Pipeline>(`/pipelines/${id}`);
    return res.data;
  },
  create: async (config: PipelineConfig): Promise<Pipeline> => {
    const res = await api.post<Pipeline>('/pipelines', config);
    return res.data;
  },
  update: async (id: string, config: PipelineConfig): Promise<Pipeline> => {
    const res = await api.patch<Pipeline>(`/pipelines/${id}`, config);
    return res.data;
  },
  delete: async (id: string): Promise<{ status: string; message: string }> => {
    const res = await api.delete<{ status: string; message: string }>(`/pipelines/${id}`);
    return res.data;
  },
  analytics: async (id: string): Promise<PipelineAnalytics> => {
    const res = await api.get<PipelineAnalytics>(`/pipelines/${id}/analytics`);
    return res.data;
  },
  runs: async (id: string, page = 1, limit = 10): Promise<Run[]> => {
    const res = await api.get<Run[]>(`/pipelines/${id}/runs`, {
      params: { page, limit },
    });
    return res.data;
  },
};

export const queryService = {
  ask: async (query: string, pipelineId: string, stream = true): Promise<{ run_id: string; answer?: string; citations?: any[] }> => {
    const res = await api.post<{ run_id: string; answer?: string; citations?: any[] }>('/query', {
      query,
      pipeline_id: pipelineId,
      stream,
    });
    return res.data;
  },
  rate: async (runId: string, rating: number): Promise<any> => {
    const res = await api.patch(`/runs/${runId}/rating`, { rating });
    return res.data;
  },
  compare: async (query: string, pipelineAId: string, pipelineBId: string): Promise<any> => {
    const res = await api.post('/pipelines/compare', {
      query,
      pipeline_a_id: pipelineAId,
      pipeline_b_id: pipelineBId,
    });
    return res.data;
  },
};

export const documentService = {
  list: async (): Promise<Document[]> => {
    const res = await api.get<Document[]>('/documents');
    return res.data;
  },
  get: async (id: string): Promise<Document> => {
    const res = await api.get<Document>(`/documents/${id}`);
    return res.data;
  },
  upload: async (file: File, onUploadProgress?: (progressEvent: any) => void): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post('/ingest', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress,
    });
    return res.data;
  },
  similar: async (id: string): Promise<SimilarChunk[]> => {
    const res = await api.get<SimilarChunk[]>(`/documents/${id}/similar`);
    return res.data;
  },
};

export const evaluationService = {
  list: async (pipelineId?: string, threshold?: number, search?: string): Promise<Run[]> => {
    const res = await api.get<Run[]>('/evaluations', {
      params: { pipeline_id: pipelineId, threshold, search },
    });
    return res.data;
  },
};
