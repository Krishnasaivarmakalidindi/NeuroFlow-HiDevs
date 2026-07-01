import { create } from 'zustand';
import { Pipeline, Run, Document } from '../types';

interface AppState {
  selectedPipeline: Pipeline | null;
  selectedPipelineB: Pipeline | null;
  compareMode: boolean;
  activeRun: Run | null;
  evaluationFeed: Run[];
  documents: Document[];
  theme: 'dark' | 'light' | 'system';
  
  setSelectedPipeline: (pipeline: Pipeline | null) => void;
  setSelectedPipelineB: (pipeline: Pipeline | null) => void;
  setCompareMode: (compareMode: boolean) => void;
  setActiveRun: (run: Run | null) => void;
  setEvaluationFeed: (feed: Run[]) => void;
  addEvaluation: (evaluation: Run) => void;
  setDocuments: (docs: Document[]) => void;
  addDocument: (doc: Document) => void;
  updateDocumentStatus: (id: string, status: Document['status']) => void;
  setTheme: (theme: 'dark' | 'light' | 'system') => void;
  toggleTheme: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  selectedPipeline: null,
  selectedPipelineB: null,
  compareMode: false,
  activeRun: null,
  evaluationFeed: [],
  documents: [],
  theme: 'dark',

  setSelectedPipeline: (pipeline) => set({ selectedPipeline: pipeline }),
  setSelectedPipelineB: (pipeline) => set({ selectedPipelineB: pipeline }),
  setCompareMode: (compareMode) => set({ compareMode }),
  setActiveRun: (run) => set({ activeRun: run }),
  setEvaluationFeed: (feed) => set({ evaluationFeed: feed }),
  addEvaluation: (evaluation) => 
    set((state) => {
      // Keep feed size limited to prevent bloating
      const newFeed = [evaluation, ...state.evaluationFeed];
      return { evaluationFeed: newFeed.slice(0, 100) };
    }),
  setDocuments: (docs) => set({ documents: docs }),
  addDocument: (doc) => set((state) => ({ documents: [doc, ...state.documents] })),
  updateDocumentStatus: (id, status) =>
    set((state) => ({
      documents: state.documents.map((doc) =>
        doc.id === id ? { ...doc, status } : doc
      ),
    })),
  setTheme: (theme) => set({ theme }),
  toggleTheme: () =>
    set((state) => {
      const nextTheme = state.theme === 'dark' ? 'light' : 'dark';
      if (typeof window !== 'undefined') {
        const root = window.document.documentElement;
        root.classList.remove('light', 'dark');
        root.classList.add(nextTheme);
      }
      return { theme: nextTheme };
    }),
}));
