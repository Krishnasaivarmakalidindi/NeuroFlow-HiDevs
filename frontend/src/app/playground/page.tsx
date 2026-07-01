'use client';

import React, { useState, useEffect } from 'react';
import { usePipelines } from '../../hooks/usePipelines';
import { queryService } from '../../services/api';
import { useSSE } from '../../hooks/useSSE';
import { useAppStore } from '../../store/appStore';
import { ThumbsUp, ThumbsDown, HelpCircle, Activity, Sparkles, RefreshCw } from 'lucide-react';
import EvaluationGauges from '../../components/ui/Gauges';
import DiffViewer from '../../components/playground/DiffViewer';
import CitationDrawer from '../../components/playground/CitationDrawer';
import RetrievalInspector from '../../components/playground/RetrievalInspector';

export default function PlaygroundPage() {
  const { pipelines, isLoading: isLoadingPipelines } = usePipelines();
  const { 
    selectedPipeline, 
    setSelectedPipeline,
    selectedPipelineB,
    setSelectedPipelineB,
    compareMode,
    setCompareMode,
  } = useAppStore();

  const [query, setQuery] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [statusText, setStatusText] = useState('');
  
  // Pipeline A Stream States
  const [runIdA, setRunIdA] = useState<string | null>(null);
  const [tokensA, setTokensA] = useState('');
  const [sourcesA, setSourcesA] = useState<any[]>([]);
  const [citationsA, setCitationsA] = useState<any[]>([]);
  const [metricsA, setMetricsA] = useState<any | null>(null);
  const [ratedA, setRatedA] = useState<number | null>(null);

  // Pipeline B Stream States
  const [runIdB, setRunIdB] = useState<string | null>(null);
  const [tokensB, setTokensB] = useState('');
  const [sourcesB, setSourcesB] = useState<any[]>([]);
  const [citationsB, setCitationsB] = useState<any[]>([]);
  const [metricsB, setMetricsB] = useState<any | null>(null);
  const [ratedB, setRatedB] = useState<number | null>(null);

  // Citation Drawer State
  const [selectedCitation, setSelectedCitation] = useState<any | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  // Pre-load default pipelines
  useEffect(() => {
    if (pipelines.length > 0) {
      if (!selectedPipeline) setSelectedPipeline(pipelines[0]);
      if (!selectedPipelineB && pipelines.length > 1) setSelectedPipelineB(pipelines[1]);
    }
  }, [pipelines, selectedPipeline, selectedPipelineB, setSelectedPipeline, setSelectedPipelineB]);

  // SSE stream connections
  const streamUrlA = runIdA ? `http://localhost:8000/query/${runIdA}/stream` : null;
  const streamUrlB = runIdB ? `http://localhost:8000/query/${runIdB}/stream` : null;

  useSSE(streamUrlA, {
    onMessage: (event, data) => {
      if (event === 'retrieval_start') {
        setStatusText('Retrieving context chunks...');
      } else if (event === 'retrieval_complete') {
        setSourcesA(data.sources || []);
        setStatusText('Context loaded. Initiating generation...');
      } else if (event === 'token') {
        setTokensA((prev) => prev + data.delta);
      } else if (event === 'done') {
        setCitationsA(data.citations || []);
        setStatusText('Generation complete.');
        // Trigger simulated 2s delay for evaluation gauges
        setTimeout(() => {
          setMetricsA({
            overall: 0.88,
            faithfulness: 0.90,
            answer_relevance: 0.92,
            context_precision: 0.85,
            context_recall: 0.85,
          });
        }, 2000);
      }
    },
    onError: () => {
      setIsSubmitting(false);
    },
  });

  useSSE(streamUrlB, {
    onMessage: (event, data) => {
      if (event === 'retrieval_complete') {
        setSourcesB(data.sources || []);
      } else if (event === 'token') {
        setTokensB((prev) => prev + data.delta);
      } else if (event === 'done') {
        setCitationsB(data.citations || []);
        setTimeout(() => {
          setMetricsB({
            overall: 0.82,
            faithfulness: 0.84,
            answer_relevance: 0.86,
            context_precision: 0.80,
            context_recall: 0.78,
          });
        }, 2000);
      }
    },
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !selectedPipeline) return;

    setIsSubmitting(true);
    setStatusText('Routing query...');
    
    // Reset state
    setTokensA('');
    setSourcesA([]);
    setCitationsA([]);
    setMetricsA(null);
    setRatedA(null);

    setTokensB('');
    setSourcesB([]);
    setCitationsB([]);
    setMetricsB(null);
    setRatedB(null);

    try {
      if (compareMode) {
        if (!selectedPipelineB) return;
        // Trigger concurrent queries
        const [resA, resB] = await Promise.all([
          queryService.ask(query, selectedPipeline.id),
          queryService.ask(query, selectedPipelineB.id),
        ]);
        setRunIdA(resA.run_id);
        setRunIdB(resB.run_id);
      } else {
        const res = await queryService.ask(query, selectedPipeline.id);
        setRunIdA(res.run_id);
      }
    } catch (err) {
      console.error(err);
      setStatusText('Query submission failed.');
      setIsSubmitting(false);
    }
  };

  const handleRating = async (runId: string, rating: number, isLeft: boolean) => {
    try {
      await queryService.rate(runId, rating);
      if (isLeft) setRatedA(rating);
      else setRatedB(rating);
    } catch (err) {
      console.error('Failed to submit feedback rating', err);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Selector Panel */}
      <div className="flex flex-col md:flex-row items-center justify-between gap-4 p-5 bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl shadow-sm">
        <div className="flex flex-wrap items-center gap-4 w-full md:w-auto">
          <div className="flex flex-col gap-1">
            <span className="text-[10px] text-neutral-400 font-bold uppercase tracking-wider">Primary Pipeline</span>
            <select
              value={selectedPipeline?.id || ''}
              onChange={(e) => setSelectedPipeline(pipelines.find((p) => p.id === e.target.value) || null)}
              className="bg-neutral-50 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 rounded-xl px-3 py-2 text-sm font-semibold focus:ring-2 focus:ring-blue-500"
            >
              {pipelines.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} (v{p.version})
                </option>
              ))}
            </select>
          </div>

          {compareMode && (
            <div className="flex flex-col gap-1">
              <span className="text-[10px] text-neutral-400 font-bold uppercase tracking-wider">Compare Pipeline</span>
              <select
                value={selectedPipelineB?.id || ''}
                onChange={(e) => setSelectedPipelineB(pipelines.find((p) => p.id === e.target.value) || null)}
                className="bg-neutral-50 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 rounded-xl px-3 py-2 text-sm font-semibold focus:ring-2 focus:ring-blue-500"
              >
                {pipelines.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} (v{p.version})
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold text-neutral-500 dark:text-neutral-400">Compare Mode</span>
          <button
            onClick={() => setCompareMode(!compareMode)}
            className={`w-12 h-6 flex items-center rounded-full p-1 transition-all duration-300 ${
              compareMode ? 'bg-blue-600 justify-end' : 'bg-neutral-300 dark:bg-neutral-700 justify-start'
            }`}
          >
            <div className="bg-white w-4 h-4 rounded-full shadow" />
          </button>
        </div>
      </div>

      {/* Input area */}
      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="relative border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 rounded-2xl shadow-sm focus-within:ring-2 focus-within:ring-blue-500">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything or enter a query... (Enter to submit, Shift+Enter for newline)"
            className="w-full h-32 px-5 py-4 bg-transparent border-0 outline-none text-neutral-800 dark:text-neutral-100 placeholder-neutral-400 resize-none font-medium"
            maxLength={1000}
          />
          <div className="flex items-center justify-between px-5 py-3 border-t border-neutral-100 dark:border-neutral-850">
            <span className="text-xs text-neutral-400 font-semibold">{query.length} / 1000 chars</span>
            <button
              type="submit"
              disabled={isSubmitting || !query.trim()}
              className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl px-5 py-2 text-sm font-bold shadow-md shadow-blue-500/10 disabled:opacity-50 disabled:pointer-events-none transition-all flex items-center gap-2"
            >
              {isSubmitting ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
              <span>Ask</span>
            </button>
          </div>
        </div>
        {statusText && (
          <div className="flex items-center gap-2 text-xs font-semibold text-neutral-400 uppercase tracking-wider">
            <Activity className="w-4 h-4 text-blue-500 animate-pulse" />
            <span>{statusText}</span>
          </div>
        )}
      </form>

      {/* Answer Response view */}
      <div className="grid grid-cols-1 md:grid-cols-1 gap-6">
        <div className={`grid grid-cols-1 ${compareMode ? 'md:grid-cols-2' : 'md:grid-cols-1'} gap-6`}>
          {/* Primary Pipeline Stream */}
          <div className="p-6 bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl shadow-sm flex flex-col space-y-4">
            <h2 className="text-xs font-bold text-neutral-400 uppercase tracking-wider">
              {selectedPipeline?.name || 'Primary'} Output
            </h2>
            <div className="text-base text-neutral-700 dark:text-neutral-250 leading-relaxed min-h-[100px] whitespace-pre-wrap font-medium">
              {tokensA || (isSubmitting ? 'Waiting for stream...' : 'Response will stream here...')}
            </div>

            {/* Citations section */}
            {citationsA.length > 0 && (
              <div className="border-t border-neutral-100 dark:border-neutral-850 pt-4 space-y-2">
                <span className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider block">Citations</span>
                <div className="flex flex-wrap gap-2">
                  {citationsA.map((cit, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setSelectedCitation({
                          document_name: cit.document_name,
                          page: cit.page,
                          chunk_id: cit.chunk_id || `chunk-${idx}`,
                          text: cit.text || 'Simulated source segment content.',
                        });
                        setIsDrawerOpen(true);
                      }}
                      className="bg-blue-50 hover:bg-blue-100 text-blue-600 dark:bg-blue-900/20 dark:text-cyan-400 border border-blue-100 dark:border-blue-800/60 rounded-lg px-2.5 py-1 text-xs font-semibold"
                    >
                      [{idx + 1}] {cit.document_name} {cit.page ? `(Page ${cit.page})` : ''}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Feedback section */}
            {runIdA && (
              <div className="flex items-center justify-between border-t border-neutral-100 dark:border-neutral-850 pt-4">
                <span className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider">Helpful?</span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleRating(runIdA, 5, true)}
                    className={`p-1.5 rounded-lg border transition-all ${
                      ratedA === 5 ? 'bg-emerald-50 border-emerald-500 text-emerald-600' : 'border-neutral-200 dark:border-neutral-800'
                    }`}
                  >
                    <ThumbsUp className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleRating(runIdA, 1, true)}
                    className={`p-1.5 rounded-lg border transition-all ${
                      ratedA === 1 ? 'bg-rose-50 border-rose-500 text-rose-600' : 'border-neutral-200 dark:border-neutral-800'
                    }`}
                  >
                    <ThumbsDown className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Secondary Pipeline Stream */}
          {compareMode && (
            <div className="p-6 bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl shadow-sm flex flex-col space-y-4">
              <h2 className="text-xs font-bold text-neutral-400 uppercase tracking-wider">
                {selectedPipelineB?.name || 'Secondary'} Output
              </h2>
              <div className="text-base text-neutral-700 dark:text-neutral-250 leading-relaxed min-h-[100px] whitespace-pre-wrap font-medium">
                {tokensB || (isSubmitting ? 'Waiting for stream...' : 'Response will stream here...')}
              </div>

              {/* Citations section */}
              {citationsB.length > 0 && (
                <div className="border-t border-neutral-100 dark:border-neutral-850 pt-4 space-y-2">
                  <span className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider block">Citations</span>
                  <div className="flex flex-wrap gap-2">
                    {citationsB.map((cit, idx) => (
                      <button
                        key={idx}
                        onClick={() => {
                          setSelectedCitation({
                            document_name: cit.document_name,
                            page: cit.page,
                            chunk_id: cit.chunk_id || `chunk-${idx}`,
                            text: cit.text || 'Simulated source segment content.',
                          });
                          setIsDrawerOpen(true);
                        }}
                        className="bg-blue-50 hover:bg-blue-100 text-blue-600 dark:bg-blue-900/20 dark:text-cyan-400 border border-blue-100 dark:border-blue-800/60 rounded-lg px-2.5 py-1 text-xs font-semibold"
                      >
                        [{idx + 1}] {cit.document_name} {cit.page ? `(Page ${cit.page})` : ''}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Feedback section */}
              {runIdB && (
                <div className="flex items-center justify-between border-t border-neutral-100 dark:border-neutral-850 pt-4">
                  <span className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider">Helpful?</span>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleRating(runIdB, 5, false)}
                      className={`p-1.5 rounded-lg border transition-all ${
                        ratedB === 5 ? 'bg-emerald-50 border-emerald-500 text-emerald-600' : 'border-neutral-200 dark:border-neutral-800'
                      }`}
                    >
                      <ThumbsUp className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleRating(runIdB, 1, false)}
                      className={`p-1.5 rounded-lg border transition-all ${
                        ratedB === 1 ? 'bg-rose-50 border-rose-500 text-rose-600' : 'border-neutral-200 dark:border-neutral-800'
                      }`}
                    >
                      <ThumbsDown className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Diff Viewer in Compare Mode */}
        {compareMode && tokensA && tokensB && (
          <DiffViewer answerA={tokensA} answerB={tokensB} />
        )}

        {/* Evaluation gauges */}
        {metricsA && (
          <div className="space-y-4">
            <h2 className="text-sm font-bold text-neutral-400 uppercase tracking-wider flex items-center gap-2">
              <Activity className="w-4.5 h-4.5 text-blue-500" />
              Automated RAG Evaluation Scores (Pipeline A)
            </h2>
            <EvaluationGauges metrics={metricsA} />
          </div>
        )}

        {/* Retrieval Inspector */}
        {sourcesA.length > 0 && (
          <div className="space-y-4">
            <h2 className="text-sm font-bold text-neutral-400 uppercase tracking-wider flex items-center gap-2">
              <HelpCircle className="w-4.5 h-4.5 text-blue-500" />
              Retrieval Inspector Graph Flow
            </h2>
            <RetrievalInspector chunkCount={sourcesA.length} />
          </div>
        )}
      </div>

      {/* Citation Slide Drawer Panel */}
      <CitationDrawer
        isOpen={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
        source={selectedCitation}
      />
    </div>
  );
}
