'use client';

import React, { useState, useEffect } from 'react';
import { useEvaluations } from '../../hooks/useEvaluations';
import { usePipelines } from '../../hooks/usePipelines';
import { useSSE } from '../../hooks/useSSE';
import { useAppStore } from '../../store/appStore';
import { ShieldCheck, Search, Sliders, Calendar, ChevronDown, ChevronUp, Clock, HelpCircle } from 'lucide-react';
import { Run } from '../../types';

export default function EvaluationsPage() {
  const { pipelines } = usePipelines();
  
  // Filter States
  const [pipelineId, setPipelineId] = useState('');
  const [threshold, setThreshold] = useState<number>(0.0);
  const [search, setSearch] = useState('');
  
  // Fetch historical evaluations
  const { evaluations: historicalEvals, refetch } = useEvaluations(pipelineId || undefined, threshold || undefined, search || undefined);
  
  const { evaluationFeed, addEvaluation, setEvaluationFeed } = useAppStore();
  const [expandedCardId, setExpandedCardId] = useState<string | null>(null);

  // Sync historical evaluations to store on fetch
  useEffect(() => {
    if (historicalEvals.length > 0) {
      setEvaluationFeed(historicalEvals);
    }
  }, [historicalEvals, setEvaluationFeed]);

  // Connect to live evaluations stream
  useSSE('http://localhost:8000/evaluations/stream', {
    onMessage: (event, data) => {
      if (event === 'evaluation') {
        const evalRun: Run = {
          run_id: data.run_id,
          query: data.query,
          answer: data.answer,
          pipeline_name: data.pipeline_name,
          created_at: new Date().toISOString(),
          evaluation: data.metrics,
        };
        addEvaluation(evalRun);
      }
    },
  });

  const toggleExpand = (id: string) => {
    setExpandedCardId(expandedCardId === id ? null : id);
  };

  // Trigger filter refresh
  const handleFilterChange = () => {
    refetch();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold dark:text-white flex items-center gap-3">
            <ShieldCheck className="w-7 h-7 text-blue-500 animate-pulse" />
            Evaluation Hub
          </h1>
          <p className="text-sm text-neutral-500 dark:text-neutral-400 font-medium">
            Live stream of query faithfulness, relevance, precision, and recall scores.
          </p>
        </div>
      </div>

      {/* Filter Toolbar Panel */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-5 bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl shadow-sm">
        {/* Search */}
        <div className="relative flex items-center">
          <Search className="w-4 h-4 text-neutral-400 absolute left-3.5" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search query or answer..."
            className="w-full bg-neutral-50 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 rounded-xl pl-10 pr-3 py-2 text-sm focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Pipeline Filter */}
        <select
          value={pipelineId}
          onChange={(e) => setPipelineId(e.target.value)}
          className="bg-neutral-50 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 rounded-xl px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 font-semibold"
        >
          <option value="">All Pipelines</option>
          {pipelines.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>

        {/* Threshold slider */}
        <div className="flex flex-col justify-center">
          <div className="flex justify-between text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1">
            <span>Score Threshold</span>
            <span>{(threshold * 100).toFixed(0)}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full accent-blue-600 cursor-pointer"
          />
        </div>

        {/* Apply Filters */}
        <button
          onClick={handleFilterChange}
          className="bg-neutral-100 hover:bg-neutral-200 dark:bg-neutral-800 dark:hover:bg-neutral-750 text-neutral-700 dark:text-neutral-200 rounded-xl px-4 py-2 text-sm font-bold border border-neutral-250 dark:border-neutral-700 flex items-center justify-center gap-2"
        >
          <Sliders className="w-4 h-4" />
          <span>Apply Filters</span>
        </button>
      </div>

      {/* Live Feed List */}
      <div className="space-y-4">
        {evaluationFeed.length === 0 ? (
          <div className="p-8 text-center bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-850 rounded-2xl">
            <span className="text-sm font-bold text-neutral-400">Waiting for live evaluation stream data...</span>
          </div>
        ) : (
          evaluationFeed.map((run) => {
            const isExpanded = expandedCardId === run.run_id;
            const evalScore = run.evaluation?.overall ?? 0.0;

            // Rating badge logic
            let badgeColor = 'text-emerald-600 bg-emerald-50 dark:bg-emerald-950/20';
            if (evalScore < 0.6) {
              badgeColor = 'text-rose-600 bg-rose-50 dark:bg-rose-950/20';
            } else if (evalScore < 0.8) {
              badgeColor = 'text-amber-600 bg-amber-50 dark:bg-amber-950/20';
            }

            return (
              <div
                key={run.run_id}
                className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl shadow-sm overflow-hidden"
              >
                {/* Main Card Content */}
                <div
                  onClick={() => toggleExpand(run.run_id)}
                  className="p-5 flex items-center justify-between gap-4 cursor-pointer hover:bg-neutral-50 dark:hover:bg-neutral-850/30 transition-all duration-200"
                >
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-bold text-neutral-400 bg-neutral-100 dark:bg-neutral-800 px-2.5 py-0.5 rounded">
                        {run.pipeline_name || 'Legal Parser'}
                      </span>
                      <div className="flex items-center gap-1.5 text-xs text-neutral-400">
                        <Clock className="w-3.5 h-3.5" />
                        <span>{run.created_at ? new Date(run.created_at).toLocaleTimeString() : 'Just now'}</span>
                      </div>
                    </div>
                    <h3 className="text-sm font-semibold text-neutral-800 dark:text-neutral-100 line-clamp-1">
                      {run.query}
                    </h3>
                  </div>

                  <div className="flex items-center gap-4">
                    <div className={`px-3 py-1 rounded-xl text-sm font-extrabold ${badgeColor}`}>
                      {Math.round(evalScore * 100)}%
                    </div>
                    {isExpanded ? <ChevronUp className="w-5 h-5 text-neutral-400" /> : <ChevronDown className="w-5 h-5 text-neutral-400" />}
                  </div>
                </div>

                {/* Progress bars inside card footer */}
                <div className="px-5 pb-5 grid grid-cols-2 md:grid-cols-4 gap-4 border-t border-neutral-50 dark:border-neutral-850/40 pt-4">
                  {/* Faithfulness */}
                  <div>
                    <div className="flex justify-between text-[9px] font-bold text-neutral-400 uppercase tracking-wider mb-1">
                      <span>Faithfulness</span>
                      <span>{Math.round((run.evaluation?.faithfulness || 0) * 100)}%</span>
                    </div>
                    <div className="w-full bg-neutral-100 dark:bg-neutral-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-emerald-500 h-full rounded-full"
                        style={{ width: `${(run.evaluation?.faithfulness || 0) * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Relevance */}
                  <div>
                    <div className="flex justify-between text-[9px] font-bold text-neutral-400 uppercase tracking-wider mb-1">
                      <span>Relevance</span>
                      <span>{Math.round((run.evaluation?.answer_relevance || 0) * 100)}%</span>
                    </div>
                    <div className="w-full bg-neutral-100 dark:bg-neutral-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-blue-500 h-full rounded-full"
                        style={{ width: `${(run.evaluation?.answer_relevance || 0) * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Precision */}
                  <div>
                    <div className="flex justify-between text-[9px] font-bold text-neutral-400 uppercase tracking-wider mb-1">
                      <span>Precision</span>
                      <span>{Math.round((run.evaluation?.context_precision || 0) * 100)}%</span>
                    </div>
                    <div className="w-full bg-neutral-100 dark:bg-neutral-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-indigo-500 h-full rounded-full"
                        style={{ width: `${(run.evaluation?.context_precision || 0) * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Recall */}
                  <div>
                    <div className="flex justify-between text-[9px] font-bold text-neutral-400 uppercase tracking-wider mb-1">
                      <span>Recall</span>
                      <span>{Math.round((run.evaluation?.context_recall || 0) * 100)}%</span>
                    </div>
                    <div className="w-full bg-neutral-100 dark:bg-neutral-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-purple-500 h-full rounded-full"
                        style={{ width: `${(run.evaluation?.context_recall || 0) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Expanded Details section */}
                {isExpanded && (
                  <div className="p-6 bg-neutral-50 dark:bg-neutral-850/20 border-t border-neutral-100 dark:border-neutral-850 space-y-4">
                    <div className="space-y-1">
                      <h4 className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider">User Query</h4>
                      <div className="text-sm font-semibold text-neutral-800 dark:text-neutral-200 bg-white dark:bg-neutral-900 border border-neutral-150 dark:border-neutral-800 p-4 rounded-xl">
                        {run.query}
                      </div>
                    </div>

                    <div className="space-y-1">
                      <h4 className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider">Generated Answer</h4>
                      <div className="text-sm text-neutral-700 dark:text-neutral-300 bg-white dark:bg-neutral-900 border border-neutral-150 dark:border-neutral-800 p-4 rounded-xl leading-relaxed whitespace-pre-wrap">
                        {run.answer || 'Answer not populated in log.'}
                      </div>
                    </div>

                    <div className="space-y-1">
                      <h4 className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider">Retrieved Chunks</h4>
                      <div className="text-xs text-neutral-600 dark:text-neutral-400 bg-white dark:bg-neutral-900 border border-neutral-150 dark:border-neutral-800 p-4 rounded-xl space-y-2">
                        <div>• Document: AttentionIsAllYouNeed.pdf (Page 4)</div>
                        <div className="pl-3 border-l-2 border-neutral-200 text-neutral-400 italic">
                          &ldquo;Self-attention, sometimes called intra-attention, is an attention mechanism...&rdquo;
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
