'use client';

import React from 'react';
import { X, TrendingUp, Cpu, BarChart as BarChartIcon, Shield } from 'lucide-react';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  LineChart, 
  Line, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  Radar 
} from 'recharts';
import { PipelineAnalytics, Run } from '../../types';

interface AnalyticsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  analytics: PipelineAnalytics | null;
  failedRuns: Run[];
  isLoading: boolean;
}

export default function AnalyticsDrawer({ isOpen, onClose, analytics, failedRuns, isLoading }: AnalyticsDrawerProps) {
  if (!isOpen) return null;

  // 1. Latency data
  const latencyData = analytics ? [
    { name: 'P50 (Median)', latency: analytics.retrieval_latency.p50 },
    { name: 'P95', latency: analytics.retrieval_latency.p95 },
    { name: 'P99', latency: analytics.retrieval_latency.p99 },
  ] : [];

  // 2. Cost data (calculated as queries_count * cost_per_query)
  const costTrendData = analytics ? analytics.queries_per_day.map(qd => ({
    day: qd.day.substring(5), // Short format MM-DD
    cost: Number((qd.count * analytics.cost_per_query).toFixed(4)),
  })) : [];

  // 3. Radar data
  const radarData = analytics ? [
    { subject: 'Faithfulness', value: analytics.evaluation.faithfulness * 100 },
    { subject: 'Relevance', value: analytics.evaluation.answer_relevance * 100 },
    { subject: 'Precision', value: analytics.evaluation.context_precision * 100 },
    { subject: 'Recall', value: analytics.evaluation.context_recall * 100 },
    { subject: 'Overall', value: analytics.evaluation.overall * 100 },
  ] : [];

  return (
    <>
      <div className="fixed inset-0 bg-neutral-950/45 backdrop-blur-sm z-40" onClick={onClose} />

      <div className="fixed top-0 right-0 h-full w-[450px] max-w-full bg-white dark:bg-neutral-900 border-l border-neutral-200 dark:border-neutral-800 shadow-2xl z-50 flex flex-col transform transition-transform duration-300">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-100 dark:border-neutral-850">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-50 dark:bg-indigo-900/20 rounded-xl text-indigo-500">
              <TrendingUp className="w-5 h-5" />
            </div>
            <div>
              <h2 className="font-bold text-neutral-900 dark:text-white">Pipeline Performance</h2>
              <span className="text-[10px] text-neutral-400 font-semibold uppercase tracking-wider">
                Metrics & Analytics Logs
              </span>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 hover:bg-neutral-100 dark:hover:bg-neutral-800 rounded-lg text-neutral-500">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Drawer Body */}
        {isLoading || !analytics ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            
            {/* Latency Section */}
            <div className="space-y-3">
              <h3 className="text-xs font-bold text-neutral-400 uppercase tracking-wider flex items-center gap-2">
                <BarChartIcon className="w-4 h-4 text-blue-500" />
                Latency Distribution (ms)
              </h3>
              <div className="h-48 w-full bg-neutral-50 dark:bg-neutral-850 p-4 rounded-xl border border-neutral-100 dark:border-neutral-800/40">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={latencyData}>
                    <XAxis dataKey="name" stroke="#888888" fontSize={10} tickLine={false} axisLine={false} />
                    <YAxis stroke="#888888" fontSize={10} tickLine={false} axisLine={false} />
                    <Tooltip cursor={{ fill: 'rgba(0, 0, 0, 0.05)' }} />
                    <Bar dataKey="latency" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Cost Trend Section */}
            <div className="space-y-3">
              <h3 className="text-xs font-bold text-neutral-400 uppercase tracking-wider flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-emerald-500" />
                30-Day Cost Trend (USD)
              </h3>
              <div className="h-48 w-full bg-neutral-50 dark:bg-neutral-850 p-4 rounded-xl border border-neutral-100 dark:border-neutral-800/40">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={costTrendData}>
                    <XAxis dataKey="day" stroke="#888888" fontSize={10} tickLine={false} axisLine={false} />
                    <YAxis stroke="#888888" fontSize={10} tickLine={false} axisLine={false} />
                    <Tooltip />
                    <Line type="monotone" dataKey="cost" stroke="#10B981" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Radar Metrics Section */}
            <div className="space-y-3">
              <h3 className="text-xs font-bold text-neutral-400 uppercase tracking-wider flex items-center gap-2">
                <Shield className="w-4 h-4 text-indigo-500" />
                Evaluation Alignment Radar
              </h3>
              <div className="h-56 w-full bg-neutral-50 dark:bg-neutral-850 p-4 rounded-xl border border-neutral-100 dark:border-neutral-800/40 flex justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData} outerRadius="70%">
                    <PolarGrid stroke="#E5E7EB" />
                    <PolarAngleAxis dataKey="subject" stroke="#888888" fontSize={9} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#888888" fontSize={8} />
                    <Radar name="Metrics" dataKey="value" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.2} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Failed Runs List */}
            <div className="space-y-3">
              <h3 className="text-xs font-bold text-neutral-400 uppercase tracking-wider flex items-center gap-2">
                <Cpu className="w-4.5 h-4.5 text-rose-500" />
                Failed Execution Logs ({failedRuns.length})
              </h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {failedRuns.length === 0 ? (
                  <span className="text-xs font-semibold text-neutral-400 block text-center py-4">
                    No failed pipeline executions found.
                  </span>
                ) : (
                  failedRuns.map((run) => (
                    <div key={run.run_id} className="p-3 bg-rose-50/40 dark:bg-rose-950/15 border border-rose-100 dark:border-rose-900/30 rounded-xl space-y-1">
                      <div className="flex items-center justify-between text-[10px] font-bold text-neutral-400">
                        <span className="font-mono text-rose-700 dark:text-rose-400">ID: {run.run_id.substring(0, 8)}</span>
                        <span>{run.created_at || 'Just now'}</span>
                      </div>
                      <span className="text-xs font-semibold text-rose-800 dark:text-rose-350 block">
                        TimeoutExceededError: API execution exceeded limit.
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>

          </div>
        )}
      </div>
    </>
  );
}
