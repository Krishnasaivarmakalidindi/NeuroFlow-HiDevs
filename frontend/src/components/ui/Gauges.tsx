'use client';

import React from 'react';
import { ResponsiveContainer, RadialBarChart, RadialBar, PolarAngleAxis } from 'recharts';
import { EvaluationMetrics } from '../../types';

interface GaugeProps {
  value: number;
  label: string;
}

export function Gauge({ value, label }: GaugeProps) {
  // Convert value to percentage (0 - 100)
  const percent = Math.round(value * 100);

  // Determine color based on threshold rules
  let color = '#EF4444'; // Red (<0.6)
  if (value >= 0.8) {
    color = '#10B981'; // Green (>0.8)
  } else if (value >= 0.6) {
    color = '#F59E0B'; // Yellow (0.6-0.8)
  }

  // Data formatted for Recharts RadialBarChart
  const data = [
    {
      name: label,
      value: percent,
      fill: color,
    },
  ];

  return (
    <div className="flex flex-col items-center justify-center p-4 bg-white dark:bg-neutral-900 border border-neutral-100 dark:border-neutral-850 rounded-2xl shadow-sm transition-all duration-300 hover:shadow-md">
      <div className="w-24 h-24 relative flex items-center justify-center">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            innerRadius="75%"
            outerRadius="100%"
            data={data}
            startAngle={225}
            endAngle={-45}
            barSize={8}
          >
            <PolarAngleAxis
              type="number"
              domain={[0, 100]}
              angleAxisId={0}
              tick={false}
            />
            <RadialBar
              background={{ fill: '#E5E7EB' }} // Light gray background ring
              dataKey="value"
              cornerRadius={5}
            />
          </RadialBarChart>
        </ResponsiveContainer>
        
        {/* Absolute Centered Value */}
        <div className="absolute flex flex-col items-center justify-center">
          <span className="text-xl font-bold dark:text-white" style={{ color }}>
            {percent}%
          </span>
        </div>
      </div>
      <span className="text-xs font-semibold text-neutral-500 dark:text-neutral-400 mt-2 text-center">
        {label}
      </span>
    </div>
  );
}

interface EvaluationGaugesProps {
  metrics: EvaluationMetrics;
}

export default function EvaluationGauges({ metrics }: EvaluationGaugesProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4 w-full">
      <Gauge value={metrics.overall} label="Overall Score" />
      <Gauge value={metrics.faithfulness} label="Faithfulness" />
      <Gauge value={metrics.answer_relevance} label="Answer Relevance" />
      <Gauge value={metrics.context_precision} label="Context Precision" />
      <Gauge value={gaugeSafeRecall(metrics.context_recall)} label="Context Recall" />
    </div>
  );
}

// Safety fallback for metric mismatch if backend returns null
function gaugeSafeRecall(val: any) {
  return typeof val === 'number' ? val : 0;
}
