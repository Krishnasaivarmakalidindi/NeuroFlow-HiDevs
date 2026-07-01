'use client';

import React from 'react';
import { ReactFlow, Background, Controls } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

interface RetrievalInspectorProps {
  chunkCount: number;
}

export default function RetrievalInspector({ chunkCount }: RetrievalInspectorProps) {
  // Define hierarchical nodes styling with glassmorphic cards
  const nodes = [
    {
      id: 'n-query',
      type: 'input',
      data: { 
        label: (
          <div className="text-center">
            <div className="text-[10px] font-bold uppercase tracking-wider text-neutral-400">Input</div>
            <div className="font-semibold text-neutral-800 dark:text-neutral-200">Query</div>
          </div>
        )
      },
      position: { x: 250, y: 10 },
      style: { background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)', borderRadius: '12px', padding: '10px' },
    },
    {
      id: 'n-dense',
      data: { 
        label: (
          <div className="text-left">
            <div className="text-[9px] font-bold uppercase tracking-wider text-emerald-500">Dense Retrieval</div>
            <div className="text-xs text-neutral-600 dark:text-neutral-350">k = 20, score ~0.84</div>
            <div className="text-[9px] text-neutral-400">Latency: 14ms</div>
          </div>
        )
      },
      position: { x: 50, y: 110 },
      style: { background: 'rgba(16, 185, 129, 0.05)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '12px', padding: '10px', width: 160 },
    },
    {
      id: 'n-sparse',
      data: { 
        label: (
          <div className="text-left">
            <div className="text-[9px] font-bold uppercase tracking-wider text-purple-500">Sparse Retrieval</div>
            <div className="text-xs text-neutral-600 dark:text-neutral-350">k = 20, BM25</div>
            <div className="text-[9px] text-neutral-400">Latency: 8ms</div>
          </div>
        )
      },
      position: { x: 250, y: 110 },
      style: { background: 'rgba(139, 92, 246, 0.05)', border: '1px solid rgba(139, 92, 246, 0.2)', borderRadius: '12px', padding: '10px', width: 160 },
    },
    {
      id: 'n-meta',
      data: { 
        label: (
          <div className="text-left">
            <div className="text-[9px] font-bold uppercase tracking-wider text-amber-500">Metadata Filters</div>
            <div className="text-xs text-neutral-600 dark:text-neutral-350">Enabled: Yes</div>
            <div className="text-[9px] text-neutral-400">Latency: 4ms</div>
          </div>
        )
      },
      position: { x: 450, y: 110 },
      style: { background: 'rgba(245, 158, 11, 0.05)', border: '1px solid rgba(245, 158, 11, 0.2)', borderRadius: '12px', padding: '10px', width: 160 },
    },
    {
      id: 'n-rrf',
      data: { 
        label: (
          <div className="text-center">
            <div className="text-[10px] font-bold uppercase tracking-wider text-neutral-400">Merge</div>
            <div className="font-semibold text-neutral-800 dark:text-neutral-200">RRF Rank Fusion</div>
          </div>
        )
      },
      position: { x: 250, y: 220 },
      style: { background: 'rgba(107, 114, 128, 0.1)', border: '1px solid rgba(107, 114, 128, 0.3)', borderRadius: '12px', padding: '10px' },
    },
    {
      id: 'n-rerank',
      data: { 
        label: (
          <div className="text-left">
            <div className="text-[9px] font-bold uppercase tracking-wider text-cyan-500">Cross-Encoder</div>
            <div className="text-xs text-neutral-800 dark:text-neutral-200 font-semibold">Cohere Rerank v3</div>
            <div className="text-[9px] text-neutral-400">Latency: 42ms</div>
          </div>
        )
      },
      position: { x: 250, y: 310 },
      style: { background: 'rgba(6, 182, 212, 0.05)', border: '1px solid rgba(6, 182, 212, 0.2)', borderRadius: '12px', padding: '10px', width: 160 },
    },
    {
      id: 'n-context',
      type: 'output',
      data: { 
        label: (
          <div className="text-center">
            <div className="text-[10px] font-bold uppercase tracking-wider text-emerald-500">Output Context</div>
            <div className="font-bold text-neutral-800 dark:text-neutral-200">{chunkCount} Chunks Selected</div>
          </div>
        )
      },
      position: { x: 250, y: 410 },
      style: { background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: '12px', padding: '10px' },
    },
  ];

  // Connect flow paths
  const edges = [
    { id: 'e-query-dense', source: 'n-query', target: 'n-dense', animated: true },
    { id: 'e-query-sparse', source: 'n-query', target: 'n-sparse', animated: true },
    { id: 'e-query-meta', source: 'n-query', target: 'n-meta', animated: true },
    { id: 'e-dense-rrf', source: 'n-dense', target: 'n-rrf' },
    { id: 'e-sparse-rrf', source: 'n-sparse', target: 'n-rrf' },
    { id: 'e-meta-rrf', source: 'n-meta', target: 'n-rrf' },
    { id: 'e-rrf-rerank', source: 'n-rrf', target: 'n-rerank', animated: true },
    { id: 'e-rerank-context', source: 'n-rerank', target: 'n-context', animated: true },
  ];

  return (
    <div className="h-[500px] border border-neutral-200 dark:border-neutral-800 rounded-2xl overflow-hidden bg-neutral-50 dark:bg-neutral-900/50 shadow-inner">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        nodesConnectable={false}
        nodesDraggable={false}
        zoomOnScroll={false}
        panOnDrag={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
      >
        <Background gap={12} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
