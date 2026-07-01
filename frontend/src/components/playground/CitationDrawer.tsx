'use client';

import React from 'react';
import { X, FileText, Database, Layers, Tag } from 'lucide-react';

interface CitationSource {
  document_name: string;
  page?: number | string;
  chunk_id: string;
  text: string;
  metadata?: Record<string, any>;
}

interface CitationDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  source: CitationSource | null;
}

export default function CitationDrawer({ isOpen, onClose, source }: CitationDrawerProps) {
  if (!source) return null;

  return (
    <>
      {/* Background Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-neutral-900/40 backdrop-blur-sm z-40 transition-opacity duration-300"
          onClick={onClose}
        />
      )}

      {/* Slide-out Drawer Panel */}
      <div
        className={`fixed top-0 right-0 h-full w-96 max-w-full bg-white dark:bg-neutral-900 border-l border-neutral-200 dark:border-neutral-800 shadow-2xl z-50 transform transition-transform duration-350 ease-out flex flex-col ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        {/* Drawer Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-100 dark:border-neutral-800/80">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-50 dark:bg-blue-900/30 rounded-xl">
              <FileText className="w-5 h-5 text-blue-500" />
            </div>
            <div>
              <h2 className="font-bold text-neutral-900 dark:text-white truncate max-w-[200px]">
                {source.document_name}
              </h2>
              <span className="text-[10px] text-neutral-400 font-semibold uppercase tracking-wider">
                Source Document
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-neutral-100 dark:hover:bg-neutral-805 rounded-lg text-neutral-500 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Drawer Body (Scrollable) */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Metadata Grid */}
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 bg-neutral-50 dark:bg-neutral-850 rounded-xl border border-neutral-100 dark:border-neutral-800/40">
              <div className="flex items-center gap-2 text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1">
                <Database className="w-3.5 h-3.5" />
                Page
              </div>
              <span className="font-bold text-neutral-800 dark:text-neutral-200">
                {source.page || 'N/A'}
              </span>
            </div>
            <div className="p-3 bg-neutral-50 dark:bg-neutral-850 rounded-xl border border-neutral-100 dark:border-neutral-800/40">
              <div className="flex items-center gap-2 text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1">
                <Layers className="w-3.5 h-3.5" />
                Chunk ID
              </div>
              <span className="font-semibold text-neutral-800 dark:text-neutral-200 text-xs truncate block max-w-[120px]" title={source.chunk_id}>
                {source.chunk_id || 'N/A'}
              </span>
            </div>
          </div>

          {/* Chunk Content */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold text-neutral-400 uppercase tracking-wider flex items-center gap-2">
              <FileText className="w-4 h-4" />
              Retrieved Text Segment
            </h3>
            <div className="p-4 bg-neutral-50 dark:bg-neutral-850 rounded-xl border border-neutral-100 dark:border-neutral-800/50 text-sm leading-relaxed text-neutral-700 dark:text-neutral-300 font-medium">
              {source.text}
            </div>
          </div>

          {/* Additional Metadata Attributes */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold text-neutral-400 uppercase tracking-wider flex items-center gap-2">
              <Tag className="w-4 h-4" />
              Metadata Attributes
            </h3>
            <div className="p-4 bg-neutral-50 dark:bg-neutral-850 rounded-xl border border-neutral-100 dark:border-neutral-800/50 overflow-x-auto">
              <pre className="text-xs text-neutral-600 dark:text-neutral-400 font-mono">
                {JSON.stringify(source.metadata || { "similarity_score": 0.92, "source_type": "pdf" }, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
