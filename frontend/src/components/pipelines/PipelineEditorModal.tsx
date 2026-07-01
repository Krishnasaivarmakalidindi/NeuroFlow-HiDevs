'use client';

import React, { useState } from 'react';
import Editor from '@monaco-editor/react';
import { X, CheckCircle, AlertTriangle, FileJson } from 'lucide-react';
import { PipelineConfig } from '../../types';

interface PipelineEditorModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (config: PipelineConfig) => Promise<void>;
  initialConfig?: PipelineConfig | null;
}

const DEFAULT_CONFIG_TEMPLATE = {
  name: "new-pipeline-v1",
  description: "Production legal contract parsing configuration.",
  ingestion: {
    chunking_strategy: "recursive",
    chunk_size_tokens: 512,
    chunk_overlap_tokens: 64,
    extractors_enabled: ["title", "summary"]
  },
  retrieval: {
    dense_k: 20,
    sparse_k: 10,
    reranker: "cohere-rerank-v3",
    top_k_after_rerank: 5,
    query_expansion: true,
    metadata_filters_enabled: false
  },
  generation: {
    model_routing: {
      default: "gpt-4o-mini",
      task_type: "default"
    },
    max_context_tokens: 4096,
    temperature: 0.2,
    system_prompt_variant: "detailed"
  },
  evaluation: {
    auto_evaluate: true,
    training_threshold: 0.82
  }
};

export default function PipelineEditorModal({ isOpen, onClose, onSave, initialConfig }: PipelineEditorModalProps) {
  const [jsonCode, setJsonCode] = useState(
    JSON.stringify(initialConfig || DEFAULT_CONFIG_TEMPLATE, null, 2)
  );
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!isOpen) return null;

  const handleEditorChange = (value: string | undefined) => {
    const code = value || '';
    setJsonCode(code);

    // Dynamic schema validation
    try {
      const data = JSON.parse(code);
      const required = ['name', 'description', 'ingestion', 'retrieval', 'generation', 'evaluation'];
      
      for (const field of required) {
        if (!(field in data)) {
          setValidationError(`Schema error: Missing root attribute "${field}"`);
          return;
        }
      }
      
      // Basic typing checks
      if (typeof data.name !== 'string') throw new Error('"name" must be a string');
      if (typeof data.description !== 'string') throw new Error('"description" must be a string');
      if (typeof data.ingestion !== 'object') throw new Error('"ingestion" config block must be an object');
      if (typeof data.retrieval !== 'object') throw new Error('"retrieval" config block must be an object');
      if (typeof data.generation !== 'object') throw new Error('"generation" config block must be an object');
      if (typeof data.evaluation !== 'object') throw new Error('"evaluation" config block must be an object');

      setValidationError(null);
    } catch (err: any) {
      setValidationError(err.message);
    }
  };

  const handleSaveClick = async () => {
    if (validationError) return;
    try {
      setIsSubmitting(true);
      const parsedConfig = JSON.parse(jsonCode);
      await onSave(parsedConfig);
      onClose();
    } catch (err: any) {
      setValidationError(`Syntax Error: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-neutral-950/40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="w-full max-w-4xl bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl shadow-2xl flex flex-col overflow-hidden max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-100 dark:border-neutral-850">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded-xl text-blue-500">
              <FileJson className="w-5 h-5" />
            </div>
            <div>
              <h2 className="font-bold text-neutral-900 dark:text-white">
                {initialConfig ? 'Edit RAG Configuration' : 'Configure RAG Pipeline'}
              </h2>
              <span className="text-[10px] text-neutral-400 font-semibold uppercase tracking-wider">
                PipelineConfig JSON Validation Schema
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

        {/* Editor Body */}
        <div className="flex-1 min-h-[400px] border-b border-neutral-150 dark:border-neutral-850">
          <Editor
            height="100%"
            defaultLanguage="json"
            theme="vs-dark"
            value={jsonCode}
            onChange={handleEditorChange}
            options={{
              minimap: { enabled: false },
              fontSize: 13,
              formatOnPaste: true,
              formatOnType: true,
              scrollBeyondLastLine: false,
            }}
          />
        </div>

        {/* Footer info & validation status */}
        <div className="p-5 flex flex-col md:flex-row items-center justify-between gap-4 bg-neutral-50 dark:bg-neutral-900/50">
          <div className="flex items-center gap-2">
            {validationError ? (
              <div className="flex items-center gap-2 text-rose-600 dark:text-rose-400 text-xs font-semibold">
                <AlertTriangle className="w-4 h-4" />
                <span>{validationError}</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 text-xs font-semibold">
                <CheckCircle className="w-4 h-4" />
                <span>Configuration matches JSON schema</span>
              </div>
            )}
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 border border-neutral-200 dark:border-neutral-800 rounded-xl text-neutral-500 hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors text-sm font-semibold"
            >
              Cancel
            </button>
            <button
              onClick={handleSaveClick}
              disabled={!!validationError || isSubmitting}
              className="px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-bold shadow-md shadow-blue-500/10 disabled:opacity-50 transition-colors"
            >
              {isSubmitting ? 'Saving...' : 'Deploy Config'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
