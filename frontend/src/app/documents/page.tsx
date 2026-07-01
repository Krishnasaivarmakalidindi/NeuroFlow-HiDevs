'use client';

import React, { useState, useEffect } from 'react';
import { useDocuments } from '../../hooks/useDocuments';
import { documentService } from '../../services/api';
import { useAppStore } from '../../store/appStore';
import { UploadCloud, File, AlertCircle, FileText, ChevronRight, Search, Activity, Sparkles, Check } from 'lucide-react';
import { Document, SimilarChunk } from '../../types';

export default function DocumentsPage() {
  const { documents, isLoading, refetch, uploadDocument } = useDocuments();
  const { addDocument, updateDocumentStatus } = useAppStore();

  const [dragActive, setDragActive] = useState(false);
  const [uploads, setUploads] = useState<{ name: string; progress: number; status: 'uploading' | 'done' | 'error' }[]>([]);
  
  // Selection / Search details state
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [similarChunks, setSimilarChunks] = useState<SimilarChunk[]>([]);
  const [isSearchingSimilar, setIsSearchingSimilar] = useState(false);

  // Poll for document status updates if any is 'queued' or 'processing'
  useEffect(() => {
    const activeDocs = documents.filter((doc) => doc.status === 'queued' || doc.status === 'processing');
    if (activeDocs.length === 0) return;

    const interval = setInterval(async () => {
      let changed = false;
      for (const doc of activeDocs) {
        try {
          const detail = await documentService.get(doc.id);
          if (detail.status !== doc.status) {
            updateDocumentStatus(doc.id, detail.status);
            changed = true;
          }
        } catch (err) {
          console.error(err);
        }
      }
      if (changed) refetch();
    }, 5000);

    return () => clearInterval(interval);
  }, [documents, refetch, updateDocumentStatus]);

  // Load document detail
  const handleDocClick = async (doc: Document) => {
    setSelectedDocId(doc.id);
    setSelectedDoc(null);
    setSimilarChunks([]);
    try {
      const detail = await documentService.get(doc.id);
      setSelectedDoc(detail);
      
      // Load default similarity search
      setIsSearchingSimilar(true);
      const similar = await documentService.similar(doc.id);
      setSimilarChunks(similar);
    } catch (err) {
      console.error('Failed to fetch document detail', err);
    } finally {
      setIsSearchingSimilar(false);
    }
  };

  // Drag and drop handlers
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await handleFilesUpload(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileInputChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      await handleFilesUpload(Array.from(e.target.files));
    }
  };

  const handleFilesUpload = async (files: File[]) => {
    const newUploads = files.map((f) => ({ name: f.name, progress: 0, status: 'uploading' as const }));
    setUploads((prev) => [...newUploads, ...prev]);

    for (let idx = 0; idx < files.length; idx++) {
      const file = files[idx];
      try {
        await uploadDocument({
          file,
          onProgress: (progressEvent) => {
            const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploads((prev) =>
              prev.map((up) => (up.name === file.name ? { ...up, progress: percent } : up))
            );
          },
        });
        
        setUploads((prev) =>
          prev.map((up) => (up.name === file.name ? { ...up, status: 'done', progress: 100 } : up))
        );
        refetch();
      } catch (err) {
        setUploads((prev) =>
          prev.map((up) => (up.name === file.name ? { ...up, status: 'error' } : up))
        );
      }
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      {/* Left side: Upload and List */}
      <div className="lg:col-span-2 space-y-6">
        <div>
          <h1 className="text-2xl font-bold dark:text-white">Knowledge Ingestion</h1>
          <p className="text-sm text-neutral-500 dark:text-neutral-400 font-medium">
            Upload PDFs, Markdown, text files, or URLs to build your knowledge chunks.
          </p>
        </div>

        {/* Drag and drop upload zone */}
        <div
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center text-center transition-all duration-300 ${
            dragActive
              ? 'border-blue-500 bg-blue-500/5'
              : 'border-neutral-250 dark:border-neutral-800 bg-white dark:bg-neutral-900'
          }`}
        >
          <UploadCloud className="w-12 h-12 text-neutral-400 mb-3 animate-pulse" />
          <h3 className="text-sm font-bold text-neutral-800 dark:text-neutral-200">
            Drag and drop your document here
          </h3>
          <span className="text-xs text-neutral-400 font-medium mt-1">PDF, TXT, MD, DOCX up to 100MB</span>
          
          <label className="mt-4 bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs px-4 py-2 rounded-xl cursor-pointer shadow-md shadow-blue-500/10">
            Browse Files
            <input type="file" multiple onChange={handleFileInputChange} className="hidden" />
          </label>
        </div>

        {/* Upload progress bars */}
        {uploads.length > 0 && (
          <div className="p-4 bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl space-y-3">
            <h4 className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider">Active Uploads</h4>
            <div className="space-y-2">
              {uploads.map((up, idx) => (
                <div key={idx} className="flex items-center justify-between gap-4 text-xs font-semibold">
                  <div className="flex items-center gap-2 truncate max-w-[200px]">
                    <File className="w-4 h-4 text-neutral-400" />
                    <span className="text-neutral-700 dark:text-neutral-300 truncate">{up.name}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {up.status === 'uploading' && (
                      <>
                        <div className="w-24 bg-neutral-200 dark:bg-neutral-800 h-1 rounded-full overflow-hidden">
                          <div className="bg-blue-500 h-full" style={{ width: `${up.progress}%` }} />
                        </div>
                        <span className="text-[10px] text-neutral-400">{up.progress}%</span>
                      </>
                    )}
                    {up.status === 'done' && <span className="text-emerald-500 flex items-center gap-1"><Check className="w-3.5 h-3.5"/> Done</span>}
                    {up.status === 'error' && <span className="text-rose-500 flex items-center gap-1"><AlertCircle className="w-3.5 h-3.5"/> Failed</span>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Ingested Documents list table */}
        <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl overflow-hidden shadow-sm">
          <div className="p-5 border-b border-neutral-100 dark:border-neutral-850">
            <h2 className="text-sm font-bold text-neutral-800 dark:text-neutral-100">Ingested Knowledge Base</h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm border-collapse">
              <thead>
                <tr className="border-b border-neutral-100 dark:border-neutral-850 text-neutral-400 font-bold text-[10px] uppercase tracking-wider bg-neutral-50/50 dark:bg-neutral-900/30">
                  <th className="px-6 py-4">Filename</th>
                  <th className="px-6 py-4">Type</th>
                  <th className="px-6 py-4">Status</th>
                  <th className="px-6 py-4 text-right">Chunks</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-100 dark:divide-neutral-850">
                {isLoading ? (
                  <tr>
                    <td colSpan={4} className="px-6 py-8 text-center text-neutral-400 font-semibold">
                      Loading ingested files...
                    </td>
                  </tr>
                ) : documents.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="px-6 py-8 text-center text-neutral-400 font-semibold">
                      No documents ingested yet. Upload files to get started!
                    </td>
                  </tr>
                ) : (
                  documents.map((doc) => {
                    const isSelected = selectedDocId === doc.id;
                    const statusClass = 
                      doc.status === 'complete' 
                        ? 'text-emerald-600 bg-emerald-50 dark:bg-emerald-950/20'
                        : doc.status === 'failed'
                        ? 'text-rose-600 bg-rose-50 dark:bg-rose-950/20'
                        : 'text-blue-600 bg-blue-50 dark:bg-blue-950/20 animate-pulse'; // Pulsing blue status indicator
                        
                    return (
                      <tr
                        key={doc.id}
                        onClick={() => handleDocClick(doc)}
                        className={`hover:bg-neutral-50 dark:hover:bg-neutral-850/20 cursor-pointer transition-colors ${
                          isSelected ? 'bg-neutral-50 dark:bg-neutral-850/40' : ''
                        }`}
                      >
                        <td className="px-6 py-4 font-bold text-neutral-800 dark:text-neutral-200">
                          {doc.filename}
                        </td>
                        <td className="px-6 py-4 text-xs font-bold text-neutral-450 uppercase">
                          {doc.type}
                        </td>
                        <td className="px-6 py-4">
                          <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-bold ${statusClass}`}>
                            {doc.status === 'queued' || doc.status === 'processing' ? (
                              <Activity className="w-3 h-3 animate-spin" />
                            ) : null}
                            {doc.status}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-right font-bold text-neutral-800 dark:text-neutral-200">
                          {doc.chunk_count}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Right side: Inspector Panel */}
      <div className="lg:col-span-1 p-6 bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl shadow-sm h-fit space-y-6">
        <div className="border-b border-neutral-100 dark:border-neutral-850 pb-4">
          <h2 className="text-sm font-bold text-neutral-800 dark:text-neutral-100">Document Inspector</h2>
          <span className="text-[9px] text-neutral-400 font-bold uppercase tracking-wider">
            Semantic chunk explorer
          </span>
        </div>

        {!selectedDoc ? (
          <div className="text-center py-12 text-neutral-450 font-semibold text-xs space-y-2">
            <FileText className="w-10 h-10 text-neutral-300 mx-auto" />
            <p>Select a document to inspect chunks and search similar matches.</p>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Metadata Info */}
            <div className="p-4 bg-neutral-50 dark:bg-neutral-850 rounded-xl space-y-2 border border-neutral-100 dark:border-neutral-800/40">
              <span className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider block">Document Attributes</span>
              <div className="text-xs space-y-1.5 font-semibold text-neutral-700 dark:text-neutral-350">
                <div className="flex justify-between">
                  <span>Name:</span>
                  <span className="text-neutral-900 dark:text-neutral-100">{selectedDoc.filename}</span>
                </div>
                <div className="flex justify-between">
                  <span>Chunks:</span>
                  <span className="text-neutral-900 dark:text-neutral-100">{selectedDoc.chunk_count}</span>
                </div>
                <div className="flex justify-between">
                  <span>Created:</span>
                  <span className="text-neutral-900 dark:text-neutral-100">{selectedDoc.created_at}</span>
                </div>
              </div>
            </div>

            {/* Semantic Similar Chunk Search */}
            <div className="space-y-3">
              <h3 className="text-xs font-bold text-neutral-400 uppercase tracking-wider flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-blue-500 animate-pulse" />
                Similar Chunks Finder
              </h3>
              
              {isSearchingSimilar ? (
                <div className="flex justify-center py-6">
                  <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                </div>
              ) : (
                <div className="space-y-3 max-h-[350px] overflow-y-auto">
                  {similarChunks.map((chunk) => (
                    <div key={chunk.chunk_id} className="p-3 bg-neutral-50 dark:bg-neutral-850/40 border border-neutral-100 dark:border-neutral-800/50 rounded-xl space-y-2">
                      <div className="flex items-center justify-between text-[10px] font-bold">
                        <span className="text-neutral-400 font-mono">ID: {chunk.chunk_id.substring(6)}</span>
                        <span className="text-emerald-500 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                          {(chunk.similarity_score * 100).toFixed(0)}% Match
                        </span>
                      </div>
                      <p className="text-xs text-neutral-600 dark:text-neutral-350 leading-relaxed font-semibold">
                        &ldquo;{chunk.content}&rdquo;
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
