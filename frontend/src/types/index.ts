export interface IngestionConfig {
  chunking_strategy: string;
  chunk_size_tokens: number;
  chunk_overlap_tokens: number;
  extractors_enabled: string[];
}

export interface RetrievalConfig {
  dense_k: number;
  sparse_k: number;
  reranker: string;
  top_k_after_rerank: number;
  query_expansion: boolean;
  metadata_filters_enabled: boolean;
}

export interface GenerationConfig {
  model_routing: Record<string, any>;
  max_context_tokens: number;
  temperature: number;
  system_prompt_variant: string;
}

export interface EvaluationConfig {
  auto_evaluate: boolean;
  training_threshold: number;
}

export interface PipelineConfig {
  name: string;
  description: string;
  ingestion: IngestionConfig;
  retrieval: RetrievalConfig;
  generation: GenerationConfig;
  evaluation: EvaluationConfig;
}

export interface Pipeline {
  id: string;
  name: string;
  version: string;
  config: PipelineConfig;
  created_at?: string;
  updated_at?: string;
  // Computed values
  queries_count?: number;
  avg_score?: number;
  sparkline?: number[];
}

export interface EvaluationMetrics {
  faithfulness: number;
  answer_relevance: number;
  context_precision: number;
  context_recall: number;
  overall: number;
}

export interface Run {
  run_id: string;
  query?: string;
  answer?: string;
  pipeline_id?: string;
  pipeline_name?: string;
  created_at?: string;
  latency?: number;
  tokens?: number;
  evaluation?: EvaluationMetrics;
  rating?: number;
}

export interface Document {
  id: string;
  filename: string;
  type: string;
  status: 'queued' | 'processing' | 'complete' | 'failed';
  chunk_count: number;
  created_at: string;
  chunks?: { id: string; content: string; metadata: any }[];
  metadata?: any;
}

export interface SimilarChunk {
  chunk_id: string;
  content: string;
  similarity_score: number;
  metadata: any;
}

export interface LatencyStats {
  p50: number;
  p95: number;
  p99: number;
}

export interface PipelineAnalytics {
  retrieval_latency: LatencyStats;
  generation_latency: number;
  evaluation: EvaluationMetrics;
  cost_per_query: number;
  queries_per_day: { day: string; count: number }[];
}

export interface SSEMessage {
  type: 'retrieval_start' | 'retrieval_complete' | 'token' | 'done' | 'error' | 'keepalive';
  message?: string;
  data?: any;
  chunk?: string;
  citations?: any[];
}
