from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class IngestionConfig(BaseModel):
    chunking_strategy: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    extractors_enabled: List[str]

    model_config = {
        "extra": "forbid"
    }

class RetrievalConfig(BaseModel):
    dense_k: int
    sparse_k: int
    reranker: str
    top_k_after_rerank: int
    query_expansion: bool
    metadata_filters_enabled: bool

    model_config = {
        "extra": "forbid"
    }

class GenerationConfig(BaseModel):
    model_routing: Dict
    max_context_tokens: int
    temperature: float
    system_prompt_variant: str

    model_config = {
        "extra": "forbid"
    }

class EvaluationConfig(BaseModel):
    auto_evaluate: bool
    training_threshold: float

    model_config = {
        "extra": "forbid"
    }

class PipelineConfig(BaseModel):
    name: str
    description: str
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig

    model_config = {
        "extra": "forbid"
    }
