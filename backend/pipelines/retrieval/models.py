"""
models.py — Shared data-classes for the retrieval pipeline.

All inter-module communication uses these types so that each component is
easily swappable and testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core result type
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single retrieved chunk with its provenance and relevance score."""

    chunk_id: str
    content: str
    score: float
    source: str  # e.g. filename / URL
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Rich provenance populated from DB row
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    token_count: Optional[int] = None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RetrievalResult(chunk_id={self.chunk_id!r}, "
            f"score={self.score:.4f}, source={self.source!r})"
        )


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"


# ---------------------------------------------------------------------------
# Processed query — output of QueryProcessor
# ---------------------------------------------------------------------------

@dataclass
class ProcessedQuery:
    """Enriched query produced by the query-processor step."""

    original: str
    expanded: List[str] = field(default_factory=list)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    query_type: QueryType = QueryType.FACTUAL


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    """Tunable knobs that control the whole pipeline."""

    # How many candidates each retriever returns
    dense_top_k: int = 20
    sparse_top_k: int = 20
    metadata_top_k: int = 10

    # How many candidates enter the cross-encoder
    rerank_top_n: int = 40

    # How many results come out of the cross-encoder
    rerank_final_k: int = 10

    # RRF constant
    rrf_k: int = 60

    # Context-assembly token budget
    context_token_budget: int = 4_000

    # Whether to use HyDE
    use_hyde: bool = False

    # Whether to run cross-encoder reranking
    use_reranker: bool = True

    # Whether to expand the query
    use_query_expansion: bool = True

    # Cross-encoder model identifier
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Embedding model used for dense retrieval
    embedding_model: str = "text-embedding-3-small"
