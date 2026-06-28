"""
reranker.py — Cross-encoder reranking with optional API-based fallback.

The primary reranker uses the Hugging Face ``sentence-transformers`` library
to load ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (≈ 22 M parameters, fast
on CPU).

Workflow:
  1. Take the top-N candidates from RRF (default 40).
  2. Score every (query, chunk) pair with the cross-encoder.
  3. Return the top-K by cross-encoder score (default 10).

An optional API reranker interface is also provided for services like
Cohere Rerank or Jina.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from opentelemetry import trace

try:
    from pipelines.retrieval.models import RetrievalResult, RetrievalConfig
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from pipelines.retrieval.models import RetrievalResult, RetrievalConfig

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Score and return the top-k candidates."""


# ---------------------------------------------------------------------------
# Cross-encoder (local)
# ---------------------------------------------------------------------------

class CrossEncoderReranker(BaseReranker):
    """
    Uses ``cross-encoder/ms-marco-MiniLM-L-6-v2`` loaded via
    sentence-transformers.  The model is loaded lazily on first call and
    cached for the lifetime of the process.
    """

    _MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _model = None  # module-level cache

    def __init__(self, model_name: Optional[str] = None, top_input: int = 40):
        self._model_name = model_name or self._MODEL_NAME
        self._top_input = top_input  # how many candidates enter the cross-encoder

    @classmethod
    def _load_model(cls, model_name: str):
        if cls._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading cross-encoder model: %s", model_name)
                cls._model = CrossEncoder(model_name, max_length=512)
                logger.info("Cross-encoder model loaded.")
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
        return cls._model

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Score candidates with the cross-encoder and return top-k."""
        with tracer.start_as_current_span("retrieval.rerank") as span:
            span.set_attribute("candidates_in", len(candidates))
            span.set_attribute("top_k", top_k)
            span.set_attribute("model", self._model_name)

            if not candidates:
                return []

            # Trim to top-N before scoring (expensive)
            pool = candidates[: self._top_input]

            model = self._load_model(self._model_name)

            # Build (query, passage) pairs
            pairs = [(query, r.content) for r in pool]

            # Run synchronously — cross-encoder inference is CPU-bound
            import asyncio, functools
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                functools.partial(model.predict, pairs, show_progress_bar=False),
            )

            # Attach scores and sort
            scored = [
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    score=float(s),
                    source=r.source,
                    metadata=r.metadata,
                    document_id=r.document_id,
                    chunk_index=r.chunk_index,
                    token_count=r.token_count,
                )
                for r, s in zip(pool, scores)
            ]
            scored.sort(key=lambda x: x.score, reverse=True)

            results = scored[:top_k]
            span.set_attribute("results_out", len(results))
            return results


# ---------------------------------------------------------------------------
# Optional API-based reranker (Cohere / Jina compatible)
# ---------------------------------------------------------------------------

class APIReranker(BaseReranker):
    """
    Reranker backed by an external HTTP API (e.g. Cohere Rerank).

    Expects the API to follow the Cohere Rerank v2 schema:
      POST /rerank  →  {"results": [{"index": int, "relevance_score": float}]}
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str = "rerank-english-v3.0",
    ):
        self._api_url = api_url
        self._api_key = api_key
        self._model = model

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        with tracer.start_as_current_span("retrieval.rerank.api") as span:
            span.set_attribute("candidates_in", len(candidates))
            span.set_attribute("model", self._model)

            try:
                import httpx

                documents = [r.content for r in candidates]
                payload = {
                    "model": self._model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_k,
                }
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(self._api_url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()

                reranked = []
                for item in data.get("results", []):
                    idx = item["index"]
                    score = item["relevance_score"]
                    r = candidates[idx]
                    reranked.append(
                        RetrievalResult(
                            chunk_id=r.chunk_id,
                            content=r.content,
                            score=float(score),
                            source=r.source,
                            metadata=r.metadata,
                            document_id=r.document_id,
                            chunk_index=r.chunk_index,
                            token_count=r.token_count,
                        )
                    )

                span.set_attribute("results_out", len(reranked))
                return reranked

            except Exception as exc:
                logger.error("API reranker failed: %s", exc)
                span.record_exception(exc)
                # Graceful degradation — return original ordering
                return candidates[:top_k]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_reranker(config: Optional[RetrievalConfig] = None) -> BaseReranker:
    """Return the default (local cross-encoder) reranker."""
    cfg = config or RetrievalConfig()
    return CrossEncoderReranker(
        model_name=cfg.cross_encoder_model,
        top_input=cfg.rerank_top_n,
    )
