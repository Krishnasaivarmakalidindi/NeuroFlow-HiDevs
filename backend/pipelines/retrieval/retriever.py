"""
retriever.py — Hybrid retrieval (dense + sparse + metadata).

All three retrieval strategies run concurrently via ``asyncio.gather``:

  * Dense retrieval   — pgvector cosine-distance ordering
  * Sparse retrieval  — PostgreSQL full-text search (tsvector / tsquery)
  * Metadata retrieval — JSONB containment filter + vector ordering

The embeddings for the query (and expanded variants) are generated once and
reused across all dense sub-queries.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from opentelemetry import trace

try:
    from db.pool import DatabasePool
    from providers.openai_provider import OpenAIProvider
    from pipelines.retrieval.models import RetrievalResult, RetrievalConfig, ProcessedQuery
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from db.pool import DatabasePool
    from providers.openai_provider import OpenAIProvider
    from pipelines.retrieval.models import RetrievalResult, RetrievalConfig, ProcessedQuery

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

_DENSE_SQL = """
    SELECT
        c.id::text            AS chunk_id,
        c.content,
        c.metadata,
        c.chunk_index,
        c.token_count,
        d.filename            AS source,
        d.id::text            AS document_id,
        1 - (c.embedding <=> $1::vector) AS score
    FROM   chunks c
    JOIN   documents d ON d.id = c.document_id
    WHERE  c.embedding IS NOT NULL
    ORDER  BY c.embedding <=> $1::vector
    LIMIT  $2
"""

_SPARSE_SQL = """
    SELECT
        c.id::text                                               AS chunk_id,
        c.content,
        c.metadata,
        c.chunk_index,
        c.token_count,
        d.filename                                               AS source,
        d.id::text                                               AS document_id,
        ts_rank_cd(to_tsvector('english', c.content),
                   plainto_tsquery('english', $1))               AS score
    FROM   chunks c
    JOIN   documents d ON d.id = c.document_id
    WHERE  to_tsvector('english', c.content)
           @@ plainto_tsquery('english', $1)
    ORDER  BY score DESC
    LIMIT  $2
"""

_METADATA_SQL = """
    SELECT
        c.id::text            AS chunk_id,
        c.content,
        c.metadata,
        c.chunk_index,
        c.token_count,
        d.filename            AS source,
        d.id::text            AS document_id,
        CASE WHEN c.embedding IS NOT NULL
             THEN 1 - (c.embedding <=> $2::vector)
             ELSE 0.0
        END                   AS score
    FROM   chunks c
    JOIN   documents d ON d.id = c.document_id
    WHERE  (c.metadata @> $1::jsonb OR d.metadata @> $1::jsonb)
    ORDER  BY score DESC
    LIMIT  $3
"""


def _row_to_result(row: Any) -> RetrievalResult:
    """Convert an asyncpg Record to a RetrievalResult."""
    return RetrievalResult(
        chunk_id=row["chunk_id"],
        content=row["content"],
        score=float(row["score"] or 0.0),
        source=row["source"],
        metadata=dict(row["metadata"] or {}),
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        token_count=row["token_count"],
    )


class HybridRetriever:
    """Orchestrates concurrent dense, sparse, and metadata retrieval."""

    def __init__(
        self,
        provider: Optional[OpenAIProvider] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self._provider = provider or OpenAIProvider()
        self._config = config or RetrievalConfig()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        processed_query: ProcessedQuery,
        query_embedding: Optional[List[float]] = None,
    ) -> Dict[str, List[RetrievalResult]]:
        """Run all retrieval strategies in parallel.

        Returns a dict with keys ``dense``, ``sparse``, ``metadata``.
        """
        cfg = self._config

        # Embed the original query (and expansions) once
        all_queries = [processed_query.original] + processed_query.expanded
        if query_embedding is None:
            embeddings = await self._embed_queries(all_queries)
        else:
            # Caller pre-computed the query embedding (e.g. HyDE)
            embeddings = [query_embedding] + await self._embed_queries(
                processed_query.expanded
            ) if processed_query.expanded else [query_embedding]

        with tracer.start_as_current_span("retrieval.hybrid") as span:
            span.set_attribute("num_query_variants", len(embeddings))

            dense_task = self._dense_retrieval_multi(embeddings, cfg.dense_top_k)
            sparse_task = self._sparse_retrieval(processed_query.original, cfg.sparse_top_k)
            meta_task = self._metadata_retrieval(
                processed_query.metadata_filters,
                embeddings[0],
                cfg.metadata_top_k,
            )

            dense_results, sparse_results, meta_results = await asyncio.gather(
                dense_task, sparse_task, meta_task
            )

        return {
            "dense": dense_results,
            "sparse": sparse_results,
            "metadata": meta_results,
        }

    # ------------------------------------------------------------------
    # Dense
    # ------------------------------------------------------------------

    async def _embed_queries(self, queries: List[str]) -> List[List[float]]:
        """Embed a list of queries, returning one vector per query."""
        if not queries:
            return []
        try:
            return await self._provider.embed(queries, model=self._config.embedding_model)
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            return [[0.0] * 1536 for _ in queries]

    async def _dense_single(
        self,
        pool,
        embedding: List[float],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Run one pgvector cosine-distance query."""
        async with pool.acquire() as conn:
            rows = await conn.fetch(_DENSE_SQL, str(embedding), top_k)
        return [_row_to_result(r) for r in rows]

    async def _dense_retrieval_multi(
        self,
        embeddings: List[List[float]],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Run dense retrieval for every query variant and union the results."""
        with tracer.start_as_current_span("retrieval.dense") as span:
            try:
                pool = await DatabasePool.get_pool()
                tasks = [
                    self._dense_single(pool, emb, top_k) for emb in embeddings
                ]
                per_query = await asyncio.gather(*tasks, return_exceptions=True)

                seen: Dict[str, RetrievalResult] = {}
                for batch in per_query:
                    if isinstance(batch, Exception):
                        logger.warning("Dense retrieval batch failed: %s", batch)
                        continue
                    for r in batch:
                        if r.chunk_id not in seen or r.score > seen[r.chunk_id].score:
                            seen[r.chunk_id] = r

                results = sorted(seen.values(), key=lambda x: x.score, reverse=True)
                span.set_attribute("results_count", len(results))
                return results

            except Exception as exc:
                logger.error("Dense retrieval failed: %s", exc)
                span.record_exception(exc)
                return []

    # ------------------------------------------------------------------
    # Sparse
    # ------------------------------------------------------------------

    async def _sparse_retrieval(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Full-text search using PostgreSQL tsvector + tsquery."""
        with tracer.start_as_current_span("retrieval.sparse") as span:
            try:
                pool = await DatabasePool.get_pool()
                async with pool.acquire() as conn:
                    rows = await conn.fetch(_SPARSE_SQL, query, top_k)
                results = [_row_to_result(r) for r in rows]
                span.set_attribute("results_count", len(results))
                return results
            except Exception as exc:
                logger.error("Sparse retrieval failed: %s", exc)
                span.record_exception(exc)
                return []

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    async def _metadata_retrieval(
        self,
        filters: Dict[str, Any],
        query_embedding: List[float],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Filter by JSONB containment then order by cosine similarity."""
        with tracer.start_as_current_span("retrieval.metadata") as span:
            if not filters:
                span.set_attribute("skipped", True)
                return []
            try:
                pool = await DatabasePool.get_pool()
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        _METADATA_SQL,
                        json.dumps(filters),
                        str(query_embedding),
                        top_k,
                    )
                results = [_row_to_result(r) for r in rows]
                span.set_attribute("results_count", len(results))
                return results
            except Exception as exc:
                logger.error("Metadata retrieval failed: %s", exc)
                span.record_exception(exc)
                return []
