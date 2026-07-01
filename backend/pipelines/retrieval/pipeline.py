"""
pipeline.py — Top-level retrieval pipeline orchestrator.

Wires together:
  QueryProcessor → HybridRetriever → RRF Fusion → CrossEncoder Reranker
  → ContextAssembler

Usage::

    pipeline = RetrievalPipeline()
    result   = await pipeline.run(query="What is HNSW?")
    print(result["context"])

Configuration knobs live in :class:`~pipelines.retrieval.models.RetrievalConfig`.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from opentelemetry import trace

try:
    from config import settings
    from providers.openai_provider import OpenAIProvider
    from pipelines.retrieval.models import RetrievalResult, RetrievalConfig, ProcessedQuery
    from pipelines.retrieval.query_processor import QueryProcessor
    from pipelines.retrieval.retriever import HybridRetriever
    from pipelines.retrieval.fusion import reciprocal_rank_fusion
    from pipelines.retrieval.reranker import CrossEncoderReranker, build_reranker
    from pipelines.retrieval.context_assembler import ContextAssembler
    from pipelines.retrieval.hyde import HyDEGenerator
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from config import settings
    from providers.openai_provider import OpenAIProvider
    from pipelines.retrieval.models import RetrievalResult, RetrievalConfig, ProcessedQuery
    from pipelines.retrieval.query_processor import QueryProcessor
    from pipelines.retrieval.retriever import HybridRetriever
    from pipelines.retrieval.fusion import reciprocal_rank_fusion
    from pipelines.retrieval.reranker import CrossEncoderReranker, build_reranker
    from pipelines.retrieval.context_assembler import ContextAssembler
    from pipelines.retrieval.hyde import HyDEGenerator

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class RetrievalPipeline:
    """Production retrieval pipeline for NeuroFlow.

    Parameters
    ----------
    config:
        Tunable parameters (top-k, token budget, feature flags …).
    provider:
        LLM provider for query expansion, HyDE and classification.
        Defaults to the shared ``OpenAIProvider`` (Groq backend).
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        provider: Optional[OpenAIProvider] = None,
    ):
        self._config = config or RetrievalConfig()
        self._provider = provider or OpenAIProvider()

        self._query_processor = QueryProcessor(provider=self._provider)
        self._retriever = HybridRetriever(provider=self._provider, config=self._config)
        self._reranker = build_reranker(self._config)
        self._assembler = ContextAssembler(token_budget=self._config.context_token_budget)
        self._hyde = HyDEGenerator(provider=self._provider, config=self._config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        *,
        use_hyde: Optional[bool] = None,
        use_reranker: Optional[bool] = None,
        use_query_expansion: Optional[bool] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the full retrieval pipeline.

        Parameters
        ----------
        query:
            Raw user question.
        use_hyde:
            Override the config flag for this call.
        use_reranker:
            Override the config flag for this call.
        use_query_expansion:
            Override the config flag for this call.
        extra_filters:
            Additional metadata filters merged with those extracted from
            the query.

        Returns
        -------
        dict with keys:
            ``context``     — formatted prompt-ready string
            ``chunks_used`` — list of chunk IDs
            ``total_tokens``— token count of context
            ``sources``     — source metadata list
            ``reranked``    — final list of :class:`RetrievalResult`
            ``latency_ms``  — wall-clock time for the full pipeline
            ``pipeline_meta`` — debug info (retriever counts, flags …)
        """
        t0 = time.perf_counter()

        # Resolve feature flags
        _use_hyde = use_hyde if use_hyde is not None else self._config.use_hyde
        _use_reranker = use_reranker if use_reranker is not None else self._config.use_reranker
        _use_expansion = (
            use_query_expansion
            if use_query_expansion is not None
            else self._config.use_query_expansion
        )

        pipeline_uuid = extra_filters.get("pipeline_id") if extra_filters else None
        if not pipeline_uuid:
            pipeline_uuid = "default"
        run_uuid = extra_filters.get("run_id") if extra_filters else "default"

        with tracer.start_as_current_span("retrieval.pipeline") as span:
            span.set_attribute("pipeline_id", str(pipeline_uuid))
            span.set_attribute("run_id", str(run_uuid))
            span.set_attribute("query", query)
            span.set_attribute("use_hyde", _use_hyde)
            span.set_attribute("use_reranker", _use_reranker)
            span.set_attribute("reranker", self._config.reranker_model or "cohere-rerank-v3")

            # ---- Step 1: Query processing ----------------------------------------
            processed: ProcessedQuery = await self._query_processor.process(query)
            if not _use_expansion:
                processed.expanded = []
            if extra_filters:
                processed.metadata_filters.update(extra_filters)

            span.set_attribute("query_type", processed.query_type.value)

            # ---- Step 2: HyDE (optional) -----------------------------------------
            hyde_embedding: Optional[List[float]] = None
            if _use_hyde:
                hyde_embedding = await self._hyde.get_hyde_embedding(query)

            # ---- Step 3: Hybrid retrieval ----------------------------------------
            retrieval_map = await self._retriever.retrieve(
                processed_query=processed,
                query_embedding=hyde_embedding,
            )
            dense_results = retrieval_map["dense"]
            sparse_results = retrieval_map["sparse"]
            meta_results = retrieval_map["metadata"]

            # ---- Step 4: RRF fusion ----------------------------------------------
            with tracer.start_as_current_span("retrieval.fusion") as fusion_span:
                fusion_span.set_attribute("pipeline_id", str(pipeline_uuid))
                fusion_span.set_attribute("run_id", str(run_uuid))
                all_lists = [lst for lst in [dense_results, sparse_results, meta_results] if lst]
                fused: List[RetrievalResult] = (
                    reciprocal_rank_fusion(all_lists, k=self._config.rrf_k)
                    if all_lists
                    else []
                )
                fusion_span.set_attribute("fused_count", len(fused))

            # ---- Step 5: Reranking -----------------------------------------------
            with tracer.start_as_current_span("retrieval.rerank") as rerank_span:
                rerank_span.set_attribute("pipeline_id", str(pipeline_uuid))
                rerank_span.set_attribute("run_id", str(run_uuid))
                rerank_span.set_attribute("reranker", self._config.reranker_model or "cohere-rerank-v3")
                final_results: List[RetrievalResult]
                if _use_reranker and fused:
                    final_results = await self._reranker.rerank(
                        query=query,
                        candidates=fused,
                        top_k=self._config.rerank_final_k,
                    )
                else:
                    final_results = fused[: self._config.rerank_final_k]
                rerank_span.set_attribute("reranked_count", len(final_results))

            # ---- Step 6: Context assembly ----------------------------------------
            with tracer.start_as_current_span("retrieval.assemble") as assemble_span:
                assemble_span.set_attribute("pipeline_id", str(pipeline_uuid))
                assemble_span.set_attribute("run_id", str(run_uuid))
                assembled = self._assembler.assemble(final_results)
                assemble_span.set_attribute("token_count", assembled.get("total_tokens", 0))

            latency_ms = (time.perf_counter() - t0) * 1_000
            span.set_attribute("latency_ms", latency_ms)
            span.set_attribute("chunks_used", len(assembled["chunks_used"]))
            span.set_attribute("retrieved_chunks", ",".join(assembled["chunks_used"]))

            # Prometheus Metrics Update
            from monitoring import metrics
            metrics.retrieval_latency.labels(pipeline_id=str(pipeline_uuid)).observe(latency_ms / 1000.0)

            return {
                **assembled,
                "reranked": final_results,
                "latency_ms": round(latency_ms, 2),
                "pipeline_meta": {
                    "query_type": processed.query_type.value,
                    "expansions": processed.expanded,
                    "metadata_filters": processed.metadata_filters,
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "metadata_count": len(meta_results),
                    "fused_count": len(fused),
                    "use_hyde": _use_hyde,
                    "use_reranker": _use_reranker,
                },
            }

    # ------------------------------------------------------------------
    # Convenience: run without DB (for eval / testing with mock data)
    # ------------------------------------------------------------------

    async def run_from_candidates(
        self,
        query: str,
        candidates: List[RetrievalResult],
        *,
        use_reranker: bool = True,
    ) -> Dict[str, Any]:
        """Run only the reranking + context-assembly steps on pre-supplied chunks.

        Useful in evaluation when retrieval is mocked.
        """
        if use_reranker and candidates:
            final = await self._reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=self._config.rerank_final_k,
            )
        else:
            final = candidates[: self._config.rerank_final_k]

        assembled = self._assembler.assemble(final)
        return {**assembled, "reranked": final}
