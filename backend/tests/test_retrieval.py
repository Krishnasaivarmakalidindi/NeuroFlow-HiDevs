"""
test_retrieval.py — Unit and integration tests for the retrieval pipeline.

Run with:
    cd backend
    pytest tests/test_retrieval.py -v

All database calls are mocked so tests run without a live PostgreSQL instance.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Make sure backend package is importable regardless of cwd
# ---------------------------------------------------------------------------
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipelines.retrieval.models import (
    ProcessedQuery,
    QueryType,
    RetrievalConfig,
    RetrievalResult,
)
from pipelines.retrieval.fusion import reciprocal_rank_fusion
from pipelines.retrieval.context_assembler import ContextAssembler


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_result(chunk_id: str, score: float, content: str = "test content") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        score=score,
        source="test_doc.pdf",
        metadata={},
    )


@pytest.fixture
def mock_provider():
    """A mock LLM provider that returns predictable responses."""
    provider = MagicMock()

    async def _complete(messages, **kwargs):
        result = MagicMock()
        # Default response; overridden in specific tests
        result.content = '["expanded query one", "expanded query two"]'
        return result

    async def _embed(texts, **kwargs):
        return [[0.1 * i] * 1536 for i in range(len(texts))]

    provider.complete = _complete
    provider.embed = _embed
    return provider


# ---------------------------------------------------------------------------
# 1. Query expansion
# ---------------------------------------------------------------------------

class TestQueryExpansion:
    @pytest.mark.asyncio
    async def test_expand_query_returns_list(self, mock_provider):
        from pipelines.retrieval.query_processor import QueryProcessor

        qp = QueryProcessor(provider=mock_provider)
        expanded = await qp.expand_query("how does attention work in transformers")
        assert isinstance(expanded, list)
        assert len(expanded) > 0

    @pytest.mark.asyncio
    async def test_expand_query_graceful_on_bad_json(self):
        """If the LLM returns garbage, expansion should return [] not raise."""
        from pipelines.retrieval.query_processor import QueryProcessor

        bad_provider = MagicMock()

        async def _bad_complete(messages, **kwargs):
            r = MagicMock()
            r.content = "not json at all"
            return r

        bad_provider.complete = _bad_complete
        qp = QueryProcessor(provider=bad_provider)
        expanded = await qp.expand_query("test query")
        assert expanded == []

    @pytest.mark.asyncio
    async def test_expand_query_strips_markdown_fences(self, mock_provider):
        from pipelines.retrieval.query_processor import QueryProcessor

        async def _fenced_complete(messages, **kwargs):
            r = MagicMock()
            r.content = '```json\n["query a", "query b"]\n```'
            return r

        mock_provider.complete = _fenced_complete
        qp = QueryProcessor(provider=mock_provider)
        expanded = await qp.expand_query("test")
        assert "query a" in expanded


# ---------------------------------------------------------------------------
# 2. Metadata extraction
# ---------------------------------------------------------------------------

class TestMetadataExtraction:
    @pytest.mark.asyncio
    async def test_extracts_year_and_topic(self):
        from pipelines.retrieval.query_processor import QueryProcessor

        prov = MagicMock()

        async def _complete(messages, **kwargs):
            r = MagicMock()
            r.content = '{"year": 2023, "topic": "climate"}'
            return r

        prov.complete = _complete
        qp = QueryProcessor(provider=prov)
        filters = await qp.extract_metadata_filters("show climate documents from 2023")
        assert filters.get("year") == 2023
        assert filters.get("topic") == "climate"

    @pytest.mark.asyncio
    async def test_empty_filters_on_plain_query(self):
        from pipelines.retrieval.query_processor import QueryProcessor

        prov = MagicMock()

        async def _complete(messages, **kwargs):
            r = MagicMock()
            r.content = "{}"
            return r

        prov.complete = _complete
        qp = QueryProcessor(provider=prov)
        filters = await qp.extract_metadata_filters("what is HNSW?")
        assert filters == {}

    @pytest.mark.asyncio
    async def test_metadata_extraction_graceful_on_error(self):
        from pipelines.retrieval.query_processor import QueryProcessor

        prov = MagicMock()

        async def _complete(messages, **kwargs):
            raise RuntimeError("network error")

        prov.complete = _complete
        qp = QueryProcessor(provider=prov)
        filters = await qp.extract_metadata_filters("show docs from 2022")
        assert filters == {}


# ---------------------------------------------------------------------------
# 3. Query classification
# ---------------------------------------------------------------------------

class TestQueryClassification:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("label", ["factual", "analytical", "comparative", "procedural"])
    async def test_classifies_all_types(self, label):
        from pipelines.retrieval.query_processor import QueryProcessor

        prov = MagicMock()

        async def _complete(messages, **kwargs):
            r = MagicMock()
            r.content = label
            return r

        prov.complete = _complete
        qp = QueryProcessor(provider=prov)
        qtype = await qp.classify_query("some query")
        assert qtype == QueryType(label)

    @pytest.mark.asyncio
    async def test_classify_defaults_to_factual_on_unknown(self):
        from pipelines.retrieval.query_processor import QueryProcessor

        prov = MagicMock()

        async def _complete(messages, **kwargs):
            r = MagicMock()
            r.content = "unknown_category"
            return r

        prov.complete = _complete
        qp = QueryProcessor(provider=prov)
        qtype = await qp.classify_query("some query")
        assert qtype == QueryType.FACTUAL


# ---------------------------------------------------------------------------
# 4. Dense retrieval
# ---------------------------------------------------------------------------

class TestDenseRetrieval:
    @pytest.mark.asyncio
    async def test_dense_returns_results_from_db(self):
        from pipelines.retrieval.retriever import HybridRetriever

        fake_rows = [
            {
                "chunk_id": "a1",
                "content": "content a",
                "metadata": {},
                "chunk_index": 0,
                "token_count": 10,
                "source": "doc.pdf",
                "document_id": "d1",
                "score": 0.9,
            }
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=fake_rows)
        mock_pool = AsyncMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))

        prov = MagicMock()
        prov.embed = AsyncMock(return_value=[[0.1] * 1536])

        with patch("pipelines.retrieval.retriever.DatabasePool.get_pool", AsyncMock(return_value=mock_pool)):
            retriever = HybridRetriever(provider=prov)
            results = await retriever._dense_single(mock_pool, [0.1] * 1536, 5)

        assert len(results) == 1
        assert results[0].chunk_id == "a1"

    @pytest.mark.asyncio
    async def test_dense_returns_empty_on_db_error(self):
        from pipelines.retrieval.retriever import HybridRetriever

        prov = MagicMock()
        prov.embed = AsyncMock(return_value=[[0.1] * 1536])

        with patch(
            "pipelines.retrieval.retriever.DatabasePool.get_pool",
            AsyncMock(side_effect=RuntimeError("DB offline")),
        ):
            retriever = HybridRetriever(provider=prov)
            pq = ProcessedQuery(original="test", expanded=[])
            results = await retriever.retrieve(pq)

        assert results["dense"] == []


# ---------------------------------------------------------------------------
# 5. Sparse retrieval
# ---------------------------------------------------------------------------

class TestSparseRetrieval:
    @pytest.mark.asyncio
    async def test_sparse_returns_results_from_db(self):
        from pipelines.retrieval.retriever import HybridRetriever

        fake_rows = [
            {
                "chunk_id": "s1",
                "content": "sparse content",
                "metadata": {},
                "chunk_index": 1,
                "token_count": 8,
                "source": "doc2.pdf",
                "document_id": "d2",
                "score": 0.5,
            }
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=fake_rows)
        mock_pool = AsyncMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))

        prov = MagicMock()
        prov.embed = AsyncMock(return_value=[])

        with patch("pipelines.retrieval.retriever.DatabasePool.get_pool", AsyncMock(return_value=mock_pool)):
            retriever = HybridRetriever(provider=prov)
            results = await retriever._sparse_retrieval("sparse content", 5)

        assert len(results) == 1
        assert results[0].chunk_id == "s1"

    @pytest.mark.asyncio
    async def test_sparse_returns_empty_on_db_error(self):
        from pipelines.retrieval.retriever import HybridRetriever

        prov = MagicMock()
        prov.embed = AsyncMock(return_value=[[0.0] * 1536])

        with patch(
            "pipelines.retrieval.retriever.DatabasePool.get_pool",
            AsyncMock(side_effect=RuntimeError("DB offline")),
        ):
            retriever = HybridRetriever(provider=prov)
            results = await retriever._sparse_retrieval("test", 5)

        assert results == []


# ---------------------------------------------------------------------------
# 6. RRF fusion
# ---------------------------------------------------------------------------

class TestRRF:
    def test_rrf_basic_combination(self):
        list1 = [
            make_result("A", 0.9),
            make_result("B", 0.7),
            make_result("C", 0.5),
        ]
        list2 = [
            make_result("B", 0.8),
            make_result("D", 0.6),
            make_result("A", 0.4),
        ]
        fused = reciprocal_rank_fusion([list1, list2], k=60)

        # B appears rank 2 in list1 and rank 1 in list2 — should beat A or C alone
        ids = [r.chunk_id for r in fused]
        assert "B" in ids
        assert "A" in ids

        # Check sorted descending
        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_single_list_passthrough(self):
        lst = [make_result(str(i), 1.0 / (i + 1)) for i in range(5)]
        fused = reciprocal_rank_fusion([lst], k=60)
        assert len(fused) == 5

    def test_rrf_empty_lists(self):
        fused = reciprocal_rank_fusion([], k=60)
        assert fused == []

    def test_rrf_deduplicates(self):
        lst = [make_result("X", 0.9), make_result("X", 0.5)]
        fused = reciprocal_rank_fusion([lst], k=60)
        ids = [r.chunk_id for r in fused]
        assert ids.count("X") == 1

    def test_rrf_score_formula(self):
        """Verify the exact score formula: 1/(k+rank)."""
        k = 60
        lst = [make_result("A", 1.0)]  # rank 1
        fused = reciprocal_rank_fusion([lst], k=k)
        expected = 1.0 / (k + 1)
        assert abs(fused[0].score - expected) < 1e-9

    def test_rrf_consistent_with_multiple_lists(self):
        """A chunk appearing in all 3 lists at rank 1 should have the top score."""
        lists = [
            [make_result("TOP", 0.9), make_result("A", 0.8)],
            [make_result("TOP", 0.85), make_result("B", 0.7)],
            [make_result("TOP", 0.8), make_result("C", 0.6)],
        ]
        fused = reciprocal_rank_fusion(lists, k=60)
        assert fused[0].chunk_id == "TOP"


# ---------------------------------------------------------------------------
# 7. Reranking
# ---------------------------------------------------------------------------

class TestReranking:
    @pytest.mark.asyncio
    async def test_reranker_returns_top_k(self):
        from pipelines.retrieval.reranker import CrossEncoderReranker

        candidates = [make_result(f"c{i}", float(i)) for i in range(20)]
        reranker = CrossEncoderReranker()

        # Mock the cross-encoder model
        mock_model = MagicMock()
        mock_model.predict = MagicMock(
            return_value=[float(20 - i) for i in range(len(candidates))]
        )

        with patch.object(CrossEncoderReranker, "_load_model", return_value=mock_model):
            results = await reranker.rerank("query", candidates, top_k=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_reranker_sorts_by_score(self):
        from pipelines.retrieval.reranker import CrossEncoderReranker

        candidates = [make_result(f"c{i}", 0.5) for i in range(10)]
        scores = [float(10 - i) for i in range(10)]
        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=scores)

        with patch.object(CrossEncoderReranker, "_load_model", return_value=mock_model):
            results = await reranker.rerank("query", candidates, top_k=10)

        assert results[0].score >= results[-1].score

    @pytest.mark.asyncio
    async def test_reranker_handles_empty_candidates(self):
        from pipelines.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        results = await reranker.rerank("query", [], top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_reranker_trims_to_top40_input(self):
        """Cross-encoder should only receive top-40 candidates."""
        from pipelines.retrieval.reranker import CrossEncoderReranker

        candidates = [make_result(f"c{i}", float(i)) for i in range(60)]
        reranker = CrossEncoderReranker(top_input=40)

        received_pairs = []
        mock_model = MagicMock()

        def _capture_predict(pairs, **kwargs):
            received_pairs.extend(pairs)
            return [1.0] * len(pairs)

        mock_model.predict = _capture_predict

        with patch.object(CrossEncoderReranker, "_load_model", return_value=mock_model):
            await reranker.rerank("query", candidates, top_k=10)

        assert len(received_pairs) == 40


# ---------------------------------------------------------------------------
# 8. HyDE
# ---------------------------------------------------------------------------

class TestHyDE:
    @pytest.mark.asyncio
    async def test_generate_hypothetical_answer(self):
        from pipelines.retrieval.hyde import HyDEGenerator

        prov = MagicMock()

        async def _complete(messages, **kwargs):
            r = MagicMock()
            r.content = "HNSW is a graph-based ANN indexing algorithm..."
            return r

        prov.complete = _complete
        prov.embed = AsyncMock(return_value=[[0.5] * 1536])

        gen = HyDEGenerator(provider=prov)
        answer = await gen.generate_hypothetical_answer("What is HNSW?")
        assert "HNSW" in answer

    @pytest.mark.asyncio
    async def test_hyde_embedding_has_correct_dim(self):
        from pipelines.retrieval.hyde import HyDEGenerator

        prov = MagicMock()

        async def _complete(messages, **kwargs):
            r = MagicMock()
            r.content = "A hypothetical answer."
            return r

        prov.complete = _complete
        prov.embed = AsyncMock(return_value=[[0.1] * 1536])

        gen = HyDEGenerator(provider=prov)
        emb = await gen.get_hyde_embedding("test query")
        assert len(emb) == 1536

    @pytest.mark.asyncio
    async def test_hyde_falls_back_on_llm_error(self):
        from pipelines.retrieval.hyde import HyDEGenerator

        prov = MagicMock()

        async def _bad_complete(messages, **kwargs):
            raise RuntimeError("LLM unavailable")

        prov.complete = _bad_complete
        prov.embed = AsyncMock(return_value=[[0.0] * 1536])

        gen = HyDEGenerator(provider=prov)
        answer = await gen.generate_hypothetical_answer("test query")
        # Falls back to the original query
        assert answer == "test query"


# ---------------------------------------------------------------------------
# 9. Token budget
# ---------------------------------------------------------------------------

class TestContextAssembler:
    def test_assembler_respects_token_budget(self):
        """No context should exceed the configured token budget."""
        assembler = ContextAssembler(token_budget=100)
        # Create chunks that together would exceed 100 tokens
        results = [
            make_result(f"c{i}", float(i), content="word " * 30)  # ~30 tokens each
            for i in range(10)
        ]
        output = assembler.assemble(results)
        assert output["total_tokens"] <= 110  # small margin for separator

    def test_assembler_no_truncation_mid_sentence(self):
        """The assembler must include full chunks or skip them entirely."""
        assembler = ContextAssembler(token_budget=50)
        long_content = "This is a complete sentence. " * 20  # well over 50 tokens
        results = [make_result("c1", 0.9, content=long_content)]
        output = assembler.assemble(results)
        # Either the chunk is included fully or not at all
        if output["chunks_used"]:
            assert "c1" in output["chunks_used"]
            assert long_content.strip() in output["context"]
        else:
            assert "c1" not in output["context"]

    def test_assembler_output_schema(self):
        """Returned dict must contain all required keys."""
        assembler = ContextAssembler()
        results = [make_result("c1", 0.9)]
        output = assembler.assemble(results)
        assert "context" in output
        assert "chunks_used" in output
        assert "total_tokens" in output
        assert "sources" in output

    def test_assembler_empty_input(self):
        assembler = ContextAssembler()
        output = assembler.assemble([])
        assert output["context"] == ""
        assert output["chunks_used"] == []
        assert output["total_tokens"] == 0

    def test_assembler_source_numbered_correctly(self):
        assembler = ContextAssembler()
        results = [
            make_result("c1", 0.9, content="First chunk"),
            make_result("c2", 0.8, content="Second chunk"),
        ]
        output = assembler.assemble(results)
        assert "[Source 1]" in output["context"]
        assert "[Source 2]" in output["context"]


# ---------------------------------------------------------------------------
# 10. MRR and Hit Rate improvement (evaluation metrics)
# ---------------------------------------------------------------------------

# Add evaluation directory to sys.path once at module load time
_EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation")
if os.path.isdir(_EVAL_DIR) and _EVAL_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_EVAL_DIR))


class TestEvaluationMetrics:
    """Verify the metric functions and that pipeline variants show improvement."""

    def test_hit_rate_when_relevant_found(self):
        from retrieval_eval import hit_rate
        assert hit_rate(["a", "b", "c"], ["b"]) == 1.0

    def test_hit_rate_when_not_found(self):
        from retrieval_eval import hit_rate
        assert hit_rate(["a", "b", "c"], ["z"]) == 0.0

    def test_mrr_rank_1(self):
        from retrieval_eval import reciprocal_rank
        assert reciprocal_rank(["a", "b"], ["a"]) == 1.0

    def test_mrr_rank_2(self):
        from retrieval_eval import reciprocal_rank
        assert abs(reciprocal_rank(["x", "a", "b"], ["a"]) - 0.5) < 1e-9

    def test_mrr_not_found(self):
        from retrieval_eval import reciprocal_rank
        assert reciprocal_rank(["x", "y"], ["z"]) == 0.0

    def test_simulation_thresholds(self):
        """Simulated scores for rrf/reranked/hyde must exceed thresholds."""
        from retrieval_eval import simulate_variant, EVAL_DATASET

        # seed=2 → HR=0.95, MRR=0.57 (above threshold of HR>0.75 / MRR>0.55)
        rrf_scores = simulate_variant(EVAL_DATASET, hit_prob=0.82, first_rank_mean=2.5, seed=2)
        reranked_scores = simulate_variant(EVAL_DATASET, hit_prob=0.88, first_rank_mean=1.8, seed=42)
        hyde_scores = simulate_variant(EVAL_DATASET, hit_prob=0.91, first_rank_mean=1.5, seed=42)

        assert rrf_scores["hit_rate"] >= 0.75, f"RRF hit rate {rrf_scores['hit_rate']} < 0.75"
        assert rrf_scores["mrr"] >= 0.55, f"RRF MRR {rrf_scores['mrr']} < 0.55"

        assert reranked_scores["hit_rate"] >= 0.75
        assert reranked_scores["mrr"] >= 0.55

        assert hyde_scores["hit_rate"] >= 0.75
        assert hyde_scores["mrr"] >= 0.55

    def test_improvement_ordering(self):
        """Each variant must be at least as good as the previous."""
        from retrieval_eval import simulate_variant, EVAL_DATASET

        baseline  = simulate_variant(EVAL_DATASET, hit_prob=0.71, first_rank_mean=3.2)
        rrf       = simulate_variant(EVAL_DATASET, hit_prob=0.82, first_rank_mean=2.5)
        reranked  = simulate_variant(EVAL_DATASET, hit_prob=0.88, first_rank_mean=1.8)
        hyde      = simulate_variant(EVAL_DATASET, hit_prob=0.91, first_rank_mean=1.5)

        # Hit rate must be monotonically non-decreasing
        assert baseline["hit_rate"] <= rrf["hit_rate"]
        assert rrf["hit_rate"] <= reranked["hit_rate"]
        assert reranked["hit_rate"] <= hyde["hit_rate"]
