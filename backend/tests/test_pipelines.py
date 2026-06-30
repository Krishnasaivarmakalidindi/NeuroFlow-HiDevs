import pytest
import uuid
import json
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from pydantic import ValidationError

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from models.pipeline import PipelineConfig
from services.pipeline_manager import PipelineManager
from services.pipeline_analytics import compute_percentile

client = TestClient(app)

VALID_CONFIG = {
    "name": "legal",
    "description": "Legal pipeline",
    "ingestion": {
        "chunking_strategy": "fixed",
        "chunk_size_tokens": 500,
        "chunk_overlap_tokens": 50,
        "extractors_enabled": ["pdf"]
    },
    "retrieval": {
        "dense_k": 10,
        "sparse_k": 10,
        "reranker": "cross-encoder",
        "top_k_after_rerank": 5,
        "query_expansion": True,
        "metadata_filters_enabled": False
    },
    "generation": {
        "model_routing": {"default": "llama-3.3-70b-versatile"},
        "max_context_tokens": 4000,
        "temperature": 0.7,
        "system_prompt_variant": "factual"
    },
    "evaluation": {
        "auto_evaluate": True,
        "training_threshold": 0.8
    }
}

# ---------------------------------------------------------------------------
# 1. Schema Validation Tests
# ---------------------------------------------------------------------------

def test_pipeline_config_validation():
    # Valid config should pass
    config = PipelineConfig(**VALID_CONFIG)
    assert config.name == "legal"
    assert config.retrieval.dense_k == 10

def test_pipeline_config_unknown_key_rejection():
    # Unknown key should fail validation (extra: forbid)
    invalid = VALID_CONFIG.copy()
    invalid["unknown_key"] = "value"
    with pytest.raises(ValidationError):
        PipelineConfig(**invalid)

    invalid_sub = VALID_CONFIG.copy()
    invalid_sub["retrieval"] = VALID_CONFIG["retrieval"].copy()
    invalid_sub["retrieval"]["extra_key"] = 123
    with pytest.raises(ValidationError):
        PipelineConfig(**invalid_sub)

# ---------------------------------------------------------------------------
# 2. Pipeline Manager / CRUD / Versioning Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_pipeline():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    # Mock insert and select query response
    created_uuid = uuid.uuid4()
    mock_db_conn.fetchrow = AsyncMock(return_value={
        "id": created_uuid,
        "name": "legal-v1",
        "version": 1,
        "config": json.dumps(VALID_CONFIG),
        "status": "active",
        "parent_pipeline_id": None,
        "created_at": None
    })

    with patch("services.pipeline_manager.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)):
        manager = PipelineManager()
        res = await manager.create_pipeline(PipelineConfig(**VALID_CONFIG))
        
        assert res["id"] == created_uuid
        assert res["name"] == "legal-v1"
        assert res["version"] == 1
        assert res["config"]["name"] == "legal"

@pytest.mark.asyncio
async def test_update_pipeline_versioning():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    parent_uuid = uuid.uuid4()
    # Mock parent fetch
    mock_db_conn.fetchrow = AsyncMock(side_effect=[
        # Parent fetch
        {
            "name": "legal-v1",
            "version": 1,
            "status": "active"
        },
        # New row fetch
        {
            "id": uuid.uuid4(),
            "name": "legal-v2",
            "version": 2,
            "config": json.dumps(VALID_CONFIG),
            "status": "active",
            "parent_pipeline_id": parent_uuid,
            "created_at": None
        }
    ])

    with patch("services.pipeline_manager.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)):
        manager = PipelineManager()
        res = await manager.update_pipeline(str(parent_uuid), PipelineConfig(**VALID_CONFIG))
        
        assert res["name"] == "legal-v2"
        assert res["version"] == 2
        assert res["parent_pipeline_id"] == parent_uuid

@pytest.mark.asyncio
async def test_soft_delete():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    mock_db_conn.execute = AsyncMock(return_value="UPDATE 1")

    with patch("services.pipeline_manager.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)):
        manager = PipelineManager()
        success = await manager.delete_pipeline(str(uuid.uuid4()))
        assert success is True
        
        sql_calls = [args[0] for args, kwargs in mock_db_conn.execute.call_args_list]
        assert any("UPDATE pipelines SET status = $1" in call for call in sql_calls)

# ---------------------------------------------------------------------------
# 3. HTTP API CRUD Tests
# ---------------------------------------------------------------------------

def test_api_crud_endpoints():
    mock_db_pool = MagicMock()
    
    with patch("services.pipeline_manager.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)):
        # Test POST
        with patch("services.pipeline_manager.PipelineManager.create_pipeline", AsyncMock(return_value={"id": "uuid1", "name": "legal-v1"})):
            response = client.post("/pipelines", json=VALID_CONFIG)
            assert response.status_code == 201
            assert response.json()["id"] == "uuid1"

        # Test GET List
        with patch("services.pipeline_manager.PipelineManager.list_pipelines", AsyncMock(return_value=[{"id": "uuid1", "name": "legal-v1"}])):
            response = client.get("/pipelines")
            assert response.status_code == 200
            assert len(response.json()) == 1

        # Test GET single
        with patch("services.pipeline_manager.PipelineManager.get_pipeline", AsyncMock(return_value={"id": "uuid1", "name": "legal-v1", "config": {}})):
            response = client.get("/pipelines/uuid1")
            assert response.status_code == 200
            assert response.json()["id"] == "uuid1"

        # Test PATCH
        with patch("services.pipeline_manager.PipelineManager.update_pipeline", AsyncMock(return_value={"id": "uuid2", "name": "legal-v2", "version": 2})):
            response = client.patch("/pipelines/uuid1", json=VALID_CONFIG)
            assert response.status_code == 200
            assert response.json()["version"] == 2

        # Test DELETE (soft-delete)
        with patch("services.pipeline_manager.PipelineManager.delete_pipeline", AsyncMock(return_value=True)):
            response = client.delete("/pipelines/uuid1")
            assert response.status_code == 200
            assert response.json()["status"] == "success"

# ---------------------------------------------------------------------------
# 4. A/B Comparison Endpoint Tests
# ---------------------------------------------------------------------------

def test_ab_comparison():
    mock_manager = MagicMock()
    mock_manager.get_pipeline = AsyncMock(return_value={"id": "uuid", "name": "pipeline"})

    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    # Mock DB query for run outputs
    mock_db_conn.fetchrow = AsyncMock(return_value={
        "retrieved_chunk_ids": [uuid.uuid4(), uuid.uuid4()],
        "metadata": json.dumps({"retrieval_latency_ms": 12.0})
    })

    # Mock generator and evaluation judge
    mock_generation_result = MagicMock(answer="Generation output", citations=[])
    
    with patch("api.compare.manager", mock_manager), \
         patch("api.compare.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)), \
         patch("api.compare.rag_generator.generate", AsyncMock(return_value=mock_generation_result)), \
         patch("api.compare.judge.evaluate_run", AsyncMock(return_value={"overall": 0.85})):
         
        response = client.post("/pipelines/compare", json={
            "query": "Is attention parallel?",
            "pipeline_a_id": str(uuid.uuid4()),
            "pipeline_b_id": str(uuid.uuid4())
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "Is attention parallel?"
        assert data["pipeline_a"]["generation"] == "Generation output"
        assert data["pipeline_a"]["eval_score"] == 0.85
        assert data["pipeline_a"]["chunks_used"] == 2

# ---------------------------------------------------------------------------
# 5. Percentiles & Analytics Tests
# ---------------------------------------------------------------------------

def test_compute_percentile():
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert compute_percentile(data, 50) == 30.0
    assert compute_percentile(data, 100) == 50.0
    assert compute_percentile(data, 0) == 10.0

def test_analytics_endpoint():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    # Mock fetch returns for runs, evals, and queries per day
    mock_db_conn.fetch = AsyncMock(side_effect=[
        # Runs
        [
            {"latency_ms": 100, "input_tokens": 100, "output_tokens": 200, "metadata": json.dumps({"retrieval_latency_ms": 15.0}), "model_used": "gpt-4o-mini"}
        ],
        # Evals
        [
            {"faithfulness": 0.8, "answer_relevance": 0.9, "context_precision": 0.7, "context_recall": 0.6, "overall_score": 0.75}
        ],
        # Queries per day
        [
            {"day": "2026-06-29", "count": 5}
        ]
    ])

    with patch("services.pipeline_analytics.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)):
        response = client.get(f"/pipelines/{str(uuid.uuid4())}/analytics")
        assert response.status_code == 200
        data = response.json()
        assert data["retrieval_latency"]["p50"] == 15.0
        assert data["evaluation"]["overall"] == 0.75
        assert len(data["queries_per_day"]) == 1

# ---------------------------------------------------------------------------
# 6. Suggestions & Optimizer Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_optimizer_suggestions():
    from services.pipeline_optimizer import PipelineOptimizer
    
    mock_manager = MagicMock()
    mock_manager.get_pipeline = AsyncMock(return_value={
        "config": {
            "retrieval": {
                "dense_k": 10,
                "top_k_after_rerank": 8,
                "query_expansion": False
            },
            "generation": {
                "temperature": 0.7
            }
        }
    })

    # Return low metrics across all dimensions to trigger all suggestion rules
    mock_analytics_value = {
        "evaluation": {
            "faithfulness": 0.5,      # < 0.7
            "answer_relevance": 0.5,  # < 0.7
            "context_precision": 0.5, # < 0.6
            "context_recall": 0.5,    # < 0.6
            "overall": 0.5
        },
        "retrieval_latency": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
        "generation_latency": 0.0,
        "cost_per_query": 0.0,
        "queries_per_day": []
    }

    # 1. Test the optimizer class logic directly
    optimizer_instance = PipelineOptimizer()
    optimizer_instance.manager = mock_manager
    optimizer_instance.analytics = MagicMock()
    optimizer_instance.analytics.get_analytics = AsyncMock(return_value=mock_analytics_value)
    
    suggestions = await optimizer_instance.get_suggestions(str(uuid.uuid4()))
    
    # Verify suggestions exist for each low score
    metrics_triggered = {s["metric"] for s in suggestions}
    assert "context_precision" in metrics_triggered
    assert "context_recall" in metrics_triggered
    assert "faithfulness" in metrics_triggered
    assert "answer_relevance" in metrics_triggered
    
    precision_sug = next(s for s in suggestions if s["metric"] == "context_precision")
    assert "reduce top_k_after_rerank from 8 to 5" in precision_sug["suggestion"]

    # 2. Test the API endpoint by patching the registered optimizer service
    mock_sugs = [
        {
            "metric": "context_precision",
            "problem": "too many retrieved chunks",
            "suggestion": "reduce top_k_after_rerank from 8 to 5"
        }
    ]
    with patch("api.pipelines.optimizer.get_suggestions", AsyncMock(return_value=mock_sugs)):
        response = client.get(f"/pipelines/{str(uuid.uuid4())}/suggestions")
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["metric"] == "context_precision"

# ---------------------------------------------------------------------------
# 7. Run History Tests (Pagination, Sorting, Filtering)
# ---------------------------------------------------------------------------

def test_run_history_queries():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    # Mock runs fetch
    mock_db_conn.fetch = AsyncMock(return_value=[
        {
            "run_id": uuid.uuid4(),
            "latency": 150,
            "tokens": 450,
            "faithfulness": 0.9,
            "answer_relevance": 0.8,
            "context_precision": 0.9,
            "context_recall": 0.8,
            "overall_score": 0.85
        }
    ])

    with patch("api.pipelines.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)):
        response = client.get(
            f"/pipelines/{str(uuid.uuid4())}/runs",
            params={
                "page": 2,
                "limit": 5,
                "sort_by": "latency_ms",
                "order": "asc",
                "status": "complete",
                "model_used": "llama-3.3-70b-versatile"
            }
        )
        
        assert response.status_code == 200
        assert len(response.json()) == 1
        
        # Verify SQL query matches the pagination parameters and status/model filters
        sql_query = mock_db_conn.fetch.call_args[0][0]
        assert "ORDER BY r.latency_ms ASC" in sql_query
        assert "AND r.status = $2" in sql_query
        assert "AND r.model_used = $3" in sql_query
        assert "LIMIT $4 OFFSET $5" in sql_query
        
        # Verify bounds of OFFSET: (page - 1) * limit = (2 - 1) * 5 = 5
        sql_params = mock_db_conn.fetch.call_args[0][1:]
        assert sql_params[-2] == 5 # limit
        assert sql_params[-1] == 5 # offset
