import pytest
import asyncio
import json
import uuid
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from main import app
from providers.base import GenerationResult
from evaluation.metrics import (
    evaluate_faithfulness,
    evaluate_answer_relevance,
    evaluate_context_precision,
    evaluate_context_recall
)
from evaluation.judge import EvaluationJudge
from evaluation.self_consistency import SelfConsistencyJudge
from evaluation.calibrate import run_calibration

# Setup Test Client
client = TestClient(app)

@pytest.fixture
def mock_client():
    with patch("providers.client.NeuroFlowClient.chat", new_callable=AsyncMock) as mock_chat, \
         patch("providers.client.NeuroFlowClient.embed", new_callable=AsyncMock) as mock_embed:
        class MockWrapper:
            chat = mock_chat
            embed = mock_embed
        # Reset side_effects and return_values before every test
        mock_chat.side_effect = None
        mock_chat.return_value = MagicMock()
        mock_embed.side_effect = None
        mock_embed.return_value = []
        yield MockWrapper

# ---------------------------------------------------------------------------
# 1. Faithfulness Metric Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_faithfulness_yes(mock_client):
    # Step 1: Extract claims -> return JSON list
    mock_client.chat.side_effect = [
        MagicMock(content='["Transformers use self-attention", "HNSW uses graph traversal"]'), # Claims
        MagicMock(content='yes'), # Claim 1 verify
        MagicMock(content='yes')  # Claim 2 verify
    ]
    
    score = await evaluate_faithfulness("query", "answer", "context")
    assert score == 1.0

@pytest.mark.asyncio
async def test_evaluate_faithfulness_partial_and_no(mock_client):
    # Step 1: Extract claims -> return JSON list
    mock_client.chat.side_effect = [
        MagicMock(content='["Claim 1", "Claim 2", "Claim 3"]'), # Claims
        MagicMock(content='yes'), # Claim 1
        MagicMock(content='partial'), # Claim 2
        MagicMock(content='no')  # Claim 3
    ]
    
    score = await evaluate_faithfulness("query", "answer", "context")
    # yes (1.0) + partial (0.5) + no (0.0) = 1.5 / 3 = 0.5
    assert score == 0.5

@pytest.mark.asyncio
async def test_evaluate_faithfulness_empty_context(mock_client):
    score = await evaluate_faithfulness("query", "answer", "")
    assert score == 0.0

# ---------------------------------------------------------------------------
# 2. Answer Relevance Metric Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_answer_relevance(mock_client):
    # Step 1: Generate questions
    mock_client.chat.side_effect = None
    mock_client.chat.return_value = MagicMock(content="1. What does attention do?\n2. How to traverse HNSW?\n3. What is search?")
    
    # Step 2: Embed (query + 3 questions) -> returns 4 vectors
    # We return identical vectors to yield cosine similarity = 1.0
    mock_client.embed.return_value = [
        [1.0, 0.0], # query
        [1.0, 0.0], # q1
        [1.0, 0.0], # q2
        [1.0, 0.0]  # q3
    ]
    
    score = await evaluate_answer_relevance("What is attention?", "Answer")
    assert score == 1.0

# ---------------------------------------------------------------------------
# 3. Context Precision Metric Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_context_precision(mock_client):
    # Step 1: Was chunk useful -> chunk 1: yes, chunk 2: no
    mock_client.chat.side_effect = [
        MagicMock(content="yes"),
        MagicMock(content="no")
    ]
    
    score = await evaluate_context_precision("query", ["chunk 1", "chunk 2"], "answer")
    # usefulness = [1.0, 0.0]
    # score = (1.0 * (1/1) + 0.0 * (1/2)) / (1/1 + 1/2) = 1.0 / 1.5 = 0.6667
    assert abs(score - 0.6667) < 0.001

# ---------------------------------------------------------------------------
# 4. Context Recall Metric Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_context_recall(mock_client):
    # Split answer: "Sentence one. Sentence two."
    # Step 1: check attribution -> sentence 1: yes, sentence 2: no
    mock_client.chat.side_effect = [
        MagicMock(content="yes"),
        MagicMock(content="no")
    ]
    
    score = await evaluate_context_recall("query", ["chunk 1"], "Sentence one. Sentence two.")
    # attributable = 1 / 2 = 0.5
    assert score == 0.5

# ---------------------------------------------------------------------------
# 5. Judge Parallel Execution & SQL Persistence Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluation_judge_persistence_low_score():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    # Mock DB select for run details
    mock_db_conn.fetchrow = AsyncMock(return_value={
        "query": "What is attention?",
        "generation": "Transformers use self-attention.",
        "retrieved_chunk_ids": [uuid.uuid4()],
        "metadata": "{}"
    })
    
    # Mock chunks retrieve
    mock_db_conn.fetch = AsyncMock(return_value=[{"content": "Context info about attention."}])

    # We mock metrics to return scores that yield overall < 0.8
    # overall = 0.35*0.5 + 0.30*0.5 + 0.20*0.5 + 0.15*0.5 = 0.5
    with patch("evaluation.judge.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)), \
         patch("evaluation.judge.evaluate_faithfulness", AsyncMock(return_value=0.5)), \
         patch("evaluation.judge.evaluate_answer_relevance", AsyncMock(return_value=0.5)), \
         patch("evaluation.judge.evaluate_context_precision", AsyncMock(return_value=0.5)), \
         patch("evaluation.judge.evaluate_context_recall", AsyncMock(return_value=0.5)):
         
        judge = EvaluationJudge()
        judge.client.router.route = AsyncMock(return_value="gpt-4o")
        res = await judge.evaluate_run(str(uuid.uuid4()))
        
        assert res["overall"] == pytest.approx(0.5)
        assert res["judge_model"] == "gpt-4o"
        
        # Verify it inserted into evaluations table
        # Check that connection.execute was called to insert into evaluations
        # But training_pairs insert should NOT be called (overall = 0.5 <= 0.8)
        sql_calls = [args[0] for args, kwargs in mock_db_conn.execute.call_args_list]
        
        # Must have evaluations INSERT
        assert any("INSERT INTO evaluations" in call for call in sql_calls)
        # Must NOT have training_pairs INSERT
        assert not any("INSERT INTO training_pairs" in call for call in sql_calls)

@pytest.mark.asyncio
async def test_evaluation_judge_persistence_high_score():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    mock_db_conn.fetchrow = AsyncMock(return_value={
        "query": "What is attention?",
        "generation": "Transformers use self-attention.",
        "retrieved_chunk_ids": [uuid.uuid4()],
        "metadata": json.dumps({"prompt": "System prompt info."})
    })
    mock_db_conn.fetch = AsyncMock(return_value=[{"content": "Context info about attention."}])

    # We mock metrics to return scores that yield overall > 0.8 (e.g. 1.0)
    with patch("evaluation.judge.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)), \
         patch("evaluation.judge.evaluate_faithfulness", AsyncMock(return_value=1.0)), \
         patch("evaluation.judge.evaluate_answer_relevance", AsyncMock(return_value=1.0)), \
         patch("evaluation.judge.evaluate_context_precision", AsyncMock(return_value=1.0)), \
         patch("evaluation.judge.evaluate_context_recall", AsyncMock(return_value=1.0)):
         
        judge = EvaluationJudge()
        judge.client.router.route = AsyncMock(return_value="gpt-4o")
        res = await judge.evaluate_run(str(uuid.uuid4()))
        
        assert res["overall"] == pytest.approx(1.0)
        
        # Verify it inserted into both evaluations and training_pairs table
        sql_calls = [args[0] for args, kwargs in mock_db_conn.execute.call_args_list]
        assert any("INSERT INTO evaluations" in call for call in sql_calls)
        assert any("INSERT INTO training_pairs" in call for call in sql_calls)

# ---------------------------------------------------------------------------
# 6. Self Consistency & High Variance Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_self_consistency_low_variance():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    mock_db_conn.fetchrow = AsyncMock(return_value={
        "query": "query",
        "generation": "generation",
        "retrieved_chunk_ids": []
    })

    # Return same score for all iterations (std = 0)
    with patch("evaluation.self_consistency.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)), \
         patch("evaluation.self_consistency.evaluate_faithfulness", AsyncMock(return_value=0.9)), \
         patch("evaluation.self_consistency.evaluate_answer_relevance", AsyncMock(return_value=0.9)), \
         patch("evaluation.self_consistency.evaluate_context_precision", AsyncMock(return_value=0.9)), \
         patch("evaluation.self_consistency.evaluate_context_recall", AsyncMock(return_value=0.9)):
         
        judge = SelfConsistencyJudge()
        res = await judge.evaluate_consistency(str(uuid.uuid4()))
        
        assert res["mean"] == 0.9
        assert res["std"] == 0.0
        assert res["high_variance"] is False

@pytest.mark.asyncio
async def test_self_consistency_high_variance():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    mock_db_conn.fetchrow = AsyncMock(return_value={
        "query": "query",
        "generation": "generation",
        "retrieved_chunk_ids": []
    })

    # Return highly varying scores (e.g. [1.0, 0.1, 0.0])
    with patch("evaluation.self_consistency.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)), \
         patch("evaluation.self_consistency.evaluate_faithfulness", AsyncMock(side_effect=[1.0, 0.1, 0.0])), \
         patch("evaluation.self_consistency.evaluate_answer_relevance", AsyncMock(side_effect=[1.0, 0.1, 0.0])), \
         patch("evaluation.self_consistency.evaluate_context_precision", AsyncMock(side_effect=[1.0, 0.1, 0.0])), \
         patch("evaluation.self_consistency.evaluate_context_recall", AsyncMock(side_effect=[1.0, 0.1, 0.0])):
         
        judge = SelfConsistencyJudge()
        res = await judge.evaluate_consistency(str(uuid.uuid4()))
        
        # Mean should be some middle value, std should be > 0.2
        assert res["std"] > 0.2
        assert res["high_variance"] is True

# ---------------------------------------------------------------------------
# 7. Human Feedback Endpoint Tests
# ---------------------------------------------------------------------------

def test_human_feedback_low_diff():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    # Automated score = 0.9, human rating = 5 (rating / 5 = 1.0). Diff = 0.1 <= 0.3
    mock_db_conn.fetchrow = AsyncMock(return_value={
        "overall_score": 0.9,
        "metadata": json.dumps({})
    })

    with patch("backend.api.feedback.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)):
        run_id = str(uuid.uuid4())
        response = client.patch(f"/runs/{run_id}/rating", json={"rating": 5})
        
        assert response.status_code == 200
        data = response.json()
        assert data["difference"] == 0.1
        assert data["calibration_needed"] is False

        # Verify DB update query was executed
        sql_calls = [args[0] for args, kwargs in mock_db_conn.execute.call_args_list]
        assert any("UPDATE evaluations" in call for call in sql_calls)

def test_human_feedback_high_diff():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    # Automated score = 0.3, human rating = 5 (rating / 5 = 1.0). Diff = 0.7 > 0.3
    mock_db_conn.fetchrow = AsyncMock(return_value={
        "overall_score": 0.3,
        "metadata": "{}"
    })

    with patch("backend.api.feedback.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)):
        run_id = str(uuid.uuid4())
        response = client.patch(f"/runs/{run_id}/rating", json={"rating": 5})
        
        assert response.status_code == 200
        data = response.json()
        assert data["difference"] == 0.7
        assert data["calibration_needed"] is True

# ---------------------------------------------------------------------------
# 8. Calibration Metric Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_calibration():
    res = await run_calibration(simulate=True)
    assert res["pearson"] > 0.85
    assert res["status"] == "PASS"
