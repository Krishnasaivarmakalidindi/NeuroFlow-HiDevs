import pytest
import uuid
import json
import re
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from pipelines.finetuning.extractor import TrainingDataExtractor
from pipelines.finetuning.validator import TrainingDataValidator
from pipelines.finetuning.tracker import FineTuneTracker
from pipelines.finetuning.registry import ModelRegistry
from pipelines.finetuning.job_manager import FineTuneJobManager
from pipelines.finetuning.dpo import DPOExtractor
from providers.router import ModelRouter, RoutingCriteria

client = TestClient(app)

# Mocked DB rows matching query output
MOCK_PAIRS = [
    {
        "id": str(uuid.uuid4()),
        "run_id": str(uuid.uuid4()),
        "quality_score": 0.89,
        "system_prompt": "You are a legal assistant.",
        "user_message": "What is clean hands doctrine?",
        "assistant_message": "The clean hands doctrine is an equitable defense. [Source 1] It requires that the party seeking equity must have acted fairly. This is a very long sentence that is added to increase the length of the assistant message to satisfy the validation length criteria of 50 tokens. We repeat this information to make it sufficiently long and informative. It has more than fifty tokens.",
        "user_rating": 5,
        "faithfulness": 0.95
    },
    {
        "id": str(uuid.uuid4()),
        "run_id": str(uuid.uuid4()),
        "quality_score": 0.85,
        "system_prompt": "You are a legal assistant.",
        "user_message": "Define tort law.",
        "assistant_message": "Tort law addresses civil wrongs. [Source 2] It allows individuals to claim damages for harm suffered. This is a very long sentence that is added to increase the length of the assistant message to satisfy the validation length criteria of 50 tokens. We repeat this information to make it sufficiently long and informative. It has more than fifty tokens.",
        "user_rating": None,
        "faithfulness": 0.9
    }
]

# ---------------------------------------------------------------------------
# 1. Extraction & Message Formatting Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_training_data_extraction():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    # Mock DB query results
    mock_db_conn.fetch = AsyncMock(return_value=[
        {
            "id": uuid.UUID(p["id"]),
            "run_id": uuid.UUID(p["run_id"]),
            "quality_score": p["quality_score"],
            "system_prompt": p["system_prompt"],
            "user_message": p["user_message"],
            "assistant_message": p["assistant_message"],
            "user_rating": p["user_rating"],
            "faithfulness": p["faithfulness"]
        } for p in MOCK_PAIRS
    ])

    with patch("pipelines.finetuning.extractor.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)):
        extractor = TrainingDataExtractor()
        pairs = await extractor.extract_pairs()
        
        assert len(pairs) == 2
        assert pairs[0]["quality_score"] == 0.89
        
        # Test formatting
        formatted = extractor.format_as_chat_messages(pairs[0])
        assert len(formatted["messages"]) == 3
        assert formatted["messages"][0]["role"] == "system"
        assert formatted["messages"][0]["content"] == "You are a legal assistant."
        assert formatted["messages"][1]["role"] == "user"
        assert formatted["messages"][2]["role"] == "assistant"

# ---------------------------------------------------------------------------
# 2. Validation & Filter Tests
# ---------------------------------------------------------------------------

def test_validator_valid_pair():
    validator = TrainingDataValidator()
    pair = MOCK_PAIRS[0].copy()
    
    is_valid, reason = validator.validate_pair(pair)
    assert is_valid is True
    assert reason == "valid"

def test_validator_assistant_length():
    validator = TrainingDataValidator()
    
    # 1. Too short (< 50 tokens approximation)
    short_pair = MOCK_PAIRS[0].copy()
    short_pair["assistant_message"] = "Short message. [Source 1]"
    is_valid, reason = validator.validate_pair(short_pair)
    assert is_valid is False
    assert "assistant_length" in reason

    # 2. Too long (> 2000 tokens)
    long_pair = MOCK_PAIRS[0].copy()
    long_pair["assistant_message"] = "word " * 2050 + " [Source 1]"
    is_valid, reason = validator.validate_pair(long_pair)
    assert is_valid is False
    assert "assistant_length" in reason

def test_validator_missing_citation():
    validator = TrainingDataValidator()
    
    nocite_pair = MOCK_PAIRS[0].copy()
    nocite_pair["assistant_message"] = "The clean hands doctrine is an equitable defense. It requires that the party seeking equity acts fairly. This is a very long message designed to bypass the 50 token validation check by adding a lot of unnecessary text to this test string so that we can test other validation rules without failing the length check."
    is_valid, reason = validator.validate_pair(nocite_pair)
    assert is_valid is False
    assert reason == "missing_citations"

def test_validator_faithfulness_filter():
    validator = TrainingDataValidator()
    
    # 1. Low faithfulness (<= 0.8)
    low_faith = MOCK_PAIRS[0].copy()
    low_faith["faithfulness"] = 0.75
    is_valid, reason = validator.validate_pair(low_faith)
    assert is_valid is False
    assert "low_faithfulness" in reason

    # 2. Missing faithfulness (None)
    missing_faith = MOCK_PAIRS[0].copy()
    missing_faith["faithfulness"] = None
    is_valid, reason = validator.validate_pair(missing_faith)
    assert is_valid is False
    assert "low_faithfulness" in reason

def test_validator_pii_detection():
    validator = TrainingDataValidator()
    
    # 1. Email in user message
    email_pair = MOCK_PAIRS[0].copy()
    email_pair["user_message"] = "Contact me at alice@example.com."
    is_valid, reason = validator.validate_pair(email_pair)
    assert is_valid is False
    assert "pii_detected_in_user_message" in reason

    # 2. Phone number in assistant message
    phone_pair = MOCK_PAIRS[0].copy()
    phone_pair["assistant_message"] = "Call us at +1-8005550199 for clean hands defense. [Source 1] This is a very long message designed to bypass the 50 token validation check by adding a lot of unnecessary text to this test string so that we can test other validation rules without failing the length check."
    is_valid, reason = validator.validate_pair(phone_pair)
    assert is_valid is False
    assert "pii_detected_in_assistant_message" in reason

# ---------------------------------------------------------------------------
# 3. JSONL Export & MLflow Tracking Tests
# ---------------------------------------------------------------------------

def test_tracker_exports_and_mlflow():
    tracker = FineTuneTracker()
    tracker.mlflow_available = True
    mock_mlflow = MagicMock()
    tracker.mlflow = mock_mlflow

    job_id = "test-job-uuid"
    messages = [
        {"messages": [{"role": "user", "content": "hello"}]}
    ]

    # Test export
    file_path = tracker.export_to_jsonl(messages, job_id)
    assert os.path.exists(file_path)
    with open(file_path, "r") as f:
        data = json.loads(f.read().strip())
        assert data["messages"][0]["content"] == "hello"

    # Clean up file
    if os.path.exists(file_path):
        os.remove(file_path)

    # Test MLflow log methods
    tracker.start_run(job_id)
    mock_mlflow.start_run.assert_called_once()

    tracker.log_training_params("run123", "base_model", [{}], 0.9, "range")
    mock_mlflow.log_params.assert_called_once()

    tracker.log_metrics("run123", 0.05, 0.06, 150)
    mock_mlflow.log_metrics.assert_called_once()

# ---------------------------------------------------------------------------
# 4. Registry & Model Seeding Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_model_registry():
    mock_redis_client = AsyncMock()
    # Mock Redis get to return None (no models)
    mock_redis_client.get = AsyncMock(return_value=None)
    mock_redis_client.set = AsyncMock(return_value="OK")
    
    with patch("pipelines.finetuning.registry.redis.from_url", return_value=mock_redis_client):
        registry = ModelRegistry()
        await registry.register_model_redis("finetuned-legal-v1", "legal")
        
        # Verify it fetched and set
        mock_redis_client.get.assert_called_once_with("router:models")
        
        # Check what was saved to Redis
        saved_arg = mock_redis_client.set.call_args[0][1]
        saved_models = json.loads(saved_arg)
        assert len(saved_models) == 4 # 3 default seed models + 1 new fine-tuned model
        assert any(m["model"] == "finetuned-legal-v1" and m["task_type"] == "legal" for m in saved_models)

# ---------------------------------------------------------------------------
# 5. DPO Extraction Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dpo_extraction():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    # Mock return rows for matching query (chosen/rejected pairs)
    mock_db_conn.fetch = AsyncMock(return_value=[
        {
            "query": "Is attention parallel?",
            "chosen": "Self-attention runs in parallel.",
            "rejected": "It runs sequentially."
        }
    ])

    with patch("pipelines.finetuning.dpo.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)), \
         patch("pipelines.finetuning.dpo.FineTuneTracker.export_to_jsonl") as mock_export:
         
        dpo = DPOExtractor()
        pairs = await dpo.extract_dpo_pairs("test-job")
        
        assert len(pairs) == 1
        assert pairs[0]["prompt"] == "Is attention parallel?"
        assert pairs[0]["chosen"] == "Self-attention runs in parallel."
        assert pairs[0]["rejected"] == "It runs sequentially."
        mock_export.assert_called_once_with(pairs, "test-job", is_dpo=True)

# ---------------------------------------------------------------------------
# 6. Fine-Tuning Job Manager & Mock Simulation Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_job_manager_simulation():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    # Mock fetchrow for runs details load in simulation loop
    mock_db_conn.fetchrow = AsyncMock(return_value={
        "mlflow_run_id": "run123",
        "training_pair_count": 5,
        "base_model": "llama-3.3-70b-versatile"
    })

    manager = FineTuneJobManager()
    
    with patch("pipelines.finetuning.job_manager.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)), \
         patch.object(manager.tracker, "log_metrics") as mock_metrics, \
         patch.object(manager.registry, "register_model_mlflow") as mock_mlflow_reg, \
         patch.object(manager.registry, "register_model_redis", AsyncMock()) as mock_redis_reg:
         
        # Run simulation loop
        await manager.simulate_finetune_job_loop("test-job-uuid", "legal")
        
        # Verify status updates (5 updates total: queued, running, training, validating, succeeded)
        assert mock_db_conn.execute.call_count == 6 # 5 updates in loop + 1 final completion metrics update
        mock_metrics.assert_called_once_with("run123", 0.05, 0.06, 750)
        mock_mlflow_reg.assert_called_once_with("run123", "finetuned-legal-v1")
        mock_redis_reg.assert_called_once_with("finetuned-legal-v1", "legal")

# ---------------------------------------------------------------------------
# 7. Router Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_router_fine_tuned_fallback():
    # Setup list containing a fine-tuned model
    models = [
        {"model": "gpt-4o-mini", "cost": 0.00015, "fine_tuned": False},
        {"model": "finetuned-legal-v1", "task_type": "legal", "fine_tuned": True, "cost": 0.00010}
    ]
    
    router = ModelRouter()
    router.get_models = AsyncMock(return_value=models)
    
    # 1. Without preferring fine-tuned, should select cheapest (which is fine-tuned here but let's check)
    criteria_normal = RoutingCriteria(task_type="legal", prefer_fine_tuned=False)
    res_normal = await router.route(criteria_normal)
    assert res_normal == "finetuned-legal-v1" # cheapest model in list is selected

    # 2. Prefer fine-tuned, should select the task-matching fine-tuned model
    criteria_ft = RoutingCriteria(task_type="legal", prefer_fine_tuned=True)
    res_ft = await router.route(criteria_ft)
    assert res_ft == "finetuned-legal-v1"

    # 3. Prefer fine-tuned but task_type doesn't match, should fallback or keep
    criteria_mismatch = RoutingCriteria(task_type="medical", prefer_fine_tuned=True)
    with pytest.raises(ValueError):
        await router.route(criteria_mismatch) # No medical fine-tuned model exists

# ---------------------------------------------------------------------------
# 8. API HTTP Endpoint Tests
# ---------------------------------------------------------------------------

def test_api_preview_endpoint():
    mock_extractor = MagicMock()
    mock_extractor.extract_pairs = AsyncMock(return_value=[
        {"user_message": "Query preview", "quality_score": 0.94}
    ])
    
    with patch("api.finetune.extractor", mock_extractor):
        response = client.get("/finetune/training-data/preview")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["query"] == "Query preview"
        assert data[0]["quality_score"] == 0.94

def test_api_jobs_list_endpoint():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    # Mock list query return
    mock_db_conn.fetch = AsyncMock(return_value=[
        {
            "id": uuid.uuid4(),
            "provider_job_id": "ft-job-123",
            "base_model": "llama-3.3-70b-versatile",
            "status": "queued",
            "mlflow_run_id": "run123",
            "training_pair_count": 10,
            "metrics": "{}",
            "created_at": None,
            "completed_at": None
        }
    ])

    with patch("api.finetune.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)):
        response = client.get("/finetune/jobs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["provider_job_id"] == "ft-job-123"

def test_api_job_details_endpoint():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    job_uuid = uuid.uuid4()
    mock_db_conn.fetchrow = AsyncMock(return_value={
        "id": job_uuid,
        "provider_job_id": "ft-job-123",
        "base_model": "llama-3.3-70b-versatile",
        "status": "training",
        "mlflow_run_id": "run123",
        "training_pair_count": 10,
        "metrics": "{}",
        "created_at": None,
        "completed_at": None
    })

    with patch("api.finetune.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)):
        response = client.get(f"/finetune/jobs/{str(job_uuid)}")
        assert response.status_code == 200
        data = response.json()
        assert data["provider_job_id"] == "ft-job-123"
        assert data["status"] == "training"

def test_api_submit_job_endpoint():
    mock_manager = MagicMock()
    mock_manager.submit_mock_finetune_job = AsyncMock(return_value="ft-job-123")
    
    with patch("api.finetune.manager", mock_manager):
        response = client.post("/finetune/jobs", json={
            "base_model": "llama-3.3-70b-versatile",
            "task_type": "legal"
        })
        assert response.status_code == 201
        data = response.json()
        assert data["provider_job_id"] == "ft-job-123"
        assert data["status"] == "queued"
