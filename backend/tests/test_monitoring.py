import pytest
import os
import json
import yaml
from unittest.mock import AsyncMock, patch, MagicMock
from opentelemetry import trace
from prometheus_client import REGISTRY

# Import monitoring modules
from monitoring.tracing import trace_sync, trace_async, get_tracer
from monitoring.metrics import (
    queries_total,
    ingestion_docs_total,
    llm_calls_total,
    circuit_breaker_trips,
    retrieval_latency,
    generation_latency,
    llm_cost,
    eval_faithfulness,
    eval_overall,
    queue_depth,
    circuit_breakers_open
)
from monitoring.anomaly_detector import QualityAnomalyDetector

@pytest.mark.asyncio
async def test_tracing_decorators():
    tracer = get_tracer()
    assert tracer is not None

    @trace_sync("test.sync.span")
    def dummy_sync(a, b):
        return a + b

    @trace_async("test.async.span")
    async def dummy_async(a, b):
        return a * b

    # Test executing decorated sync function
    res_sync = dummy_sync(2, 3)
    assert res_sync == 5

    # Test executing decorated async function
    res_async = await dummy_async(3, 4)
    assert res_async == 12

def test_prometheus_metrics_registry():
    # Verify that our custom metrics are registered in the global registry with their prefix
    registered_names = [metric.name for metric in REGISTRY.collect()]
    
    assert "neuroflow_queries" in registered_names
    assert "neuroflow_ingestion_docs" in registered_names
    assert "neuroflow_llm_calls" in registered_names
    assert "neuroflow_circuit_breaker_trips" in registered_names
    assert "neuroflow_retrieval_latency_seconds" in registered_names
    assert "neuroflow_generation_latency_seconds" in registered_names
    assert "neuroflow_llm_cost_usd" in registered_names
    assert "neuroflow_eval_faithfulness" in registered_names
    assert "neuroflow_eval_overall" in registered_names
    assert "neuroflow_queue_depth" in registered_names
    assert "neuroflow_circuit_breakers_open" in registered_names

@pytest.mark.asyncio
async def test_anomaly_detector_not_enough_data():
    detector = QualityAnomalyDetector()
    
    mock_db_conn = AsyncMock()
    mock_db_conn.fetch = AsyncMock(return_value=[])
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))
    
    with patch("db.pool.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)):
        anomalies = await detector.detect_anomalies("00000000-0000-0000-0000-000000000000")
        assert len(anomalies) == 0

@pytest.mark.asyncio
async def test_anomaly_detector_no_anomalies():
    detector = QualityAnomalyDetector()
    
    mock_db_conn = AsyncMock()
    mock_db_conn.fetch = AsyncMock(return_value=[
        {"overall_score": 0.85, "faithfulness": 0.85, "answer_relevance": 0.85, "context_precision": 0.85, "context_recall": 0.85, "created_at": "2026-07-01 12:00:00"},
        {"overall_score": 0.86, "faithfulness": 0.86, "answer_relevance": 0.86, "context_precision": 0.86, "context_recall": 0.86, "created_at": "2026-07-01 11:00:00"},
        {"overall_score": 0.84, "faithfulness": 0.84, "answer_relevance": 0.84, "context_precision": 0.84, "context_recall": 0.84, "created_at": "2026-07-01 10:00:00"}
    ])
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))
    
    with patch("db.pool.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)):
        anomalies = await detector.detect_anomalies("00000000-0000-0000-0000-000000000000")
        assert len(anomalies) == 0

@pytest.mark.asyncio
async def test_anomaly_detector_detects_degradation():
    detector = QualityAnomalyDetector()
    
    mock_db_conn = AsyncMock()
    mock_db_conn.fetch = AsyncMock(return_value=[
        {"overall_score": 0.40, "faithfulness": 0.40, "answer_relevance": 0.40, "context_precision": 0.40, "context_recall": 0.40, "created_at": "2026-07-01 12:00:00"}, # drop
        {"overall_score": 0.85, "faithfulness": 0.85, "answer_relevance": 0.85, "context_precision": 0.85, "context_recall": 0.85, "created_at": "2026-07-01 11:00:00"},
        {"overall_score": 0.86, "faithfulness": 0.86, "answer_relevance": 0.86, "context_precision": 0.86, "context_recall": 0.86, "created_at": "2026-07-01 10:00:00"},
        {"overall_score": 0.84, "faithfulness": 0.84, "answer_relevance": 0.84, "context_precision": 0.84, "context_recall": 0.84, "created_at": "2026-07-01 09:00:00"}
    ])
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))
    
    with patch("db.pool.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)):
        anomalies = await detector.detect_anomalies("00000000-0000-0000-0000-000000000000")
        assert len(anomalies) == 1
        assert anomalies[0]["metric"] == "overall_score"
        assert "Quality anomaly detected" in anomalies[0]["problem"]

def test_grafana_dashboards_valid():
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace_root = os.path.dirname(backend_dir)
    overview_path = os.path.join(workspace_root, "infra", "grafana", "system_overview.json")
    quality_path = os.path.join(workspace_root, "infra", "grafana", "quality_monitor.json")
    
    # Assert files exist
    assert os.path.exists(overview_path), f"{overview_path} does not exist"
    assert os.path.exists(quality_path), f"{quality_path} does not exist"
    
    # Assert valid JSON structures
    with open(overview_path, "r", encoding="utf-8") as f:
        overview_json = json.load(f)
        assert overview_json["uid"] == "neuroflow-system-overview"
        assert len(overview_json["panels"]) > 0

    with open(quality_path, "r", encoding="utf-8") as f:
        quality_json = json.load(f)
        assert quality_json["uid"] == "neuroflow-quality-monitor"
        assert len(quality_json["panels"]) > 0

def test_prometheus_alerts_valid():
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace_root = os.path.dirname(backend_dir)
    alerts_path = os.path.join(workspace_root, "infra", "prometheus", "alerts.yml")
    
    # Assert file exists
    assert os.path.exists(alerts_path), f"{alerts_path} does not exist"
    
    # Assert valid YAML and alerts list
    with open(alerts_path, "r", encoding="utf-8") as f:
        alerts_yaml = yaml.safe_load(f)
        assert "groups" in alerts_yaml
        rules = alerts_yaml["groups"][0]["rules"]
        alert_names = [r["alert"] for r in rules]
        assert "HighEvaluationFailureRate" in alert_names
        assert "CircuitBreakerOpen" in alert_names
        assert "EvaluationScoreDegraded" in alert_names
        assert "QueueDepthHigh" in alert_names
