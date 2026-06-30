import pytest
import asyncio
import time
import uuid
import logging
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from resilience.circuit_breaker import CircuitBreaker, CircuitOpenError
from resilience.rate_limiter import RedisTokenBucket, SlidingWindowRateLimiter
from resilience.backpressure import BackpressureManager
from resilience.timeout_manager import TimeoutManager, NeuroFlowTimeoutError
from resilience.adaptive_timeout import AdaptiveTimeoutTracker
from db.health import check_health_extended

client = TestClient(app)

# ---------------------------------------------------------------------------
# 1. Circuit Breaker Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures():
    name = f"test-circuit-{uuid.uuid4()}"
    cb = CircuitBreaker(name, failure_threshold=2, recovery_timeout=2)
    
    # Reset state in Redis
    await cb.redis_client.delete(f"circuit:{name}:state")
    await cb.redis_client.delete(f"circuit:{name}:failure_count")
    
    # 1. Success does not trip it
    async with cb:
        pass
        
    assert await cb.get_state() == "CLOSED"
    
    # 2. First failure
    try:
        async with cb:
            raise ValueError("First Error")
    except ValueError:
        pass
        
    assert await cb.get_state() == "CLOSED"
    assert await cb.get_failure_count() == 1
    
    # 3. Second failure trips it
    try:
        async with cb:
            raise ValueError("Second Error")
    except ValueError:
        pass
        
    assert await cb.get_state() == "OPEN"
    
    # 4. Immediate calls fail with CircuitOpenError
    with pytest.raises(CircuitOpenError):
        async with cb:
            pass
            
    await cb.close()

@pytest.mark.asyncio
async def test_circuit_breaker_half_open_limits_calls():
    name = f"test-circuit-half-{uuid.uuid4()}"
    cb = CircuitBreaker(name, failure_threshold=1, recovery_timeout=1, half_open_max_calls=2)
    
    await cb.redis_client.delete(f"circuit:{name}:state")
    await cb.redis_client.delete(f"circuit:{name}:failure_count")
    
    # Trip it
    try:
        async with cb:
            raise ValueError("Trip")
    except ValueError:
        pass
    assert await cb.get_state() == "OPEN"
    
    # Wait for recovery timeout
    await asyncio.sleep(1.1)
    
    # First half-open call (enters HALF_OPEN)
    async with cb:
        state = await cb.get_state()
        assert state == "HALF_OPEN"
        
        # Second half-open call is allowed (run concurrently)
        async with cb:
            # Third half-open call exceeds limit, fails with CircuitOpenError
            with pytest.raises(CircuitOpenError):
                async with cb:
                    pass
            
    await cb.close()

@pytest.mark.asyncio
async def test_circuit_breaker_recovery_success():
    name = f"test-circuit-rec-{uuid.uuid4()}"
    cb = CircuitBreaker(name, failure_threshold=1, recovery_timeout=1, half_open_max_calls=2)
    
    await cb.redis_client.delete(f"circuit:{name}:state")
    await cb.redis_client.delete(f"circuit:{name}:failure_count")
    
    # Trip
    try:
        async with cb:
            raise ValueError("Trip")
    except ValueError:
        pass
    
    # Wait for recovery
    await asyncio.sleep(1.1)
    
    # Execute 2 consecutive successes in HALF_OPEN to close/recover
    async with cb:
        pass
    async with cb:
        pass
        
    assert await cb.get_state() == "CLOSED"
    await cb.close()

# ---------------------------------------------------------------------------
# 2. Token Bucket Rate Limiter Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_token_bucket_consumption_and_refill():
    name = f"test-bucket-{uuid.uuid4()}"
    # Small capacity, high refill rate
    bucket = RedisTokenBucket(name, max_capacity=5, refill_rate=10.0)
    
    await bucket.redis_client.delete(f"rpb:{name}:tokens")
    await bucket.redis_client.delete(f"rpb:{name}:last_refill")
    
    # Consume all tokens
    for _ in range(5):
        assert await bucket.acquire(1) is True
        
    # Sixth call is rejected
    assert await bucket.acquire(1) is False
    
    # Wait for partial refill (0.2s = 2 tokens)
    await asyncio.sleep(0.25)
    assert await bucket.acquire(1) is True
    assert await bucket.acquire(1) is True
    assert await bucket.acquire(1) is False
    
    # Test wait_for_token
    start = time.time()
    await bucket.wait_for_token(1)
    duration = time.time() - start
    assert duration > 0.04
    
    await bucket.close()

# ---------------------------------------------------------------------------
# 3. API Sliding Window Rate Limiter Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sliding_window_rate_limiter():
    limiter = SlidingWindowRateLimiter()
    ip = "192.168.1.50"
    endpoint = "/test-endpoint"
    
    key = f"ratelimit:{ip}:{endpoint}"
    await limiter.redis_client.delete(key)
    
    # Allow 3 requests in a 10 second window
    for _ in range(3):
        allowed, retry = await limiter.is_allowed(ip, endpoint, limit=3, window_seconds=10)
        assert allowed is True
        assert retry == 0
        
    # 4th request exceeds limit
    allowed, retry = await limiter.is_allowed(ip, endpoint, limit=3, window_seconds=10)
    assert allowed is False
    assert retry > 0
    
    await limiter.close()

# ---------------------------------------------------------------------------
# 4. Backpressure Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backpressure_manager():
    bp = BackpressureManager()
    await bp.redis_client.delete("queue:ingest")
    
    # 1. Normal state
    res = await bp.check_backpressure()
    assert res is None
    
    # 2. Warning state (queue depth > 50)
    # push 60 items
    pipe = bp.redis_client.pipeline()
    for _ in range(60):
        pipe.lpush("queue:ingest", "doc-job")
    await pipe.execute()
    
    res = await bp.check_backpressure()
    assert res is not None
    status, body = res
    assert status == 202
    assert "warning" in body
    assert body["estimated_wait_minutes"] == 12.0 # 60 * 0.2
    
    # 3. Critical state (queue depth > 100)
    # push another 50 items
    pipe = bp.redis_client.pipeline()
    for _ in range(50):
        pipe.lpush("queue:ingest", "doc-job")
    await pipe.execute()
    
    res = await bp.check_backpressure()
    assert res is not None
    status, body = res
    assert status == 503
    assert "error" in body
    assert body["queue_depth"] == 110
    
    # Clean up
    await bp.redis_client.delete("queue:ingest")
    await bp.close()

# ---------------------------------------------------------------------------
# 5. Timeout & Adaptive Timeout Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_manager_static_and_redis():
    tm = TimeoutManager()
    
    await tm.redis_client.delete("timeouts:embedding")
    
    async def slow_task():
        await asyncio.sleep(0.5)
        return "OK"
        
    # Change static timeout for embedding to 0.1s temporarily for testing
    with patch.dict(tm.STATIC_TIMEOUTS, {"embedding": 0.1}):
        with pytest.raises(NeuroFlowTimeoutError):
            await tm.execute("embedding", slow_task())
            
    # Verify timeout count was incremented in Redis
    count = await tm.redis_client.get("timeouts:embedding")
    assert int(count) == 1
    
    await tm.close()

@pytest.mark.asyncio
async def test_adaptive_timeout():
    tracker = AdaptiveTimeoutTracker()
    tm = TimeoutManager()
    
    task = f"test-adaptive-{uuid.uuid4()}"
    
    # Record 10 latencies around 0.1s
    for _ in range(10):
        await tracker.record_latency(task, 0.1)
        
    # P95 should be ~0.1, so adaptive timeout = 0.1 * 1.5 = 0.15s
    timeout_val = await tm.get_timeout(task, use_adaptive=True)
    assert 0.13 <= timeout_val <= 0.17
    
    # Clean up
    await tracker.redis_client.delete(f"latency:{task}")
    await tracker.redis_client.delete(f"timeout:adaptive:{task}")
    await tracker.redis_client.delete(f"latency:{task}:p95_history")
    
    await tracker.close()
    await tm.close()

# ---------------------------------------------------------------------------
# 6. Extended Health Check Tests (Degraded & Critical)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check_statuses():
    # 1. Ok state
    with patch("db.health.check_postgres", AsyncMock(return_value=True)), \
         patch("db.health.check_redis", AsyncMock(return_value=True)), \
         patch("db.health.check_mlflow", AsyncMock(return_value=True)), \
         patch("redis.asyncio.Redis.keys", AsyncMock(return_value=[])):
         
        res = await check_health_extended()
        assert res["status"] == "ok"
        assert res["checks"]["postgres"]["status"] == "ok"

    # 2. Degraded state (a circuit is open)
    with patch("db.health.check_postgres", AsyncMock(return_value=True)), \
         patch("db.health.check_redis", AsyncMock(return_value=True)), \
         patch("db.health.check_mlflow", AsyncMock(return_value=True)), \
         patch("redis.asyncio.Redis.keys", AsyncMock(return_value=["circuit:groq:state"])), \
         patch("redis.asyncio.Redis.get", AsyncMock(return_value="OPEN")):
         
        res = await check_health_extended()
        assert res["status"] == "degraded"
        assert res["checks"]["circuit_breakers"]["groq"] == "OPEN"

    # 3. Critical state (database down)
    with patch("db.health.check_postgres", AsyncMock(return_value=False)), \
         patch("db.health.check_redis", AsyncMock(return_value=True)), \
         patch("db.health.check_mlflow", AsyncMock(return_value=True)):
         
        res = await check_health_extended()
        assert res["status"] == "critical"
        assert res["checks"]["postgres"]["status"] == "unhealthy"

# ---------------------------------------------------------------------------
# 7. Endpoint HTTP Status Integration Tests
# ---------------------------------------------------------------------------

def test_api_query_rate_limiting_http():
    # Simulate hitting rate limit by calling many times or mocking
    # Using mock on the sliding window check
    with patch("api.query.SlidingWindowRateLimiter.is_allowed", AsyncMock(return_value=(False, 45))):
        response = client.post("/query", json={"query": "hello", "pipeline_id": "p123"})
        assert response.status_code == 429
        assert response.json()["error"] == "rate_limit_exceeded"
        assert response.headers["Retry-After"] == "45"

def test_api_ingest_backpressure_critical_http():
    with patch("api.ingest.SlidingWindowRateLimiter.is_allowed", AsyncMock(return_value=(True, 0))), \
         patch("api.ingest.BackpressureManager.check_backpressure", AsyncMock(return_value=(503, {"error": "ingestion_queue_full", "queue_depth": 150, "retry_after": 30}))):
         
        response = client.post("/ingest")
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "ingestion_queue_full"
        assert data["queue_depth"] == 150
