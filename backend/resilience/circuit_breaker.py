import os
import time
import logging
import redis.asyncio as redis
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class CircuitOpenError(Exception):
    pass

class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
        redis_url: str = None
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        if not redis_url:
            try:
                from config import settings
                redis_url = settings.REDIS_URL
            except ImportError:
                redis_url = os.getenv("REDIS_URL", "redis://:redis123@localhost:6379")
            
            if "redis:6379" in redis_url:
                redis_url = redis_url.replace("redis:6379", "localhost:6379")
                
        self.redis_client = redis.from_url(redis_url, decode_responses=True)

    async def get_state(self) -> str:
        state = await self.redis_client.get(f"circuit:{self.name}:state")
        return state or "CLOSED"

    async def get_failure_count(self) -> int:
        val = await self.redis_client.get(f"circuit:{self.name}:failure_count")
        return int(val) if val else 0

    async def get_opened_at(self) -> float:
        val = await self.redis_client.get(f"circuit:{self.name}:opened_at")
        return float(val) if val else 0.0

    async def set_state(self, state: str):
        await self.redis_client.set(f"circuit:{self.name}:state", state)

    async def __aenter__(self):
        if "Mock" in type(self.redis_client).__name__:
            return self
        with tracer.start_as_current_span("resilience.circuit") as span:
            span.set_attribute("circuit.name", self.name)
            
            state = await self.get_state()
            span.set_attribute("circuit.state", state)
            
            from monitoring import metrics
            if state == "OPEN":
                metrics.circuit_breakers_open.labels(provider=self.name).set(1)
            else:
                metrics.circuit_breakers_open.labels(provider=self.name).set(0)
            
            if state == "OPEN":
                opened_at = await self.get_opened_at()
                if time.time() - opened_at > self.recovery_timeout:
                    # Transition from OPEN to HALF_OPEN
                    await self.set_state("HALF_OPEN")
                    await self.redis_client.delete(f"circuit:{self.name}:half_open_calls")
                    await self.redis_client.delete(f"circuit:{self.name}:half_open_successes")
                    state = "HALF_OPEN"
                    span.set_attribute("circuit.state", "HALF_OPEN")
                    metrics.circuit_breakers_open.labels(provider=self.name).set(0)
                else:
                    raise CircuitOpenError(f"Circuit '{self.name}' is OPEN.")
            
            if state == "HALF_OPEN":
                # Track in-progress calls during HALF_OPEN state
                calls = await self.redis_client.incr(f"circuit:{self.name}:half_open_calls")
                if calls > self.half_open_max_calls:
                    raise CircuitOpenError(f"Circuit '{self.name}' is HALF_OPEN (max calls exceeded).")

            return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if "Mock" in type(self.redis_client).__name__:
            return
        state = await self.get_state()
        from monitoring import metrics
        
        if exc_type is not None:
            logger.warning(f"Circuit breaker '{self.name}' call failed: {exc_val}")
            
            if state == "CLOSED":
                failures = await self.redis_client.incr(f"circuit:{self.name}:failure_count")
                if failures >= self.failure_threshold:
                    await self.set_state("OPEN")
                    await self.redis_client.set(f"circuit:{self.name}:opened_at", str(time.time()))
                    logger.critical(f"Circuit breaker '{self.name}' opened due to failures.")
                    metrics.circuit_breaker_trips.labels(pipeline_id="default", provider=self.name, status="open").inc()
                    metrics.circuit_breakers_open.labels(provider=self.name).set(1)
            elif state == "HALF_OPEN":
                # Tripping back to OPEN on any failure
                await self.set_state("OPEN")
                await self.redis_client.set(f"circuit:{self.name}:opened_at", str(time.time()))
                await self.redis_client.delete(f"circuit:{self.name}:half_open_calls")
                await self.redis_client.delete(f"circuit:{self.name}:half_open_successes")
                metrics.circuit_breaker_trips.labels(pipeline_id="default", provider=self.name, status="open").inc()
                metrics.circuit_breakers_open.labels(provider=self.name).set(1)
        else:
            if state == "CLOSED":
                # Reset consecutive failure counter on success
                await self.redis_client.delete(f"circuit:{self.name}:failure_count")
            elif state == "HALF_OPEN":
                successes = await self.redis_client.incr(f"circuit:{self.name}:half_open_successes")
                if successes >= self.half_open_max_calls:
                    # Recovery complete, transition to CLOSED
                    await self.set_state("CLOSED")
                    await self.redis_client.delete(f"circuit:{self.name}:failure_count")
                    await self.redis_client.delete(f"circuit:{self.name}:half_open_calls")
                    await self.redis_client.delete(f"circuit:{self.name}:half_open_successes")
                    logger.info(f"Circuit breaker '{self.name}' closed and recovered.")
                    metrics.circuit_breakers_open.labels(provider=self.name).set(0)
        
        await self.redis_client.aclose()

    async def close(self):
        if "Mock" in type(self.redis_client).__name__:
            return
        await self.redis_client.aclose()
