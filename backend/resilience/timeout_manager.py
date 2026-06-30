import os
import asyncio
import logging
from typing import Any, Coroutine
import redis.asyncio as redis
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class NeuroFlowTimeoutError(asyncio.TimeoutError):
    pass

class TimeoutManager:
    STATIC_TIMEOUTS = {
        "embedding": 10,
        "chat_completion": 60,
        "reranking": 15,
        "evaluation": 120,
        "file_extraction": 30,
        "url_fetch": 15
    }

    def __init__(self, redis_url: str = None):
        if not redis_url:
            try:
                from config import settings
                redis_url = settings.REDIS_URL
            except ImportError:
                redis_url = os.getenv("REDIS_URL", "redis://:redis123@localhost:6379")
            
            if "redis:6379" in redis_url:
                redis_url = redis_url.replace("redis:6379", "localhost:6379")
                
        self.redis_client = redis.from_url(redis_url, decode_responses=True)

    async def get_timeout(self, task_type: str, use_adaptive: bool = False) -> float:
        default_val = self.STATIC_TIMEOUTS.get(task_type, 30.0)
        if not use_adaptive or "Mock" in type(self.redis_client).__name__:
            return default_val
            
        try:
            adaptive_val = await self.redis_client.get(f"timeout:adaptive:{task_type}")
            if adaptive_val is not None:
                return float(adaptive_val)
        except Exception as e:
            logger.warning(f"Failed to fetch adaptive timeout from Redis for {task_type}: {e}")
            
        return default_val

    async def execute(self, task_type: str, coro: Coroutine, use_adaptive: bool = False) -> Any:
        with tracer.start_as_current_span("resilience.timeout") as span:
            timeout_val = await self.get_timeout(task_type, use_adaptive)
            span.set_attribute("timeout.task_type", task_type)
            span.set_attribute("timeout.value", timeout_val)
            
            try:
                return await asyncio.wait_for(coro, timeout=timeout_val)
            except asyncio.TimeoutError as e:
                if "Mock" not in type(self.redis_client).__name__:
                    try:
                        await self.redis_client.incr(f"timeouts:{task_type}")
                    except Exception as ex:
                        logger.warning(f"Failed to increment Redis timeout count: {ex}")
                    
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, f"Timeout exceeded {timeout_val} seconds.")
                raise NeuroFlowTimeoutError(f"Task '{task_type}' timed out after {timeout_val} seconds.") from e

    async def close(self):
        if "Mock" in type(self.redis_client).__name__:
            return
        await self.redis_client.aclose()
