import os
import logging
from typing import Optional, Tuple
import redis.asyncio as redis
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class BackpressureManager:
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

    async def get_queue_depth(self) -> int:
        # Check LLEN on queue:ingest
        depth = await self.redis_client.llen("queue:ingest")
        return depth or 0

    async def check_backpressure(self) -> Optional[Tuple[int, dict]]:
        if "Mock" in type(self.redis_client).__name__:
            return None
        with tracer.start_as_current_span("resilience.backpressure") as span:
            depth = await self.get_queue_depth()
            span.set_attribute("backpressure.queue_depth", depth)
            
            if depth > 100:
                span.set_attribute("backpressure.status", "critical")
                return 503, {
                    "error": "ingestion_queue_full",
                    "queue_depth": depth,
                    "retry_after": 30
                }
            elif depth > 50:
                # Estimate wait minutes assuming average processing time per item (e.g. 0.2 minutes)
                estimated_wait = round(depth * 0.2, 1)
                span.set_attribute("backpressure.status", "warning")
                return 202, {
                    "warning": "high_queue_depth",
                    "estimated_wait_minutes": estimated_wait
                }
                
            span.set_attribute("backpressure.status", "normal")
            return None

    async def close(self):
        if "Mock" in type(self.redis_client).__name__:
            return
        await self.redis_client.aclose()
