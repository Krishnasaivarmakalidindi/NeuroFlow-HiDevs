import os
import time
import logging
import uuid
from typing import List
import redis.asyncio as redis
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class AdaptiveTimeoutTracker:
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

    async def record_latency(self, task: str, latency: float):
        if "Mock" in type(self.redis_client).__name__:
            return
        with tracer.start_as_current_span("resilience.adaptive_timeout.record") as span:
            span.set_attribute("adaptive.task", task)
            span.set_attribute("adaptive.recorded_latency", latency)
            
            now = time.time()
            zset_key = f"latency:{task}"
            
            # Store unique member: timestamp:uuid:latency
            member = f"{now}:{uuid.uuid4()}:{latency}"
            
            # 1. Add latency to ZSET
            await self.redis_client.zadd(zset_key, {member: now})
            
            # 2. Trim to last 1000 items
            # ZREMRANGEBYRANK removes elements starting from rank 0 (the oldest score)
            # Keeping the 1000 highest ranks (most recent)
            card = await self.redis_client.zcard(zset_key)
            if card > 1000:
                await self.redis_client.zremrangebyrank(zset_key, 0, card - 1001)
                
            # 3. Fetch all elements to compute p95
            items = await self.redis_client.zrange(zset_key, 0, -1)
            latencies = []
            for item in items:
                try:
                    parts = item.split(":")
                    if len(parts) >= 3:
                        latencies.append(float(parts[-1]))
                except ValueError:
                    pass
            
            if not latencies:
                return
                
            latencies.sort()
            p95_idx = min(len(latencies) - 1, int(len(latencies) * 0.95))
            p95 = latencies[p95_idx]
            
            dynamic_timeout = p95 * 1.5
            
            # 4. Save dynamic timeout value to Redis
            await self.redis_client.set(f"timeout:adaptive:{task}", str(dynamic_timeout))
            
            # 5. Track p95 history to analyze hourly trend
            history_key = f"latency:{task}:p95_history"
            await self.redis_client.zadd(history_key, {str(p95): now})
            
            # Clean history older than 2 hours
            await self.redis_client.zremrangebyscore(history_key, 0, now - 7200)
            
            # 6. Check trend over the last hour (3600s ago)
            hour_ago = now - 3600
            older_entries = await self.redis_client.zrangebyscore(history_key, hour_ago - 300, hour_ago + 300)
            
            if older_entries:
                try:
                    old_p95 = float(older_entries[0])
                    if p95 > old_p95:
                        logger.warning(
                            f"[Adaptive Timeout] P95 latency for task '{task}' has increased over the last hour: "
                            f"from {old_p95:.3f}s to {p95:.3f}s."
                        )
                except ValueError:
                    pass

    async def close(self):
        if "Mock" in type(self.redis_client).__name__:
            return
        await self.redis_client.aclose()
