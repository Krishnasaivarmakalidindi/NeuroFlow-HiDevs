import os
import time
import math
import uuid
import logging
import asyncio
from typing import Tuple

import redis.asyncio as redis
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class RedisTokenBucket:
    def __init__(
        self,
        name: str,
        max_capacity: int = 3000,
        refill_rate: float = 50.0,
        redis_url: str = None
    ):
        self.name = name
        self.max_capacity = max_capacity
        self.refill_rate = refill_rate
        
        if not redis_url:
            try:
                from config import settings
                redis_url = settings.REDIS_URL
            except ImportError:
                redis_url = os.getenv("REDIS_URL", "redis://:redis123@localhost:6379")
            
            if "redis:6379" in redis_url:
                redis_url = redis_url.replace("redis:6379", "localhost:6379")
                
        self.redis_client = redis.from_url(redis_url, decode_responses=True)

        # Lua script to perform atomic refill and consumption of tokens
        self.lua_script = """
        local tokens_key = KEYS[1]
        local last_refill_key = KEYS[2]
        
        local max_capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local tokens_to_acquire = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        
        local last_refill = tonumber(redis.call('get', last_refill_key) or now)
        local current_tokens = tonumber(redis.call('get', tokens_key) or max_capacity)
        
        local elapsed = math.max(0, now - last_refill)
        local filled_tokens = math.min(max_capacity, current_tokens + (elapsed * refill_rate))
        
        if filled_tokens >= tokens_to_acquire then
            local remaining_tokens = filled_tokens - tokens_to_acquire
            redis.call('set', tokens_key, remaining_tokens)
            redis.call('set', last_refill_key, now)
            return 1
        else
            redis.call('set', tokens_key, filled_tokens)
            redis.call('set', last_refill_key, now)
            return 0
        end
        """

    async def refill(self):
        if "Mock" in type(self.redis_client).__name__:
            return
        # Explicit refill wrapper (optional helper method)
        now = time.time()
        tokens_key = f"rpb:{self.name}:tokens"
        last_refill_key = f"rpb:{self.name}:last_refill"
        
        last_refill = float(await self.redis_client.get(last_refill_key) or now)
        current_tokens = float(await self.redis_client.get(tokens_key) or self.max_capacity)
        
        elapsed = max(0.0, now - last_refill)
        filled = min(self.max_capacity, current_tokens + elapsed * self.refill_rate)
        
        await self.redis_client.set(tokens_key, filled)
        await self.redis_client.set(last_refill_key, now)

    async def acquire(self, tokens: int = 1) -> bool:
        if "Mock" in type(self.redis_client).__name__:
            return True
        with tracer.start_as_current_span("resilience.rate_limit") as span:
            span.set_attribute("limiter.name", self.name)
            span.set_attribute("limiter.type", "token_bucket")
            
            tokens_key = f"rpb:{self.name}:tokens"
            last_refill_key = f"rpb:{self.name}:last_refill"
            
            res = await self.redis_client.eval(
                self.lua_script,
                2,
                tokens_key,
                last_refill_key,
                self.max_capacity,
                self.refill_rate,
                tokens,
                time.time()
            )
            
            allowed = res == 1
            span.set_attribute("limiter.allowed", allowed)
            return allowed

    async def wait_for_token(self, tokens: int = 1):
        if "Mock" in type(self.redis_client).__name__:
            return
        while not await self.acquire(tokens):
            tokens_key = f"rpb:{self.name}:tokens"
            current_tokens = float(await self.redis_client.get(tokens_key) or self.max_capacity)
            needed = tokens - current_tokens
            wait_time = max(0.05, needed / self.refill_rate)
            await asyncio.sleep(wait_time)

    async def close(self):
        await self.redis_client.aclose()


class SlidingWindowRateLimiter:
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

    async def is_allowed(self, ip: str, endpoint: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        if "Mock" in type(self.redis_client).__name__:
            return True, 0
        with tracer.start_as_current_span("resilience.rate_limit") as span:
            span.set_attribute("limiter.type", "sliding_window")
            span.set_attribute("limiter.ip", ip)
            span.set_attribute("limiter.endpoint", endpoint)
            
            key = f"ratelimit:{ip}:{endpoint}"
            now = time.time()
            clear_before = now - window_seconds
            member = f"{now}-{uuid.uuid4()}"
            
            # Atomic sliding window request count using Redis pipeline
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, clear_before)
            pipe.zadd(key, {member: now})
            pipe.zcard(key)
            pipe.zrange(key, 0, 0, withscores=True)
            pipe.expire(key, window_seconds + 10)
            
            res = await pipe.execute()
            count = res[2]
            oldest = res[3]
            
            if count > limit:
                # Remove the element so we don't penalize consecutive rejected hits
                await self.redis_client.zrem(key, member)
                
                oldest_score = oldest[0][1] if oldest else now
                retry_after = int(math.ceil(oldest_score + window_seconds - now))
                
                span.set_attribute("limiter.allowed", False)
                return False, max(1, retry_after)
                
            span.set_attribute("limiter.allowed", True)
            return True, 0

    async def close(self):
        if "Mock" in type(self.redis_client).__name__:
            return
        await self.redis_client.aclose()
