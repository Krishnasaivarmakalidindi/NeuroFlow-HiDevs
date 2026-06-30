import time
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional
import redis.asyncio as redis
from opentelemetry import trace

from .base import BaseLLMProvider, ChatMessage, GenerationResult
from .router import ModelRouter, RoutingCriteria

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class NeuroFlowClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(NeuroFlowClient, cls).__new__(cls)
        return cls._instance

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        if not hasattr(self, "initialized"):
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.router = ModelRouter(redis_url)
            self.providers: Dict[str, BaseLLMProvider] = {}
            self.initialized = True
            
            # Register default providers to prevent uninitialized provider maps in production
            try:
                from .openai_provider import OpenAIProvider
                from .anthropic_provider import AnthropicProvider
                try:
                    from backend.config import settings
                except ImportError:
                    from config import settings
                import os
                
                self.register_provider("openai", OpenAIProvider(api_key=settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")))
                self.register_provider("anthropic", AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY") or ""))
            except Exception as e:
                logger.warning(f"Failed to register default providers: {e}")

    def register_provider(self, name: str, provider: BaseLLMProvider):
        self.providers[name] = provider

    async def _track_metrics(self, model: str, cost: float):
        calls_key = f"metrics:model:{model}:calls"
        cost_key = f"metrics:model:{model}:cost_usd"
        
        await self.redis_client.incr(calls_key)
        await self.redis_client.incrbyfloat(cost_key, cost)

    def _get_provider_for_model(self, model: str) -> BaseLLMProvider:
        # Simple static mapping based on prefix or explicit logic. 
        # Ideally this would be part of the router model data.
        if model.startswith("gpt") or model.startswith("llama") or model.startswith("finetuned") or model.startswith("fine-tuned"):
            return self.providers.get("openai")
        elif model.startswith("claude"):
            return self.providers.get("anthropic")
        
        # Fallback based on registry if available, but for simplicity here we check prefixes
        raise ValueError(f"No provider registered for model: {model}")

    async def chat(self, messages: List[ChatMessage], criteria: RoutingCriteria, stream: bool = False, **kwargs) -> Any:
        model = await self.router.route(criteria)
        provider = self._get_provider_for_model(model)
        
        if not provider:
            raise ValueError(f"Provider not found for routed model: {model}")

        provider_name = "unknown"
        for name, p in self.providers.items():
            if p == provider:
                provider_name = name
                break

        # 1. Rate Limiting (Provider & Pipeline)
        from resilience import RedisTokenBucket, CircuitBreaker, TimeoutManager, AdaptiveTimeoutTracker
        
        provider_limiter = RedisTokenBucket(provider_name, max_capacity=3000, refill_rate=50.0)
        await provider_limiter.wait_for_token()
        await provider_limiter.close()

        pipeline_id = kwargs.get("pipeline_id")
        if pipeline_id:
            limit = 60
            try:
                redis_url = self.redis_client.connection_pool.connection_kwargs.get("url")
                temp_redis = redis.from_url(redis_url, decode_responses=True)
                custom_limit = await temp_redis.get(f"rpb:pipeline:{pipeline_id}:rpm")
                if custom_limit:
                    limit = int(custom_limit)
                await temp_redis.close()
            except Exception:
                pass
            
            pipeline_limiter = RedisTokenBucket(f"pipeline:{pipeline_id}", max_capacity=limit, refill_rate=limit / 60.0)
            await pipeline_limiter.wait_for_token()
            await pipeline_limiter.close()

        # 2. Timeout and Circuit Breaker
        timeout_manager = TimeoutManager()
        adaptive_tracker = AdaptiveTimeoutTracker()
        start_time = time.time()

        async def call_with_cb():
            async with CircuitBreaker(provider_name, failure_threshold=5, recovery_timeout=60, half_open_max_calls=3):
                if stream:
                    # Resolve async stream generator inside CB
                    gen = provider.stream(messages, model, **kwargs)
                    return gen
                else:
                    return await provider.complete(messages, model, **kwargs)

        try:
            task_type = "chat_completion"
            if criteria.task_type == "evaluation" or kwargs.get("is_evaluation"):
                task_type = "evaluation"
                
            use_adaptive = kwargs.get("use_adaptive", True)
            
            if stream:
                # Setup stream within CB context
                stream_gen = await timeout_manager.execute(task_type, call_with_cb(), use_adaptive=use_adaptive)
                
                # Wrap generator yields
                async def generator_wrapper() -> AsyncGenerator[str, None]:
                    with tracer.start_as_current_span("llm.stream") as stream_span:
                        stream_span.set_attribute("model", model)
                        try:
                            async for chunk in stream_gen:
                                yield chunk
                            
                            latency_ms = (time.time() - start_time) * 1000
                            stream_span.set_attribute("latency_ms", latency_ms)
                            await self._track_metrics(model, 0.0)
                        except Exception as e:
                            stream_span.record_exception(e)
                            raise e
                return generator_wrapper()
            else:
                with tracer.start_as_current_span("llm.chat") as span:
                    span.set_attribute("model", model)
                    result = await timeout_manager.execute(task_type, call_with_cb(), use_adaptive=use_adaptive)
                    
                    span.set_attribute("input_tokens", result.input_tokens)
                    span.set_attribute("output_tokens", result.output_tokens)
                    span.set_attribute("cost_usd", result.cost_usd)
                    span.set_attribute("latency_ms", result.latency_ms)
                    
                    await self._track_metrics(model, result.cost_usd)
                    
                    # Record latency in adaptive tracker
                    latency = time.time() - start_time
                    await adaptive_tracker.record_latency(task_type, latency)
                    
                    return result
        finally:
            await timeout_manager.close()
            await adaptive_tracker.close()

    async def embed(self, texts: List[str], criteria: RoutingCriteria, **kwargs) -> List[List[float]]:
        model = await self.router.route(criteria)
        provider = self._get_provider_for_model(model)
        
        if not provider:
            raise ValueError(f"Provider not found for routed model: {model}")

        provider_name = "unknown"
        for name, p in self.providers.items():
            if p == provider:
                provider_name = name
                break

        # 1. Rate Limiting (Provider & Pipeline)
        from resilience import RedisTokenBucket, CircuitBreaker, TimeoutManager, AdaptiveTimeoutTracker

        provider_limiter = RedisTokenBucket(provider_name, max_capacity=3000, refill_rate=50.0)
        await provider_limiter.wait_for_token()
        await provider_limiter.close()

        pipeline_id = kwargs.get("pipeline_id")
        if pipeline_id:
            limit = 60
            try:
                redis_url = self.redis_client.connection_pool.connection_kwargs.get("url")
                temp_redis = redis.from_url(redis_url, decode_responses=True)
                custom_limit = await temp_redis.get(f"rpb:pipeline:{pipeline_id}:rpm")
                if custom_limit:
                    limit = int(custom_limit)
                await temp_redis.close()
            except Exception:
                pass
            
            pipeline_limiter = RedisTokenBucket(f"pipeline:{pipeline_id}", max_capacity=limit, refill_rate=limit / 60.0)
            await pipeline_limiter.wait_for_token()
            await pipeline_limiter.close()

        # 2. Timeout and Circuit Breaker
        timeout_manager = TimeoutManager()
        adaptive_tracker = AdaptiveTimeoutTracker()
        start_time = time.time()

        async def call_with_cb():
            async with CircuitBreaker(provider_name, failure_threshold=5, recovery_timeout=60, half_open_max_calls=3):
                return await provider.embed(texts, model, **kwargs)

        try:
            task_type = "embedding"
            with tracer.start_as_current_span("llm.embed") as span:
                span.set_attribute("model", model)
                result = await timeout_manager.execute(task_type, call_with_cb(), use_adaptive=kwargs.get("use_adaptive", True))
                
                latency_ms = (time.time() - start_time) * 1000
                span.set_attribute("latency_ms", latency_ms)
                
                await self._track_metrics(model, 0.0)
                
                # Record latency in adaptive tracker
                latency = time.time() - start_time
                await adaptive_tracker.record_latency(task_type, latency)
                
                return result
        finally:
            await timeout_manager.close()
            await adaptive_tracker.close()
