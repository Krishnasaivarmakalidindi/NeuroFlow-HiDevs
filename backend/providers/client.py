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
        if model.startswith("gpt"):
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

        start_time = time.time()
        
        with tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("model", model)
            
            try:
                if stream:
                    return self._stream_wrapper(provider, messages, model, start_time, span, **kwargs)
                else:
                    result = await provider.complete(messages, model, **kwargs)
                    
                    span.set_attribute("input_tokens", result.input_tokens)
                    span.set_attribute("output_tokens", result.output_tokens)
                    span.set_attribute("cost_usd", result.cost_usd)
                    span.set_attribute("latency_ms", result.latency_ms)
                    
                    await self._track_metrics(model, result.cost_usd)
                    
                    return result
            except Exception as e:
                span.record_exception(e)
                raise

    async def _stream_wrapper(self, provider, messages, model, start_time, span, **kwargs) -> AsyncGenerator[str, None]:
        # For streams, we might not get token counts directly from all providers easily without counting them ourselves
        # or checking the final chunk. For simplicity, we just emit latency.
        try:
            async for chunk in provider.stream(messages, model, **kwargs):
                yield chunk
                
            latency_ms = (time.time() - start_time) * 1000
            span.set_attribute("latency_ms", latency_ms)
            
            # Note: actual token counting for stream requires a token counter (e.g. tiktoken)
            # Here we just track a call.
            await self._track_metrics(model, 0.0)
        except Exception as e:
            span.record_exception(e)
            raise

    async def embed(self, texts: List[str], criteria: RoutingCriteria, **kwargs) -> List[List[float]]:
        model = await self.router.route(criteria)
        provider = self._get_provider_for_model(model)
        
        if not provider:
            raise ValueError(f"Provider not found for routed model: {model}")

        start_time = time.time()
        
        with tracer.start_as_current_span("llm.embed") as span:
            span.set_attribute("model", model)
            
            try:
                result = await provider.embed(texts, model, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                span.set_attribute("latency_ms", latency_ms)
                
                # Approximate cost tracking could be added here
                await self._track_metrics(model, 0.0)
                
                return result
            except Exception as e:
                span.record_exception(e)
                raise
