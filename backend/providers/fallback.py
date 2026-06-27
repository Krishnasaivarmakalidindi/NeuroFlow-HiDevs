import logging
from typing import List, Dict, Any, AsyncGenerator

from .base import ChatMessage, GenerationResult
from .client import NeuroFlowClient

logger = logging.getLogger(__name__)

class FallbackChain:
    def __init__(self, models: List[str], client: NeuroFlowClient):
        self.models = models
        self.client = client

    async def complete(self, messages: List[ChatMessage], **kwargs) -> GenerationResult:
        last_exception = None
        
        for model in self.models:
            try:
                provider = self.client._get_provider_for_model(model)
                if not provider:
                    logger.warning(f"Provider not found for fallback model {model}. Skipping.")
                    continue
                
                # Mock a RoutingCriteria that bypasses the router and goes straight to the model
                # Alternatively, use provider directly, but we want metrics.
                # So we simulate the metric tracking inline here, or call a protected client method
                # We'll just call the provider directly and do metrics tracking here to avoid 
                # routing logic overriding our fallback choice.
                
                start_time = import_time()
                
                with self.client._tracer_span(model, "llm.chat") as span:
                    # Fallback.py just returns the result untouched, but let's make sure it doesn't try to read `text`
                    result = await provider.complete(messages, model, **kwargs)
                    span.set_attribute("input_tokens", result.input_tokens)
                    span.set_attribute("output_tokens", result.output_tokens)
                    span.set_attribute("cost_usd", result.cost_usd)
                    
                    await self.client._track_metrics(model, result.cost_usd)
                    return result
                    
            except Exception as e:
                logger.error(f"Fallback model {model} failed with error: {str(e)}")
                last_exception = e
                continue
                
        raise RuntimeError(f"All fallback models failed. Last error: {str(last_exception)}")

# Helper to fix imports for time
def import_time():
    import time
    return time.time()
    
# Monkey patch NeuroFlowClient for trace spanning helper
from opentelemetry import trace
tracer = trace.get_tracer(__name__)
import contextlib

@contextlib.contextmanager
def _tracer_span(self, model: str, span_name: str):
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("model", model)
        yield span

NeuroFlowClient._tracer_span = _tracer_span
