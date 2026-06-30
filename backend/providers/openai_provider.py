import time
import asyncio
import logging
from typing import List, AsyncGenerator, Dict, Any
from openai import AsyncOpenAI, RateLimitError

try:
    from backend.config import settings
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config import settings

from .base import BaseLLMProvider, ChatMessage, GenerationResult

# Import resilience framework exceptions for client compliance
try:
    from resilience import CircuitOpenError, NeuroFlowTimeoutError
except ImportError:
    CircuitOpenError = Exception
    NeuroFlowTimeoutError = Exception

logger = logging.getLogger(__name__)

PRICE_TABLE = {
    "llama-3.3-70b-versatile": {
        "input": 0.59,
        "output": 0.79,
        "context": 131072
    },
    "qwen/qwen3-32b": {
        "input": 0.29,
        "output": 0.59,
        "context": 131072
    },
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
        "context": 128000
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
        "context": 128000
    }
}

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model: str = None):
        self.client = AsyncOpenAI(
            api_key=api_key or settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
        self.default_model = model or settings.DEFAULT_CHAT_MODEL

    @property
    def cost_per_input_token(self) -> float:
        return PRICE_TABLE.get(self.default_model, {}).get("input", 0.0) / 1_000_000

    @property
    def cost_per_output_token(self) -> float:
        return PRICE_TABLE.get(self.default_model, {}).get("output", 0.0) / 1_000_000

    @property
    def context_window(self) -> int:
        return PRICE_TABLE.get(self.default_model, {}).get("context", 4096)

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        input_price = PRICE_TABLE.get(model, {}).get("input", 0.0) / 1_000_000
        output_price = PRICE_TABLE.get(model, {}).get("output", 0.0) / 1_000_000
        return (input_tokens * input_price) + (output_tokens * output_price)

    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        formatted = []
        for msg in messages:
            d = {"role": msg.role, "content": msg.content}
            if msg.name:
                d["name"] = msg.name
            formatted.append(d)
        return formatted

    async def _with_retry(self, func, *args, **kwargs):
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except RateLimitError as e:
                if attempt == max_retries:
                    raise
                
                # Check for retry_after in response headers if available, else exponential backoff
                retry_after = getattr(e.response, "headers", {}).get("retry-after")
                if retry_after:
                    delay = float(retry_after)
                else:
                    delay = base_delay * (2 ** attempt)
                
                logger.warning(f"Rate limit hit. Retrying in {delay} seconds (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)

    async def complete(self, messages: List[ChatMessage], model: str = None, **kwargs) -> GenerationResult:
        model = model or self.default_model
        formatted_messages = self._format_messages(messages)
        
        start = time.perf_counter()
        
        response = await self._with_retry(
            self.client.chat.completions.create,
            model=model,
            messages=formatted_messages,
            **kwargs
        )
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        text = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason or "stop"
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        cost_usd = self._calculate_cost(input_tokens, output_tokens, model)
        
        return GenerationResult(
            content=text,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            finish_reason=finish_reason
        )

    async def stream(self, messages: List[ChatMessage], model: str = None, **kwargs) -> AsyncGenerator[str, None]:
        model = model or self.default_model
        formatted_messages = self._format_messages(messages)
        
        stream_response = await self._with_retry(
            self.client.chat.completions.create,
            model=model,
            messages=formatted_messages,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embed(self, texts: List[str], model: str = None, **kwargs) -> List[List[float]]:
        # Handle batching of 100
        model = model or "text-embedding-3-small"
        batch_size = 100
        all_embeddings = []
        
        start = time.perf_counter()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                response = await self._with_retry(
                    self.client.embeddings.create,
                    model=model,
                    input=batch_texts,
                    **kwargs
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception:
                # Fallback to sentence-transformers or mock embeddings
                try:
                    from sentence_transformers import SentenceTransformer
                    fallback_model = SentenceTransformer("all-MiniLM-L6-v2")
                    vectors = fallback_model.encode(batch_texts).tolist()
                    all_embeddings.extend(vectors)
                except ImportError:
                    vectors = [[0.0] * 1536 for _ in batch_texts]
                    all_embeddings.extend(vectors)
                    
        latency_ms = (time.perf_counter() - start) * 1000
        # Optional: store latency if returning a GenerationResult for embeddings, but interface returns List[List[float]]
        
        return all_embeddings
