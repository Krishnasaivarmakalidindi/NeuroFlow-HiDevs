import time
import asyncio
import logging
from typing import List, AsyncGenerator, Dict, Any, Tuple
from anthropic import AsyncAnthropic, RateLimitError

from .base import BaseLLMProvider, ChatMessage, GenerationResult

logger = logging.getLogger(__name__)

PRICE_TABLE = {
    "claude-3-haiku-20240307": {
        "input": 0.25,
        "output": 1.25,
        "context": 200000
    },
    "claude-3-opus-20240229": {
        "input": 15.00,
        "output": 75.00,
        "context": 200000
    }
}

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model: str = None):
        self.client = AsyncAnthropic(api_key=api_key)
        self.default_model = model or "claude-3-haiku-20240307"

    @property
    def cost_per_input_token(self) -> float:
        return PRICE_TABLE.get(self.default_model, {}).get("input", 0.0) / 1_000_000

    @property
    def cost_per_output_token(self) -> float:
        return PRICE_TABLE.get(self.default_model, {}).get("output", 0.0) / 1_000_000

    @property
    def context_window(self) -> int:
        return PRICE_TABLE.get(self.default_model, {}).get("context", 200000)

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        input_price = PRICE_TABLE.get(model, {}).get("input", 0.0) / 1_000_000
        output_price = PRICE_TABLE.get(model, {}).get("output", 0.0) / 1_000_000
        return (input_tokens * input_price) + (output_tokens * output_price)

    def _format_messages(self, messages: List[ChatMessage]) -> Tuple[str, List[Dict[str, Any]]]:
        system_message = ""
        formatted_messages = []
        
        for msg in messages:
            if msg.role == "system":
                # Concatenate system messages if multiple exist, or just take the content
                if system_message:
                    system_message += "\n"
                system_message += str(msg.content)
            else:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
                
        return system_message, formatted_messages

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
        system_message, formatted_messages = self._format_messages(messages)
        
        # Anthropic requires max_tokens, give a default if not present
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096
            
        params = {
            "model": model,
            "messages": formatted_messages,
            **kwargs
        }
        if system_message:
            params["system"] = system_message

        start = time.perf_counter()

        response = await self._with_retry(self.client.messages.create, **params)
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        text = response.content[0].text if response.content else ""
        finish_reason = response.stop_reason or "end_turn"
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
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
        system_message, formatted_messages = self._format_messages(messages)
        
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096

        params = {
            "model": model,
            "messages": formatted_messages,
            **kwargs
        }
        if system_message:
            params["system"] = system_message

        async with self.client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(self, texts: List[str], model: str = None, **kwargs) -> List[List[float]]:
        # Anthropic does not support embeddings. We could provide a similar fallback but usually 
        # it is not routed to Anthropic for embeddings. We'll raise NotImplementedError.
        raise NotImplementedError("Anthropic does not currently support embeddings via the messages API.")
