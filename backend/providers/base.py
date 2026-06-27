from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any, AsyncGenerator, Dict

@dataclass
class ChatMessage:
    role: str  # e.g., 'system', 'user', 'assistant'
    content: str | list  # Can be string or list of dicts for multimodal (e.g., text + images)
    name: str | None = None

@dataclass
class GenerationResult:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    finish_reason: str

class BaseLLMProvider(ABC):
    @property
    @abstractmethod
    def cost_per_input_token(self) -> float:
        pass

    @property
    @abstractmethod
    def cost_per_output_token(self) -> float:
        pass

    @property
    @abstractmethod
    def context_window(self) -> int:
        pass

    @abstractmethod
    async def complete(self, messages: List[ChatMessage], model: str = None, **kwargs) -> GenerationResult:
        """Complete the chat messages and return the result."""
        pass

    @abstractmethod
    async def stream(self, messages: List[ChatMessage], model: str = None, **kwargs) -> AsyncGenerator[str, None]:
        """Stream the response chunk by chunk."""
        pass

    @abstractmethod
    async def embed(self, texts: List[str], model: str = None, **kwargs) -> List[List[float]]:
        """Generate embeddings for the provided texts."""
        pass
