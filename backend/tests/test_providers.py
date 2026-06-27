import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from opentelemetry import trace

from providers.base import ChatMessage
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.router import ModelRouter, RoutingCriteria
from providers.client import NeuroFlowClient
from providers.fallback import FallbackChain

@pytest.fixture
def mock_openai():
    with patch("providers.openai_provider.AsyncOpenAI") as mock:
        yield mock

@pytest.fixture
def mock_anthropic():
    with patch("providers.anthropic_provider.AsyncAnthropic") as mock:
        yield mock

@pytest.fixture
def mock_redis():
    with patch("redis.asyncio.from_url") as mock:
        mock_instance = AsyncMock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.mark.asyncio
async def test_openai_provider_complete(mock_openai):
    provider = OpenAIProvider(api_key="test")
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello openai"), finish_reason="stop")]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
    
    mock_client_instance = mock_openai.return_value
    mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
    
    messages = [ChatMessage(role="user", content="Hi")]
    result = await provider.complete(messages, model="llama-3.3-70b-versatile")
    
    assert result.content == "Hello openai"
    assert result.model == "llama-3.3-70b-versatile"
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    assert result.cost_usd > 0
    assert result.latency_ms >= 0
    assert result.finish_reason == "stop"

@pytest.mark.asyncio
async def test_anthropic_provider_complete(mock_anthropic):
    provider = AnthropicProvider(api_key="test")
    
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello anthropic")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock(input_tokens=15, output_tokens=25)
    
    mock_client_instance = mock_anthropic.return_value
    mock_client_instance.messages.create = AsyncMock(return_value=mock_response)
    
    messages = [ChatMessage(role="system", content="You are AI"), ChatMessage(role="user", content="Hi")]
    result = await provider.complete(messages, model="claude-3-haiku-20240307")
    
    assert result.content == "Hello anthropic"
    assert result.model == "claude-3-haiku-20240307"
    assert result.input_tokens == 15
    assert result.output_tokens == 25
    assert result.latency_ms >= 0
    assert result.finish_reason == "end_turn"

@pytest.mark.asyncio
async def test_router_logic(mock_redis):
    router = ModelRouter()
    import json
    models = [
        {"provider": "openai", "model": "gpt-4o-mini", "vision": True, "context": 128000, "cost": 0.15, "fine_tuned": False},
        {"provider": "openai", "model": "gpt-4o", "vision": True, "context": 128000, "cost": 2.5, "is_judge": True},
        {"provider": "anthropic", "model": "claude-3-opus-20240229", "vision": True, "context": 200000, "cost": 15.0},
        {"provider": "openai", "model": "ft:gpt-3.5-turbo:my-org", "vision": False, "context": 16000, "cost": 3.0, "fine_tuned": True}
    ]
    mock_redis.get.return_value = json.dumps(models)

    model = await router.route(RoutingCriteria())
    assert model == "gpt-4o-mini"

    model = await router.route(RoutingCriteria(require_long_context=True))
    assert model in ["gpt-4o-mini", "gpt-4o", "claude-3-opus-20240229"]

    model = await router.route(RoutingCriteria(prefer_fine_tuned=True))
    assert model == "ft:gpt-3.5-turbo:my-org"

    model = await router.route(RoutingCriteria(task_type="evaluation"))
    assert model == "gpt-4o"

    model = await router.route(RoutingCriteria(require_long_context=True, max_cost_per_call=0.20))
    assert model == "gpt-4o-mini"

@pytest.mark.asyncio
async def test_fallback_chain(mock_redis, mock_openai, mock_anthropic):
    client = NeuroFlowClient()
    client.providers = {}
    
    openai_provider = OpenAIProvider()
    openai_provider.complete = AsyncMock(side_effect=[Exception("OpenAI Down"), MagicMock(content="Success", model="gpt-4o-mini", cost_usd=0.0, input_tokens=0, output_tokens=0, latency_ms=10, finish_reason="stop")])
    
    anthropic_provider = AnthropicProvider()
    anthropic_provider.complete = AsyncMock(return_value=MagicMock(content="Haiku Success", model="claude-3-haiku-20240307", cost_usd=0.1, input_tokens=10, output_tokens=10, latency_ms=20, finish_reason="stop"))
    
    client.register_provider("openai", openai_provider)
    client.register_provider("anthropic", anthropic_provider)
    
    client.redis_client = mock_redis
    
    chain = FallbackChain(["gpt-4o-mini", "claude-3-haiku-20240307"], client)
    messages = [ChatMessage(role="user", content="Test")]
    
    result = await chain.complete(messages)
    assert result.content == "Haiku Success"
    assert result.model == "claude-3-haiku-20240307"

@pytest.mark.asyncio
async def test_telemetry_and_metrics(mock_redis, mock_openai):
    client = NeuroFlowClient()
    provider = OpenAIProvider()
    provider.complete = AsyncMock(return_value=MagicMock(content="OK", model="gpt-4o-mini", input_tokens=5, output_tokens=5, cost_usd=0.01, latency_ms=50.0, finish_reason="stop"))
    client.register_provider("openai", provider)
    client.redis_client = mock_redis
    
    client.router.route = AsyncMock(return_value="gpt-4o-mini")
    
    messages = [ChatMessage(role="user", content="Hi")]
    
    tracer = trace.get_tracer(__name__)
    with patch.object(tracer, 'start_as_current_span') as mock_span:
        result = await client.chat(messages, RoutingCriteria())
        assert result.content == "OK"
        
        mock_redis.incr.assert_called_with("metrics:model:gpt-4o-mini:calls")
        mock_redis.incrbyfloat.assert_called_with("metrics:model:gpt-4o-mini:cost_usd", 0.01)

@pytest.mark.asyncio
async def test_default_model(mock_openai):
    provider = OpenAIProvider(api_key="test")
    # if model is not passed, it should use default_model
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello openai"), finish_reason="stop")]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
    
    mock_client_instance = mock_openai.return_value
    mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
    
    messages = [ChatMessage(role="user", content="Hi")]
    result = await provider.complete(messages)
    assert result.model == "llama-3.3-70b-versatile"

@pytest.mark.asyncio
async def test_embeddings_fallback(mock_openai):
    provider = OpenAIProvider(api_key="test")
    
    mock_client_instance = mock_openai.return_value
    mock_client_instance.embeddings.create = AsyncMock(side_effect=Exception("No embeddings supported"))
    
    texts = ["Test sentence"]
    
    embeddings = await provider.embed(texts)
    
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    # sentence transformers all-MiniLM-L6-v2 produces 384 dim, mock produces 1536 dim
    assert len(embeddings[0]) in (384, 1536)
