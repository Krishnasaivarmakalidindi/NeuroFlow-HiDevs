import asyncio
import json
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app
from pipelines.generation.prompt_builder import PromptBuilder
from pipelines.generation.citations import CitationParser
from pipelines.generation.generator import RAGGenerator, StreamingCoTStripper
from pipelines.generation.models import Citation, GenerationResult
from pipelines.retrieval.models import RetrievalResult

# ---------------------------------------------------------------------------
# 1. Prompt Builder Tests
# ---------------------------------------------------------------------------

def test_prompt_builder():
    context = "Context info here."
    query = "What is X?"

    # Factual
    prompt = PromptBuilder.build(query, context, "factual")
    assert "Provide a direct concise answer." in prompt
    assert "cite all of them." in prompt
    assert "<context>\n\nContext info here.\n\n</context>" in prompt
    assert "Question:\nWhat is X?" in prompt

    # Analytical
    prompt = PromptBuilder.build(query, context, "analytical")
    assert "Analyze and synthesize." in prompt
    assert "Identify agreements\nand contradictions." in prompt
    assert "<think>" in prompt

    # Comparative
    prompt = PromptBuilder.build(query, context, "comparative")
    assert "Provide structured comparison." in prompt
    assert "Use tables if useful." in prompt
    assert "<think>" in prompt

    # Procedural
    prompt = PromptBuilder.build(query, context, "procedural")
    assert "Provide numbered steps." in prompt
    assert "Every step must contain citations." in prompt
    assert "<think>" not in prompt


# ---------------------------------------------------------------------------
# 2. Citation Extraction & Hallucinated Citations Tests
# ---------------------------------------------------------------------------

def test_citation_extraction():
    sources = [
        RetrievalResult(
            chunk_id="chunk_1",
            content="NeuroFlow is cool.",
            score=0.9,
            source="paper.pdf",
            metadata={"page_number": 4}
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            content="It runs fast.",
            score=0.8,
            source="doc.docx",
            metadata={"page_number": None},
            chunk_index=2
        )
    ]

    text = "According to [Source 1], NeuroFlow is cool. Also [Source 2] says it is fast. Finally, [Source 5] is hallucinated."

    citations = CitationParser.parse(text, sources)

    assert len(citations) == 3

    # Source 1
    assert citations[0].reference == "[Source 1]"
    assert citations[0].chunk_id == "chunk_1"
    assert citations[0].document_name == "paper.pdf"
    assert citations[0].page_number == 4
    assert citations[0].invalid_citation is False
    assert citations[0].content_preview.startswith("NeuroFlow")

    # Source 2
    assert citations[1].reference == "[Source 2]"
    assert citations[1].chunk_id == "chunk_2"
    assert citations[1].document_name == "doc.docx"
    assert citations[1].page_number == 2
    assert citations[1].invalid_citation is False

    # Source 5 (Hallucinated)
    assert citations[2].reference == "[Source 5]"
    assert citations[2].chunk_id == ""
    assert citations[2].document_name == ""
    assert citations[2].page_number is None
    assert citations[2].invalid_citation is True


# ---------------------------------------------------------------------------
# 3. Hidden Reasoning Stripping Tests
# ---------------------------------------------------------------------------

def test_cot_stripper():
    # Test case 1: normal text with no <think> block
    stripper = StreamingCoTStripper()
    res = stripper.process_chunk("Hello World")
    assert res == "Hello World"
    assert stripper.think_content == ""

    # Test case 2: text with <think> block
    stripper = StreamingCoTStripper()
    c1 = stripper.process_chunk("<")
    c2 = stripper.process_chunk("th")
    c3 = stripper.process_chunk("ink>")
    c4 = stripper.process_chunk("reasoning ")
    c5 = stripper.process_chunk("here")
    c6 = stripper.process_chunk("</")
    c7 = stripper.process_chunk("think>final answer")

    assert c1 == ""
    assert c2 == ""
    assert c3 == ""
    assert c4 == ""
    assert c5 == ""
    assert c6 == ""
    assert c7 == "final answer"
    assert stripper.think_content.strip() == "reasoning here"

    # Test case 3: partial match and fallback
    stripper = StreamingCoTStripper()
    assert stripper.process_chunk("<t") == ""
    assert stripper.process_chunk("x") == "<tx"  # fails prefix match, flushes buffer


# ---------------------------------------------------------------------------
# 4. Token Accumulation Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_token_accumulation():
    # Verify generator accumulates tokens and calculates counts using tiktoken
    gen = RAGGenerator()
    prompt_tokens = len(gen.tokenizer.encode("Hello"))
    assert prompt_tokens > 0


# ---------------------------------------------------------------------------
# 5. Pipeline runs Updates & Async Evaluation Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_db_and_redis():
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    mock_redis = AsyncMock()

    # Create mock retrieval pipeline output
    mock_retrieval_output = {
        "context": "Context info.",
        "chunks_used": ["00000000-0000-0000-0000-000000000001"],
        "sources": [{"chunk_id": "00000000-0000-0000-0000-000000000001", "source": "doc.pdf"}],
        "reranked": [RetrievalResult(chunk_id="00000000-0000-0000-0000-000000000001", content="Context info.", score=0.9, source="doc.pdf")],
        "pipeline_meta": {"query_type": "factual"}
    }

    mock_provider = MagicMock()
    async def mock_stream(*args, **kwargs):
        async def gen():
            yield "Hello "
            yield "World"
        return gen()

    with patch("pipelines.generation.generator.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)), \
         patch("pipelines.generation.generator.get_redis_pool", AsyncMock(return_value=mock_redis)), \
         patch("pipelines.generation.generator.RetrievalPipeline.run", AsyncMock(return_value=mock_retrieval_output)), \
         patch("providers.client.NeuroFlowClient.chat", AsyncMock(side_effect=mock_stream)):
         
        gen = RAGGenerator()
        res = await gen.generate("test query", str(uuid.uuid4()))

        assert res.answer == "Hello World"
        assert res.input_tokens > 0
        assert res.output_tokens > 0

        # Assert DB insertion was triggered
        assert mock_db_conn.execute.call_count >= 2  # INSERT and UPDATE
        # Assert Redis arq job enqueued
        mock_redis.enqueue_job.assert_called_once()
        assert mock_redis.enqueue_job.call_args[0][0] == "evaluate_run"


# ---------------------------------------------------------------------------
# 6. Streaming & SSE endpoint Tests (SSE, Keepalive, Stripping)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sse_endpoint_stream():
    client = TestClient(app)
    
    mock_redis = AsyncMock()
    mock_db_conn = AsyncMock()
    mock_db_pool = MagicMock()
    mock_db_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db_conn),
        __aexit__=AsyncMock(return_value=None),
    ))

    # Mock provider response stream containing a <think> tag
    async def mock_stream(*args, **kwargs):
        async def gen():
            yield "<think>reasoning</think>"
            yield "Actual answer"
        return gen()

    mock_retrieval_output = {
        "context": "Context info.",
        "chunks_used": ["00000000-0000-0000-0000-000000000001"],
        "sources": [{"chunk_id": "00000000-0000-0000-0000-000000000001", "source": "doc.pdf"}],
        "reranked": [RetrievalResult(chunk_id="00000000-0000-0000-0000-000000000001", content="Context info.", score=0.9, source="doc.pdf")],
        "pipeline_meta": {"query_type": "analytical"}
    }

    with patch("pipelines.generation.generator.DatabasePool.get_pool", AsyncMock(return_value=mock_db_pool)), \
         patch("pipelines.generation.generator.get_redis_pool", AsyncMock(return_value=mock_redis)), \
         patch("pipelines.generation.generator.RetrievalPipeline.run", AsyncMock(return_value=mock_retrieval_output)), \
         patch("providers.client.NeuroFlowClient.chat", AsyncMock(side_effect=mock_stream)):

        # Post stream=True
        resp = client.post("/query", json={
            "query": "explain attention",
            "pipeline_id": str(uuid.uuid4()),
            "stream": True
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        run_id = data["run_id"]

        # Connect to stream
        with client.stream("GET", f"/query/{run_id}/stream") as stream_resp:
            assert stream_resp.status_code == 200
            
            # Parse events
            events = []
            for line in stream_resp.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

            assert len(events) >= 4
            assert events[0]["type"] == "retrieval_start"
            assert events[1]["type"] == "retrieval_complete"
            
            # Token event should contain clean tokens only
            token_events = [e for e in events if e["type"] == "token"]
            assert len(token_events) > 0
            # Clean tokens should NOT contain "<think>reasoning</think>"
            for te in token_events:
                assert "<think>" not in te["delta"]
                assert "reasoning" not in te["delta"]
            
            # Done event
            done_event = events[-1]
            assert done_event["type"] == "done"
            assert done_event["run_id"] == run_id


@pytest.mark.asyncio
async def test_sse_keepalive():
    # Test keepalive event generation on timeout
    from api.query import get_query_stream, stream_queues
    run_id = "test_keepalive_run_id"
    queue = asyncio.Queue()
    stream_queues[run_id] = queue

    # Mock queue.get to return None (non-coroutine) to avoid unawaited coroutine warning
    queue.get = MagicMock(return_value=None)

    # Put a keepalive timeout trigger
    with patch("api.query.asyncio.wait_for", side_effect=[asyncio.TimeoutError, None]):
        resp = await get_query_stream(run_id)
        # Consume the stream response generator
        events = []
        async for item in resp.body_iterator:
            if isinstance(item, dict):
                data_str = item.get("data", "")
                events.append(json.loads(data_str))
            elif isinstance(item, bytes):
                item_str = item.decode("utf-8")
                if item_str.startswith("data: "):
                    events.append(json.loads(item_str[6:]))
            elif isinstance(item, str):
                if item.startswith("data: "):
                    events.append(json.loads(item[6:]))
        
        assert len(events) == 1
        assert events[0]["type"] == "keepalive"
