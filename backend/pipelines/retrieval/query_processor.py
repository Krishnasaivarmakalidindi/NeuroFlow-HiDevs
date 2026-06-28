"""
query_processor.py — Query analysis and enrichment.

Three responsibilities:
  1. expand_query()           — generate semantically-related sub-queries
  2. extract_metadata_filters() — pull structured filters from natural language
  3. classify_query()         — categorise intent (factual / analytical / …)

All LLM calls go through the existing OpenAIProvider so that retries, cost
tracking and tracing come for free.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from opentelemetry import trace

try:
    from config import settings
    from providers.openai_provider import OpenAIProvider
    from providers.base import ChatMessage
    from pipelines.retrieval.models import ProcessedQuery, QueryType
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from config import settings
    from providers.openai_provider import OpenAIProvider
    from providers.base import ChatMessage
    from pipelines.retrieval.models import ProcessedQuery, QueryType

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# ---------------------------------------------------------------------------
# System prompts (kept concise so small models respond reliably)
# ---------------------------------------------------------------------------

_EXPANSION_SYSTEM = """\
You are a query-expansion assistant. Given a user search query, produce
3–5 alternative phrasings that are semantically equivalent but use different
vocabulary or framing. Return ONLY a JSON array of strings — no markdown, no
commentary. Example:
  Input : "how does attention work in transformers"
  Output: ["explain self attention", "transformer attention weights",
           "attention mechanism calculation", "scaled dot-product attention",
           "multi-head attention explained"]
"""

_FILTER_SYSTEM = """\
You are a metadata-filter extractor. Given a user query, extract any explicit
filters the user mentions (year, topic, author, file type, etc.). Return ONLY
a compact JSON object. If no filters are found return {}. Examples:
  "show climate documents from 2023" → {"year": 2023, "topic": "climate"}
  "find papers by Smith on neural nets" → {"author": "Smith", "topic": "neural nets"}
  "what is HNSW?" → {}
"""

_CLASSIFY_SYSTEM = """\
Classify the user query into exactly one category. Return ONLY the label.
Categories:
  factual      — asks for a specific fact or definition
  analytical   — asks for analysis, reasoning or explanation
  comparative  — asks to compare or contrast two or more things
  procedural   — asks how to do something step-by-step
"""


class QueryProcessor:
    """Enrich a raw user query before retrieval begins."""

    def __init__(self, provider: OpenAIProvider | None = None):
        self._provider = provider or OpenAIProvider()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, query: str) -> ProcessedQuery:
        """Run all enrichment steps and return a :class:`ProcessedQuery`."""
        with tracer.start_as_current_span("retrieval.query") as span:
            span.set_attribute("query", query)

            expanded, filters, qtype = await self._run_all(query)

            span.set_attribute("query_type", qtype.value)
            span.set_attribute("expansions", len(expanded))

            return ProcessedQuery(
                original=query,
                expanded=expanded,
                metadata_filters=filters,
                query_type=qtype,
            )

    async def expand_query(self, query: str) -> List[str]:
        """Return 3–5 semantically-equivalent query variants."""
        messages = [
            ChatMessage(role="system", content=_EXPANSION_SYSTEM),
            ChatMessage(role="user", content=query),
        ]
        try:
            result = await self._provider.complete(
                messages,
                max_tokens=256,
                temperature=0.3,
            )
            raw = result.content.strip()
            # Strip accidental markdown fences
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("` ")
            expansions: List[str] = json.loads(raw)
            if not isinstance(expansions, list):
                raise ValueError("Expected JSON array")
            return [str(e) for e in expansions[:5]]
        except Exception as exc:
            logger.warning("Query expansion failed (%s); using original only", exc)
            return []

    async def extract_metadata_filters(self, query: str) -> Dict[str, Any]:
        """Return a dict of extracted metadata filters, or {} if none found."""
        messages = [
            ChatMessage(role="system", content=_FILTER_SYSTEM),
            ChatMessage(role="user", content=query),
        ]
        try:
            result = await self._provider.complete(
                messages,
                max_tokens=128,
                temperature=0.0,
            )
            raw = result.content.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("` ")
            filters = json.loads(raw)
            if not isinstance(filters, dict):
                raise ValueError("Expected JSON object")
            return filters
        except Exception as exc:
            logger.warning("Metadata extraction failed (%s); using no filters", exc)
            return {}

    async def classify_query(self, query: str) -> QueryType:
        """Classify the query intent."""
        messages = [
            ChatMessage(role="system", content=_CLASSIFY_SYSTEM),
            ChatMessage(role="user", content=query),
        ]
        try:
            result = await self._provider.complete(
                messages,
                max_tokens=16,
                temperature=0.0,
            )
            label = result.content.strip().lower()
            return QueryType(label)
        except (ValueError, Exception) as exc:
            logger.warning("Query classification failed (%s); defaulting to factual", exc)
            return QueryType.FACTUAL

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_all(self, query: str):
        """Run all enrichment steps; short-circuit gracefully on errors."""
        import asyncio

        expanded_task = asyncio.create_task(self.expand_query(query))
        filters_task = asyncio.create_task(self.extract_metadata_filters(query))
        classify_task = asyncio.create_task(self.classify_query(query))

        expanded = await expanded_task
        filters = await filters_task
        qtype = await classify_task

        return expanded, filters, qtype
