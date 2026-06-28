"""
hyde.py — Hypothetical Document Embeddings (HyDE).

HyDE (Gao et al., 2022) improves dense retrieval for questions whose
surface form differs greatly from the way answers are written in the corpus.

Instead of embedding the bare question, we:
  1. Ask the LLM to write a short, plausible *answer* to the question.
  2. Embed that hypothetical answer.
  3. Use the resulting vector for pgvector retrieval.

Because the hypothetical answer uses the same vocabulary and register as real
documents, the embedding sits closer to relevant passages in latent space.

We also expose a comparison helper so the evaluation script can measure
normal retrieval vs. HyDE retrieval side-by-side.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from opentelemetry import trace

try:
    from providers.openai_provider import OpenAIProvider
    from providers.base import ChatMessage
    from pipelines.retrieval.models import RetrievalConfig
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from providers.openai_provider import OpenAIProvider
    from providers.base import ChatMessage
    from pipelines.retrieval.models import RetrievalConfig

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

_HYDE_SYSTEM = """\
You are a helpful assistant. The user will give you a question. Write a
concise, factual paragraph (80–150 words) that directly answers the question.
Write as if you are the relevant document — use the same technical vocabulary
the answer would appear in. Do NOT explain what you are doing; just write the
hypothetical answer.
"""


class HyDEGenerator:
    """Generate a hypothetical document embedding for a query."""

    def __init__(
        self,
        provider: Optional[OpenAIProvider] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self._provider = provider or OpenAIProvider()
        self._config = config or RetrievalConfig()

    async def generate_hypothetical_answer(self, query: str) -> str:
        """Generate a plausible answer paragraph for *query*."""
        with tracer.start_as_current_span("retrieval.hyde.generate") as span:
            span.set_attribute("query", query)
            messages = [
                ChatMessage(role="system", content=_HYDE_SYSTEM),
                ChatMessage(role="user", content=query),
            ]
            try:
                result = await self._provider.complete(
                    messages,
                    max_tokens=256,
                    temperature=0.7,
                )
                answer = result.content.strip()
                span.set_attribute("answer_length", len(answer))
                logger.debug("HyDE generated answer (%d chars)", len(answer))
                return answer
            except Exception as exc:
                logger.error("HyDE generation failed: %s", exc)
                span.record_exception(exc)
                # Fall back to the original query so retrieval still proceeds
                return query

    async def get_hyde_embedding(self, query: str) -> List[float]:
        """Return the embedding of a hypothetical answer for *query*.

        This vector is used *instead of* the query embedding in dense
        retrieval when HyDE is enabled.
        """
        with tracer.start_as_current_span("retrieval.hyde") as span:
            span.set_attribute("query", query)

            hypothetical_answer = await self.generate_hypothetical_answer(query)
            embeddings = await self._provider.embed(
                [hypothetical_answer],
                model=self._config.embedding_model,
            )

            if embeddings:
                span.set_attribute("embedding_dim", len(embeddings[0]))
                return embeddings[0]

            # Should not happen, but degrade gracefully
            logger.error("HyDE embedding returned empty — falling back to zero vector")
            return [0.0] * 1536
