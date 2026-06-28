"""
context_assembler.py — Token-budget-aware context builder.

Assembles a formatted context string from a ranked list of
:class:`RetrievalResult` objects.  Key guarantees:

  * Never exceeds ``token_budget`` tokens (default 4 000).
  * Never truncates mid-sentence — if a chunk does not fit entirely, it
    is skipped (not partially included).
  * Sources are numbered and formatted for easy citation.

Output schema::

    {
        "context":      str,          # ready-to-insert prompt string
        "chunks_used":  List[str],    # chunk_ids included
        "total_tokens": int,
        "sources":      List[dict],   # [{source, chunk_id, chunk_index}, …]
    }
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from opentelemetry import trace

try:
    from pipelines.retrieval.models import RetrievalResult, RetrievalConfig
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from pipelines.retrieval.models import RetrievalResult, RetrievalConfig

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def _load_tokenizer(model: str = "cl100k_base"):
    """Load a tiktoken tokenizer, with a fallback to word-splitting."""
    try:
        import tiktoken

        return tiktoken.get_encoding(model)
    except Exception:
        logger.warning("tiktoken unavailable — using word-count fallback")
        return None


def _count_tokens(text: str, enc) -> int:
    if enc is None:
        return len(text.split())
    return len(enc.encode(text))


def _trim_to_sentence_boundary(text: str) -> str:
    """
    Return ``text`` unchanged (we never truncate mid-sentence).
    This helper exists as a hook for future partial-inclusion logic.
    """
    return text


def _format_chunk(source_num: int, result: RetrievalResult) -> str:
    """Format a single chunk in the citation style required by the task."""
    page = result.metadata.get("page_number", result.chunk_index)
    page_str = f"\npage {page}" if page is not None else ""
    return (
        f"[Source {source_num}]\n"
        f"{result.source}{page_str}\n\n"
        f"{result.content.strip()}"
    )


class ContextAssembler:
    """Pack chunks into a prompt context, honouring the token budget."""

    def __init__(
        self,
        token_budget: int = 4_000,
        encoding_name: str = "cl100k_base",
    ):
        self._budget = token_budget
        self._enc = _load_tokenizer(encoding_name)

    def assemble(
        self,
        results: List[RetrievalResult],
        separator: str = "\n\n---\n\n",
    ) -> Dict[str, Any]:
        """Pack as many chunks as fit within the token budget.

        Parameters
        ----------
        results:
            Ranked list (best first).
        separator:
            String inserted between consecutive sources.

        Returns
        -------
        dict with keys ``context``, ``chunks_used``, ``total_tokens``,
        ``sources``.
        """
        with tracer.start_as_current_span("retrieval.context") as span:
            span.set_attribute("candidates", len(results))
            span.set_attribute("token_budget", self._budget)

            sep_tokens = _count_tokens(separator, self._enc)
            used_tokens = 0
            parts: List[str] = []
            chunks_used: List[str] = []
            sources: List[Dict[str, Any]] = []

            for idx, result in enumerate(results, start=1):
                chunk_text = _format_chunk(idx, result)
                chunk_tokens = _count_tokens(chunk_text, self._enc)

                # Account for separator (after the first chunk)
                extra = sep_tokens if parts else 0
                if used_tokens + chunk_tokens + extra > self._budget:
                    logger.debug(
                        "Chunk %s skipped — would exceed budget (%d + %d > %d)",
                        result.chunk_id,
                        used_tokens,
                        chunk_tokens + extra,
                        self._budget,
                    )
                    continue  # Skip — never truncate mid-sentence

                parts.append(chunk_text)
                used_tokens += chunk_tokens + extra
                chunks_used.append(result.chunk_id)
                sources.append(
                    {
                        "source_num": idx,
                        "source": result.source,
                        "chunk_id": result.chunk_id,
                        "chunk_index": result.chunk_index,
                        "score": result.score,
                        "metadata": result.metadata,
                    }
                )

            context = separator.join(parts)

            span.set_attribute("chunks_used", len(chunks_used))
            span.set_attribute("total_tokens", used_tokens)

            logger.debug(
                "Context assembled: %d/%d chunks, %d tokens",
                len(chunks_used),
                len(results),
                used_tokens,
            )

            return {
                "context": context,
                "chunks_used": chunks_used,
                "total_tokens": used_tokens,
                "sources": sources,
            }
