"""
fusion.py — Reciprocal Rank Fusion (RRF).

RRF is a simple, parameter-light method for combining ranked lists from
heterogeneous retrievers.  Each document's score is:

    score(d) = Σ  1 / (k + rank_i(d))

where k=60 (default) is a smoothing constant and rank_i is the 1-based
position in list i.  Documents that appear in more lists, or higher up,
accumulate a larger fused score.

Reference: Cormack, Clarke & Buettcher (SIGIR 2009)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from opentelemetry import trace

try:
    from pipelines.retrieval.models import RetrievalResult
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from pipelines.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def reciprocal_rank_fusion(
    lists: List[List[RetrievalResult]],
    k: int = 60,
) -> List[RetrievalResult]:
    """Combine multiple ranked result lists with RRF.

    Parameters
    ----------
    lists:
        Each element is an ordered list of :class:`RetrievalResult` objects
        (highest-ranked first).
    k:
        RRF smoothing constant (default 60 per the original paper).

    Returns
    -------
    A single list of :class:`RetrievalResult` sorted by descending RRF score.
    The ``score`` field on each result is replaced by the RRF fused score.
    """
    with tracer.start_as_current_span("retrieval.rrf") as span:
        span.set_attribute("num_lists", len(lists))
        span.set_attribute("rrf_k", k)

        # chunk_id → accumulated RRF score
        fused_scores: Dict[str, float] = {}
        # chunk_id → best representative RetrievalResult (for metadata)
        best_result: Dict[str, RetrievalResult] = {}

        for ranked_list in lists:
            for rank, result in enumerate(ranked_list, start=1):
                cid = result.chunk_id
                fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank)
                # Keep the result object with the best original score for provenance
                if cid not in best_result or result.score > best_result[cid].score:
                    best_result[cid] = result

        # Build output list with RRF scores
        fused: List[RetrievalResult] = []
        for cid, rrf_score in fused_scores.items():
            r = best_result[cid]
            fused.append(
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    score=rrf_score,
                    source=r.source,
                    metadata=r.metadata,
                    document_id=r.document_id,
                    chunk_index=r.chunk_index,
                    token_count=r.token_count,
                )
            )

        fused.sort(key=lambda x: x.score, reverse=True)

        span.set_attribute("unique_chunks", len(fused))
        logger.debug("RRF merged %d lists → %d unique chunks", len(lists), len(fused))

        return fused
