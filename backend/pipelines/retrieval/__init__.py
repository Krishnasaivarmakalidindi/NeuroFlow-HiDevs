"""
NeuroFlow Production Retrieval Pipeline.

Exports the top-level pipeline and all public types so that callers need only:

    from pipelines.retrieval import RetrievalPipeline, RetrievalResult
"""

from .models import RetrievalResult, RetrievalConfig
from .pipeline import RetrievalPipeline

__all__ = [
    "RetrievalResult",
    "RetrievalConfig",
    "RetrievalPipeline",
]
