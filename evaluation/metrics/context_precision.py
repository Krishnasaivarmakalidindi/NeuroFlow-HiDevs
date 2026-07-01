import logging
import asyncio
from opentelemetry import trace

try:
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria
    from providers.base import ChatMessage
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria
    from providers.base import ChatMessage

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

import time

async def evaluate_context_precision(
    query: str,
    chunks: list,
    answer: str,
    **kwargs
) -> float:
    if not chunks:
        return 0.0

    t0 = time.perf_counter()
    client = NeuroFlowClient()
    criteria = RoutingCriteria(task_type="evaluation")

    with tracer.start_as_current_span("evaluation.context_precision") as span:
        span.set_attribute("pipeline_id", str(kwargs.get("pipeline_id", "default")))
        span.set_attribute("run_id", str(kwargs.get("run_id", "default")))
        span.set_attribute("judge_model", criteria.model or "gpt-4o")
        async def evaluate_chunk_usefulness(chunk: str) -> float:
            prompt = (
                f"Query:\n{query}\n\n"
                f"Answer:\n{answer}\n\n"
                f"Chunk:\n{chunk}\n\n"
                "Was this chunk useful for generating the answer?\nyes/no"
            )
            try:
                res = await client.chat([ChatMessage(role="user", content=prompt)], criteria, **kwargs)
                ans = res.content.lower().strip()
                words = "".join(c for c in ans if c.isalnum() or c.isspace()).split()
                if "yes" in words:
                    return 1.0
                return 0.0
            except Exception as e:
                logger.error(f"Failed to check chunk usefulness: {e}")
                return 0.0

        tasks = [evaluate_chunk_usefulness(c) for c in chunks]
        usefulness = await asyncio.gather(*tasks)

        # Compute precision score: sum( usefulness[i] * (1 / i) ) / sum( 1 / i )
        # where i is 1-indexed rank
        numerator = 0.0
        denominator = 0.0
        for idx, val in enumerate(usefulness, start=1):
            weight = 1.0 / idx
            numerator += val * weight
            denominator += weight

        score = numerator / denominator if denominator > 0 else 0.0

        latency_ms = (time.perf_counter() - t0) * 1000
        span.set_attribute("metric_score", score)
        span.set_attribute("latency_ms", latency_ms)
        span.set_attribute("score", score)
        span.set_attribute("chunks_count", len(chunks))
        return score
