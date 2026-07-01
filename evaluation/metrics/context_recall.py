import re
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

def split_into_sentences(text: str) -> list:
    # A robust regex to split sentences based on punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

import time

async def evaluate_context_recall(
    query: str,
    chunks: list,
    answer: str,
    **kwargs
) -> float:
    t0 = time.perf_counter()
    client = NeuroFlowClient()
    criteria = RoutingCriteria(task_type="evaluation")

    with tracer.start_as_current_span("evaluation.context_recall") as span:
        span.set_attribute("pipeline_id", str(kwargs.get("pipeline_id", "default")))
        span.set_attribute("run_id", str(kwargs.get("run_id", "default")))
        span.set_attribute("judge_model", criteria.model or "gpt-4o")
        sentences = split_into_sentences(answer)
        if not sentences:
            span.set_attribute("score", 1.0)
            return 1.0

        context = "\n\n".join(chunks)
        if not context:
            span.set_attribute("score", 0.0)
            return 0.0

        async def verify_sentence(sentence: str) -> float:
            prompt = (
                f"Context:\n{context}\n\n"
                f"Sentence:\n{sentence}\n\n"
                "Can this sentence be attributed to the context?\nyes/no"
            )
            try:
                res = await client.chat([ChatMessage(role="user", content=prompt)], criteria, **kwargs)
                ans = res.content.lower().strip()
                words = "".join(c for c in ans if c.isalnum() or c.isspace()).split()
                if "yes" in words:
                    return 1.0
                return 0.0
            except Exception as e:
                logger.error(f"Failed to verify sentence in context_recall: {e}")
                return 0.0

        tasks = [verify_sentence(s) for s in sentences]
        results = await asyncio.gather(*tasks)

        score = sum(results) / len(sentences)

        latency_ms = (time.perf_counter() - t0) * 1000
        span.set_attribute("metric_score", score)
        span.set_attribute("latency_ms", latency_ms)
        span.set_attribute("score", score)
        span.set_attribute("sentences_count", len(sentences))
        return score
