import re
import json
import time
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

def parse_json_array(text: str) -> list:
    text = text.strip()
    # Strip markdown block formatting if present
    match = re.search(r"\[\s*.*?\s*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract lines that look like claims
        lines = [line.strip('"-*• \t[]{}') for line in text.split('\n') if line.strip()]
        return [l for l in lines if l]

async def evaluate_faithfulness(
    query: str,
    answer: str,
    context: str,
    **kwargs
) -> float:
    if context == "":
        return 0.0

    t0 = time.perf_counter()
    client = NeuroFlowClient()
    criteria = RoutingCriteria(task_type="evaluation")

    with tracer.start_as_current_span("evaluation.faithfulness") as span:
        span.set_attribute("pipeline_id", str(kwargs.get("pipeline_id", "default")))
        span.set_attribute("run_id", str(kwargs.get("run_id", "default")))
        span.set_attribute("judge_model", criteria.model or "gpt-4o")

        # Step 1: Extract claims
        extract_prompt = (
            "Extract all factual claims from the answer.\n"
            "Return JSON array only.\n\n"
            f"Answer:\n{answer}"
        )
        
        try:
            res = await client.chat([ChatMessage(role="user", content=extract_prompt)], criteria, **kwargs)
            claims = parse_json_array(res.content)
        except Exception as e:
            logger.error(f"Failed to extract claims in faithfulness metric: {e}")
            span.record_exception(e)
            span.set_attribute("metric_score", 0.0)
            span.set_attribute("score", 0.0)
            return 0.0

        if not claims:
            span.set_attribute("metric_score", 1.0)
            span.set_attribute("score", 1.0)
            return 1.0

        # Step 2: Verify each claim in parallel
        async def verify_claim(claim: str) -> float:
            verify_prompt = (
                f"Context:\n{context}\n\n"
                f"Claim:\n{claim}\n\n"
                "Answer:\nyes\nno\npartial"
            )
            try:
                res_verify = await client.chat([ChatMessage(role="user", content=verify_prompt)], criteria, **kwargs)
                ans = res_verify.content.lower().strip()
                words = "".join(c for c in ans if c.isalnum() or c.isspace()).split()
                if "yes" in words:
                    return 1.0
                elif "partial" in words:
                    return 0.5
                else:
                    return 0.0
            except Exception as e:
                logger.error(f"Failed to verify claim '{claim}': {e}")
                return 0.0

        tasks = [verify_claim(c) for c in claims]
        scores = await asyncio.gather(*tasks)
        
        score = sum(scores) / len(claims)
        latency_ms = (time.perf_counter() - t0) * 1000
        
        span.set_attribute("metric_score", score)
        span.set_attribute("latency_ms", latency_ms)
        span.set_attribute("score", score)
        span.set_attribute("claims_count", len(claims))
        return score
