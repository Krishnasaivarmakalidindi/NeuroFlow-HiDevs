import uuid
import logging
import asyncio
import math
from opentelemetry import trace

try:
    from db.pool import DatabasePool
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria
    from evaluation.metrics import (
        evaluate_faithfulness,
        evaluate_answer_relevance,
        evaluate_context_precision,
        evaluate_context_recall
    )
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from db.pool import DatabasePool
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria
    from evaluation.metrics import (
        evaluate_faithfulness,
        evaluate_answer_relevance,
        evaluate_context_precision,
        evaluate_context_recall
    )

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class SelfConsistencyJudge:
    def __init__(self):
        self.client = NeuroFlowClient()

    async def evaluate_consistency(self, run_id: str) -> dict:
        run_uuid = uuid.UUID(run_id) if isinstance(run_id, str) else run_id
        
        # 1. Fetch run details
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            run = await conn.fetchrow(
                "SELECT query, generation, retrieved_chunk_ids FROM pipeline_runs WHERE id = $1;",
                run_uuid
            )
            if not run:
                raise ValueError(f"Run {run_uuid} not found in pipeline_runs.")
                
            query = run["query"]
            generation = run["generation"] or ""
            retrieved_chunk_ids = run["retrieved_chunk_ids"] or []
            
            chunks = []
            if retrieved_chunk_ids:
                rows = await conn.fetch(
                    "SELECT content FROM chunks WHERE id = ANY($1::uuid[]);",
                    retrieved_chunk_ids
                )
                chunks = [r["content"] for r in rows]

        context = "\n\n".join(chunks)

        # 2. Run 3 evaluations at temperature = 0.7
        overall_scores = []
        detailed_runs = []

        with tracer.start_as_current_span("evaluation.self_consistency") as span:
            span.set_attribute("run_id", str(run_uuid))

            for i in range(3):
                # Run metrics in parallel with temperature=0.7
                f_task = evaluate_faithfulness(query, generation, context, temperature=0.7)
                r_task = evaluate_answer_relevance(query, generation, temperature=0.7)
                p_task = evaluate_context_precision(query, chunks, generation, temperature=0.7)
                c_task = evaluate_context_recall(query, chunks, generation, temperature=0.7)
                
                f, r, p, c = await asyncio.gather(f_task, r_task, p_task, c_task)
                
                overall = 0.35 * f + 0.30 * r + 0.20 * p + 0.15 * c
                overall_scores.append(overall)
                detailed_runs.append({
                    "iteration": i + 1,
                    "faithfulness": f,
                    "answer_relevance": r,
                    "context_precision": p,
                    "context_recall": c,
                    "overall": overall
                })

            # 3. Compute statistics: mean & std
            mean_score = sum(overall_scores) / len(overall_scores)
            variance = sum((x - mean_score) ** 2 for x in overall_scores) / len(overall_scores)
            std_score = math.sqrt(variance)

            high_variance = std_score > 0.2

            span.set_attribute("mean_score", mean_score)
            span.set_attribute("std_score", std_score)
            span.set_attribute("high_variance", high_variance)

            return {
                "mean": mean_score,
                "std": std_score,
                "high_variance": high_variance,
                "iterations": detailed_runs
            }
