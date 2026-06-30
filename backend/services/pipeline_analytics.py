import json
import math
import logging
from db.pool import DatabasePool

logger = logging.getLogger(__name__)

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    # Prices per 1M tokens
    prices = {
        "llama-3.3-70b-versatile": {"in": 0.59, "out": 0.79},
        "gpt-4o": {"in": 2.50, "out": 10.00},
        "gpt-4o-mini": {"in": 0.15, "out": 0.60},
        "claude-3-haiku-20240307": {"in": 0.25, "out": 1.25},
        "claude-3-opus-20240229": {"in": 15.00, "out": 75.00}
    }
    p = prices.get(model or "llama-3.3-70b-versatile", {"in": 0.59, "out": 0.79})
    cost = (input_tokens * p["in"] + output_tokens * p["out"]) / 1_000_000
    return cost

def compute_percentile(data: list, q: float) -> float:
    if not data:
        return 0.0
    try:
        import numpy as np
        return float(np.percentile(data, q))
    except ImportError:
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (q / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(sorted_data[int(k)])
        return float(sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f))

class PipelineAnalyticsService:
    async def get_analytics(self, pipeline_id: str) -> dict:
        import uuid
        try:
            pipeline_uuid = uuid.UUID(pipeline_id) if isinstance(pipeline_id, str) else pipeline_id
        except ValueError:
            raise ValueError(f"Invalid pipeline ID: {pipeline_id}")

        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            # 1. Fetch runs
            runs = await conn.fetch(
                """
                SELECT latency_ms, input_tokens, output_tokens, metadata, model_used 
                FROM pipeline_runs 
                WHERE pipeline_id = $1 AND status = 'complete';
                """,
                pipeline_uuid
            )

            # 2. Fetch evaluations
            evals = await conn.fetch(
                """
                SELECT e.faithfulness, e.answer_relevance, e.context_precision, e.context_recall, e.overall_score
                FROM evaluations e
                JOIN pipeline_runs r ON e.run_id = r.id
                WHERE r.pipeline_id = $1;
                """,
                pipeline_uuid
            )

            # 3. Queries per day
            q_days = await conn.fetch(
                """
                SELECT TO_CHAR(created_at, 'YYYY-MM-DD') as day, COUNT(*) as count 
                FROM pipeline_runs 
                WHERE pipeline_id = $1 
                GROUP BY day 
                ORDER BY day;
                """,
                pipeline_uuid
            )

        # Parse metrics
        retrieval_latencies = []
        generation_latencies = []
        costs = []

        for r in runs:
            meta = r["metadata"] or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}

            # Assume plausible fallback distributions if separate latencies not stored
            # (retrieval ~15%, generation ~85% of total latency_ms)
            total_lat = float(r["latency_ms"] or 0.0)
            ret_lat = float(meta.get("retrieval_latency_ms") or (total_lat * 0.15))
            gen_lat = float(meta.get("generation_latency_ms") or (total_lat * 0.85))

            retrieval_latencies.append(ret_lat)
            generation_latencies.append(gen_lat)

            # Calculate cost
            cost = estimate_cost(r["model_used"], r["input_tokens"] or 0, r["output_tokens"] or 0)
            costs.append(cost)

        # Compute percentiles
        p50 = compute_percentile(retrieval_latencies, 50)
        p95 = compute_percentile(retrieval_latencies, 95)
        p99 = compute_percentile(retrieval_latencies, 99)

        # Average generation latency
        avg_generation_latency = sum(generation_latencies) / len(generation_latencies) if generation_latencies else 0.0
        # Average cost per query
        avg_cost = sum(costs) / len(costs) if costs else 0.0

        # Average evaluations
        avg_evals = {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "overall": 0.0
        }
        
        if evals:
            avg_evals["faithfulness"] = sum(e["faithfulness"] or 0.0 for e in evals) / len(evals)
            avg_evals["answer_relevance"] = sum(e["answer_relevance"] or 0.0 for e in evals) / len(evals)
            avg_evals["context_precision"] = sum(e["context_precision"] or 0.0 for e in evals) / len(evals)
            avg_evals["context_recall"] = sum(e["context_recall"] or 0.0 for e in evals) / len(evals)
            avg_evals["overall"] = sum(e["overall_score"] or 0.0 for e in evals) / len(evals)

        # Queries per day list
        queries_per_day = [{"day": qd["day"], "count": qd["count"]} for qd in q_days]

        return {
            "retrieval_latency": {
                "p50": round(p50, 2),
                "p95": round(p95, 2),
                "p99": round(p99, 2)
            },
            "generation_latency": round(avg_generation_latency, 2),
            "evaluation": {
                "faithfulness": round(avg_evals["faithfulness"], 4),
                "answer_relevance": round(avg_evals["answer_relevance"], 4),
                "context_precision": round(avg_evals["context_precision"], 4),
                "context_recall": round(avg_evals["context_recall"], 4),
                "overall": round(avg_evals["overall"], 4)
            },
            "cost_per_query": round(avg_cost, 6),
            "queries_per_day": queries_per_day
        }
