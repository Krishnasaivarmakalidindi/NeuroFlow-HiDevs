import uuid
import json
import logging
import asyncio
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

class EvaluationJudge:
    def __init__(self):
        self.client = NeuroFlowClient()

    async def _ensure_db_schema(self, conn):
        try:
            await conn.execute(
                "ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';"
            )
        except Exception as e:
            logger.warning(f"Failed to ensure evaluations metadata column: {e}")

    async def evaluate_run(self, run_id: str, **kwargs) -> dict:
        t0 = time.perf_counter()
        run_uuid = uuid.UUID(run_id) if isinstance(run_id, str) else run_id
        
        # 1. Fetch run details
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            await self._ensure_db_schema(conn)
            
            run = await conn.fetchrow(
                "SELECT pipeline_id, query, generation, retrieved_chunk_ids, metadata FROM pipeline_runs WHERE id = $1;",
                run_uuid
            )
            if not run:
                raise ValueError(f"Run {run_uuid} not found in pipeline_runs.")
                
            pipeline_uuid = run["pipeline_id"]
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

        # 2. Run evaluations in parallel
        context = "\n\n".join(chunks)
        
        try:
            judge_model = await self.client.router.route(RoutingCriteria(task_type="evaluation"))
        except Exception as e:
            logger.warning(f"Could not route to evaluation judge model, falling back to gpt-4o: {e}")
            judge_model = "gpt-4o"

        with tracer.start_as_current_span("evaluation.judge") as span:
            span.set_attribute("run_id", str(run_uuid))
            span.set_attribute("pipeline_id", str(pipeline_uuid))
            span.set_attribute("judge_model", judge_model)
            
            # Pass routing context down
            kwargs["pipeline_id"] = str(pipeline_uuid)
            kwargs["run_id"] = str(run_uuid)
            
            # Execute metrics
            faithfulness_task = evaluate_faithfulness(query, generation, context, **kwargs)
            relevance_task = evaluate_answer_relevance(query, generation, **kwargs)
            precision_task = evaluate_context_precision(query, chunks, generation, **kwargs)
            recall_task = evaluate_context_recall(query, chunks, generation, **kwargs)
            
            f, r, p, c = await asyncio.gather(
                faithfulness_task,
                relevance_task,
                precision_task,
                recall_task
            )
            
            overall = 0.35 * f + 0.30 * r + 0.20 * p + 0.15 * c
            latency_ms = (time.perf_counter() - t0) * 1000
            
            span.set_attribute("overall_score", overall)
            span.set_attribute("latency_ms", latency_ms)
            
            span.set_attribute("scores", json.dumps({
                "faithfulness": f,
                "answer_relevance": r,
                "context_precision": p,
                "context_recall": c
            }))
            span.set_attribute("overall", overall)

            # Prometheus Metrics Update
            from monitoring import metrics
            metrics.eval_faithfulness.labels(pipeline_id=str(pipeline_uuid)).set(f)
            metrics.eval_overall.labels(pipeline_id=str(pipeline_uuid)).set(overall)

            # 3. Store results
            async with pool.acquire() as conn:
                eval_id = uuid.uuid4()
                await conn.execute(
                    """
                    INSERT INTO evaluations (id, run_id, faithfulness, answer_relevance, context_precision, context_recall, overall_score, judge_model)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8);
                    """,
                    eval_id,
                    run_uuid,
                    f,
                    r,
                    p,
                    c,
                    overall,
                    judge_model
                )
                
                # 4. Check for training pair extraction
                if overall > 0.8:
                    run_meta = run["metadata"] or {}
                    if isinstance(run_meta, str):
                        try:
                            run_meta = json.loads(run_meta)
                        except Exception:
                            run_meta = {}
                    
                    system_prompt = run_meta.get("prompt", "")
                    
                    await conn.execute(
                        """
                        INSERT INTO training_pairs (run_id, system_prompt, user_message, assistant_message, quality_score)
                        VALUES ($1, $2, $3, $4, $5);
                        """,
                        run_uuid,
                        system_prompt,
                        query,
                        generation,
                        overall
                    )
            
            return {
                "faithfulness": f,
                "answer_relevance": r,
                "context_precision": p,
                "context_recall": c,
                "overall": overall,
                "judge_model": judge_model
            }
