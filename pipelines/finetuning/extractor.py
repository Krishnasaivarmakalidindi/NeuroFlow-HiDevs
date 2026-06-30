import logging
import json
from typing import List, Dict, Any

try:
    from db.pool import DatabasePool
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
    from db.pool import DatabasePool

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class TrainingDataExtractor:
    async def extract_pairs(self) -> List[Dict[str, Any]]:
        pool = await DatabasePool.get_pool()
        
        with tracer.start_as_current_span("finetune.extract") as span:
            async with pool.acquire() as conn:
                # Query training pairs with filtering
                # Note: user_rating is stored inside pipeline_runs.metadata JSONB.
                # Faithfulness is stored inside evaluations.faithfulness.
                query = """
                    SELECT tp.id, tp.run_id, tp.quality_score, tp.system_prompt, tp.user_message, tp.assistant_message,
                           pr.metadata->>'user_rating' as user_rating,
                           ev.faithfulness
                    FROM training_pairs tp
                    JOIN pipeline_runs pr ON tp.run_id = pr.id
                    LEFT JOIN evaluations ev ON tp.run_id = ev.run_id
                    WHERE tp.quality_score >= 0.82
                      AND tp.included_in_job IS NULL
                      AND (
                          pr.metadata->>'user_rating' IS NULL 
                          OR (pr.metadata->>'user_rating')::int >= 4
                      );
                """
                rows = await conn.fetch(query)
                
                results = []
                for r in rows:
                    results.append({
                        "id": str(r["id"]),
                        "run_id": str(r["run_id"]),
                        "quality_score": float(r["quality_score"]),
                        "system_prompt": r["system_prompt"],
                        "user_message": r["user_message"],
                        "assistant_message": r["assistant_message"],
                        "user_rating": int(r["user_rating"]) if r["user_rating"] else None,
                        "faithfulness": float(r["faithfulness"]) if r["faithfulness"] is not None else None
                    })
                
                span.set_attribute("extracted_count", len(results))
                return results

    def format_as_chat_messages(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": pair["system_prompt"]},
                {"role": "user", "content": pair["user_message"]},
                {"role": "assistant", "content": pair["assistant_message"]}
            ]
        }
