import logging
import json
import uuid
import os
from typing import List, Dict, Any

try:
    from db.pool import DatabasePool
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
    from db.pool import DatabasePool

from opentelemetry import trace
from .tracker import FineTuneTracker

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class DPOExtractor:
    def __init__(self):
        self.tracker = FineTuneTracker()

    async def extract_dpo_pairs(self, job_id: str) -> List[Dict[str, Any]]:
        pool = await DatabasePool.get_pool()
        
        with tracer.start_as_current_span("finetune.dpo") as span:
            async with pool.acquire() as conn:
                # Query self-joined pipeline_runs where query matches
                # and one has a rating >= 4 (chosen) and the other has a rating <= 2 (rejected)
                query = """
                    SELECT r1.query, r1.generation as chosen, r2.generation as rejected
                    FROM pipeline_runs r1
                    JOIN pipeline_runs r2 ON r1.query = r2.query AND r1.id != r2.id
                    WHERE (r1.metadata->>'user_rating')::int >= 4
                      AND (r2.metadata->>'user_rating')::int <= 2
                      AND r1.generation IS NOT NULL
                      AND r2.generation IS NOT NULL;
                """
                rows = await conn.fetch(query)
                
                results = []
                for r in rows:
                    results.append({
                        "prompt": r["query"],
                        "chosen": r["chosen"],
                        "rejected": r["rejected"]
                    })

                # Write DPO training data export
                self.tracker.export_to_jsonl(results, job_id, is_dpo=True)
                
                span.set_attribute("dpo_pair_count", len(results))
                return results
        
