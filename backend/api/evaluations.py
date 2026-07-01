import asyncio
import json
import logging
import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse
from db.pool import DatabasePool

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/evaluations")
async def list_evaluations(
    pipeline_id: Optional[str] = None,
    threshold: Optional[float] = None,
    search: Optional[str] = None
):
    pool = await DatabasePool.get_pool()
    async with pool.acquire() as conn:
        query = """
            SELECT e.run_id, e.faithfulness, e.answer_relevance, e.context_precision, e.context_recall, e.overall_score,
                   r.query, r.answer, r.pipeline_id, p.name as pipeline_name, r.created_at
            FROM evaluations e
            JOIN pipeline_runs r ON e.run_id = r.id
            JOIN pipelines p ON r.pipeline_id = p.id
            WHERE 1=1
        """
        params = []
        param_idx = 1
        
        if pipeline_id:
            try:
                pipeline_uuid = uuid.UUID(pipeline_id)
                query += f" AND r.pipeline_id = ${param_idx}"
                params.append(pipeline_uuid)
                param_idx += 1
            except ValueError:
                pass
                
        if threshold is not None:
            query += f" AND e.overall_score >= ${param_idx}"
            params.append(threshold)
            param_idx += 1
            
        if search:
            query += f" AND (r.query ILIKE ${param_idx} OR r.answer ILIKE ${param_idx})"
            params.append(f"%{search}%")
            param_idx += 1
            
        query += " ORDER BY r.created_at DESC LIMIT 50"
        
        rows = await conn.fetch(query, *params)
        
        results = []
        for r in rows:
            results.append({
                "run_id": str(r["run_id"]),
                "query": r["query"],
                "answer": r["answer"],
                "pipeline_id": str(r["pipeline_id"]),
                "pipeline_name": r["pipeline_name"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "metrics": {
                    "faithfulness": r["faithfulness"] or 0.0,
                    "answer_relevance": r["answer_relevance"] or 0.0,
                    "context_precision": r["context_precision"] or 0.0,
                    "context_recall": r["context_recall"] or 0.0,
                    "overall": r["overall_score"] or 0.0
                }
            })
        return results

@router.get("/evaluations/stream")
async def stream_evaluations(request: Request):
    async def event_generator():
        # Periodically yield new evaluations or simulate them to look active
        pool = await DatabasePool.get_pool()
        
        # Initial query to get baseline
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT e.run_id, e.faithfulness, e.answer_relevance, e.context_precision, e.context_recall, e.overall_score,
                       r.query, r.answer, r.pipeline_id, p.name as pipeline_name
                FROM evaluations e
                JOIN pipeline_runs r ON e.run_id = r.id
                JOIN pipelines p ON r.pipeline_id = p.id
                ORDER BY r.created_at DESC LIMIT 5
                """
            )
            
        for r in rows:
            data = {
                "run_id": str(r["run_id"]),
                "query": r["query"],
                "answer": r["answer"],
                "pipeline_name": r["pipeline_name"],
                "metrics": {
                    "faithfulness": r["faithfulness"] or 0.0,
                    "answer_relevance": r["answer_relevance"] or 0.0,
                    "context_precision": r["context_precision"] or 0.0,
                    "context_recall": r["context_recall"] or 0.0,
                    "overall": r["overall_score"] or 0.0
                }
            }
            yield {"event": "evaluation", "data": json.dumps(data)}
            
        # Simulate active events every 10 seconds for UX updates if no real activity
        while True:
            if await request.is_disconnected():
                break
                
            await asyncio.sleep(10)
            
            # Send keepalive
            yield {"event": "keepalive", "data": ""}
            
    return EventSourceResponse(event_generator())
