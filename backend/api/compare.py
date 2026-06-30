import logging
import asyncio
import uuid
import time
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from opentelemetry import trace

try:
    from db.pool import DatabasePool
    from pipelines.generation.generator import RAGGenerator
    from services.pipeline_manager import PipelineManager
    from evaluation.judge import EvaluationJudge
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from db.pool import DatabasePool
    from pipelines.generation.generator import RAGGenerator
    from services.pipeline_manager import PipelineManager
    from evaluation.judge import EvaluationJudge

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

router = APIRouter()
rag_generator = RAGGenerator()
manager = PipelineManager()
judge = EvaluationJudge()

class CompareRequest(BaseModel):
    query: str
    pipeline_a_id: str
    pipeline_b_id: str

async def run_pipeline_comparison(pipeline_id: str, query: str) -> dict:
    run_uuid = uuid.uuid4()
    
    start_time = time.time()
    try:
        res = await rag_generator.generate(query, pipeline_id, run_id=run_uuid)
    except Exception as e:
        logger.error(f"Generation failed for pipeline {pipeline_id} during comparison: {e}")
        raise ValueError(f"Generation failed for pipeline {pipeline_id}: {str(e)}")
        
    total_latency_ms = (time.time() - start_time) * 1000

    # Retrieve info from database
    pool = await DatabasePool.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT retrieved_chunk_ids, metadata FROM pipeline_runs WHERE id = $1;",
            run_uuid
        )

    chunks_used = len(row["retrieved_chunk_ids"]) if row and row["retrieved_chunk_ids"] else 0
    
    meta = row["metadata"] or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
            
    retrieval_latency_ms = meta.get("retrieval_latency_ms") or (total_latency_ms * 0.15)
    
    # Run evaluation synchronously to get immediate evaluation score
    try:
        eval_res = await judge.evaluate_run(str(run_uuid))
        eval_score = eval_res["overall"]
    except Exception as e:
        logger.warning(f"Sync evaluation failed for run {run_uuid} in comparison: {e}")
        eval_score = 0.0

    return {
        "run_id": str(run_uuid),
        "generation": res.answer,
        "retrieval_latency_ms": round(retrieval_latency_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "chunks_used": chunks_used,
        "eval_score": round(eval_score, 4)
    }

@router.post("/pipelines/compare")
async def compare_pipelines(request: CompareRequest):
    try:
        pipeline_a = await manager.get_pipeline(request.pipeline_a_id)
        pipeline_b = await manager.get_pipeline(request.pipeline_b_id)
        if not pipeline_a:
            raise HTTPException(status_code=404, detail=f"Pipeline A with ID {request.pipeline_a_id} not found.")
        if not pipeline_b:
            raise HTTPException(status_code=404, detail=f"Pipeline B with ID {request.pipeline_b_id} not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    start_time = time.time()
    with tracer.start_as_current_span("pipeline.compare") as span:
        span.set_attribute("pipeline_id", request.pipeline_a_id)
        
        try:
            run_a_task = run_pipeline_comparison(request.pipeline_a_id, request.query)
            run_b_task = run_pipeline_comparison(request.pipeline_b_id, request.query)
            
            res_a, res_b = await asyncio.gather(run_a_task, run_b_task)
            
            latency = (time.time() - start_time) * 1000
            span.set_attribute("latency", latency)
            span.set_attribute("pipeline_a_id", request.pipeline_a_id)
            span.set_attribute("pipeline_b_id", request.pipeline_b_id)
            span.set_attribute("pipeline_a_score", res_a["eval_score"])
            span.set_attribute("pipeline_b_score", res_b["eval_score"])
            
            return {
                "query": request.query,
                "pipeline_a": res_a,
                "pipeline_b": res_b
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Pipelines comparison failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
