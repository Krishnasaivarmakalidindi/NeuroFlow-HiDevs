import logging
import uuid
import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

try:
    from models.pipeline import PipelineConfig
    from services.pipeline_manager import PipelineManager
    from services.pipeline_analytics import PipelineAnalyticsService
    from services.pipeline_optimizer import PipelineOptimizer
    from db.pool import DatabasePool
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.pipeline import PipelineConfig
    from services.pipeline_manager import PipelineManager
    from services.pipeline_analytics import PipelineAnalyticsService
    from services.pipeline_optimizer import PipelineOptimizer
    from db.pool import DatabasePool

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

router = APIRouter()
manager = PipelineManager()
analytics_service = PipelineAnalyticsService()
optimizer = PipelineOptimizer()

@router.post("/pipelines", status_code=201)
async def create_pipeline(config: PipelineConfig):
    try:
        return await manager.create_pipeline(config)
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipelines")
async def list_pipelines():
    try:
        return await manager.list_pipelines()
    except Exception as e:
        logger.error(f"Failed to list pipelines: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    try:
        pipeline = await manager.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found.")
        return pipeline
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/pipelines/{pipeline_id}")
async def update_pipeline(pipeline_id: str, config: PipelineConfig):
    try:
        return await manager.update_pipeline(pipeline_id, config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    try:
        success = await manager.delete_pipeline(pipeline_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found or could not be archived.")
        return {"status": "success", "message": "Pipeline archived."}
    except Exception as e:
        logger.error(f"Failed to delete pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipelines/{pipeline_id}/runs")
async def get_pipeline_runs(
    pipeline_id: str,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    sort_by: str = Query("created_at"),
    order: str = Query("desc"),
    status: Optional[str] = Query(None),
    model_used: Optional[str] = Query(None)
):
    try:
        pipeline_uuid = uuid.UUID(pipeline_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid pipeline ID format: {pipeline_id}")

    # Validate sorting and ordering
    allowed_sort = {"created_at", "latency_ms", "input_tokens", "output_tokens"}
    allowed_order = {"asc", "desc"}
    if sort_by not in allowed_sort:
        sort_by = "created_at"
    if order not in allowed_order:
        order = "desc"

    # Map sort column name to table attributes
    sort_column = "r.created_at"
    if sort_by == "latency_ms":
        sort_column = "r.latency_ms"
    elif sort_by == "input_tokens":
        sort_column = "r.input_tokens"
    elif sort_by == "output_tokens":
        sort_column = "r.output_tokens"

    pool = await DatabasePool.get_pool()
    async with pool.acquire() as conn:
        # Build SQL dynamically with bind parameters to prevent SQL injection
        query = """
            SELECT r.id as run_id, r.latency_ms as latency, (COALESCE(r.input_tokens, 0) + COALESCE(r.output_tokens, 0)) as tokens,
                   e.faithfulness, e.answer_relevance, e.context_precision, e.context_recall, e.overall_score
            FROM pipeline_runs r
            LEFT JOIN evaluations e ON r.id = e.run_id
            WHERE r.pipeline_id = $1
        """
        
        params = [pipeline_uuid]
        param_idx = 2

        if status:
            query += f" AND r.status = ${param_idx}"
            params.append(status)
            param_idx += 1

        if model_used:
            query += f" AND r.model_used = ${param_idx}"
            params.append(model_used)
            param_idx += 1

        # Safe dynamic formatting since sort_column and order are whitelisted
        query += f" ORDER BY {sort_column} {order.upper()}"
        
        query += f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        offset = (page - 1) * limit
        params.extend([limit, offset])

        rows = await conn.fetch(query, *params)

        results = []
        for r in rows:
            eval_data = {}
            if r["overall_score"] is not None:
                eval_data = {
                    "faithfulness": r["faithfulness"],
                    "answer_relevance": r["answer_relevance"],
                    "context_precision": r["context_precision"],
                    "context_recall": r["context_recall"],
                    "overall": r["overall_score"]
                }
            results.append({
                "run_id": str(r["run_id"]),
                "latency": r["latency"] or 0,
                "tokens": r["tokens"] or 0,
                "evaluation": eval_data
            })
        return results

@router.get("/pipelines/{pipeline_id}/analytics")
async def get_pipeline_analytics(pipeline_id: str):
    start_time = time.time()
    with tracer.start_as_current_span("pipeline.analytics") as span:
        try:
            span.set_attribute("pipeline_id", pipeline_id)
            analytics = await analytics_service.get_analytics(pipeline_id)
            
            latency = (time.time() - start_time) * 1000
            span.set_attribute("latency", latency)
            span.set_attribute("evaluation_score", analytics["evaluation"]["overall"])
            
            return analytics
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to get pipeline analytics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipelines/{pipeline_id}/suggestions")
async def get_pipeline_suggestions(pipeline_id: str):
    start_time = time.time()
    with tracer.start_as_current_span("pipeline.optimize") as span:
        try:
            span.set_attribute("pipeline_id", pipeline_id)
            suggestions = await optimizer.get_suggestions(pipeline_id)
            
            latency = (time.time() - start_time) * 1000
            span.set_attribute("latency", latency)
            
            return suggestions
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to get pipeline suggestions: {e}")
            raise HTTPException(status_code=500, detail=str(e))
