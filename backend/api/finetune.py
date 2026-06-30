import logging
import json
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

try:
    from db.pool import DatabasePool
    from pipelines.finetuning.job_manager import FineTuneJobManager
    from pipelines.finetuning.extractor import TrainingDataExtractor
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from db.pool import DatabasePool
    from pipelines.finetuning.job_manager import FineTuneJobManager
    from pipelines.finetuning.extractor import TrainingDataExtractor

logger = logging.getLogger(__name__)

router = APIRouter()
manager = FineTuneJobManager()
extractor = TrainingDataExtractor()

class FineTuneJobRequest(BaseModel):
    base_model: str = "llama-3.3-70b-versatile"
    task_type: str = "legal"

@router.post("/finetune/jobs", status_code=201)
async def create_finetune_job(request: FineTuneJobRequest):
    try:
        provider_job_id = await manager.submit_mock_finetune_job(request.base_model, request.task_type)
        return {
            "provider_job_id": provider_job_id,
            "status": "queued"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to submit fine-tuning job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/finetune/jobs")
async def list_finetune_jobs():
    try:
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, provider_job_id, base_model, status, mlflow_run_id, training_pair_count, metrics, created_at, completed_at 
                FROM finetune_jobs 
                ORDER BY created_at DESC;
                """
            )
            
            results = []
            for r in rows:
                metrics_data = r["metrics"]
                if isinstance(metrics_data, str):
                    try:
                        metrics_data = json.loads(metrics_data)
                    except Exception:
                        pass
                
                results.append({
                    "id": str(r["id"]),
                    "provider_job_id": r["provider_job_id"],
                    "base_model": r["base_model"],
                    "status": r["status"],
                    "mlflow_run_id": r["mlflow_run_id"],
                    "training_pair_count": r["training_pair_count"],
                    "metrics": metrics_data,
                    "created_at": r["created_at"],
                    "completed_at": r["completed_at"]
                })
            return results
    except Exception as e:
        logger.error(f"Failed to list fine-tuning jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/finetune/jobs/{job_id}")
async def get_finetune_job(job_id: str):
    try:
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, provider_job_id, base_model, status, mlflow_run_id, training_pair_count, metrics, created_at, completed_at 
                FROM finetune_jobs 
                WHERE provider_job_id = $1 or id::text = $1;
                """,
                job_id
            )
            
            if not row:
                raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found.")
                
            metrics_data = row["metrics"]
            if isinstance(metrics_data, str):
                try:
                    metrics_data = json.loads(metrics_data)
                except Exception:
                    pass
                    
            return {
                "id": str(row["id"]),
                "provider_job_id": row["provider_job_id"],
                "base_model": row["base_model"],
                "status": row["status"],
                "mlflow_run_id": row["mlflow_run_id"],
                "training_pair_count": row["training_pair_count"],
                "metrics": metrics_data,
                "created_at": row["created_at"],
                "completed_at": row["completed_at"]
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve fine-tuning job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/finetune/training-data/preview")
async def preview_training_data():
    try:
        pairs = await extractor.extract_pairs()
        preview = []
        for p in pairs[:5]: # Return top 5 preview
            preview.append({
                "query": p["user_message"],
                "quality_score": p["quality_score"]
            })
        return preview
    except Exception as e:
        logger.error(f"Failed to preview training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
