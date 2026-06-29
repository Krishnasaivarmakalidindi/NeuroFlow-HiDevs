import uuid
import json
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from db.pool import DatabasePool
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from db.pool import DatabasePool

logger = logging.getLogger(__name__)
router = APIRouter()

class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="User rating from 1 to 5")

@router.patch("/runs/{run_id}/rating")
async def update_rating(run_id: str, request: FeedbackRequest):
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run_id format. Must be a UUID.")

    pool = await DatabasePool.get_pool()
    async with pool.acquire() as conn:
        # Fetch the evaluation record
        row = await conn.fetchrow(
            "SELECT overall_score, metadata FROM evaluations WHERE run_id = $1;",
            run_uuid
        )
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation for run_id {run_id} not found."
            )

        automated = row["overall_score"] or 0.0
        metadata = row["metadata"] or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}

        # Calculate absolute difference
        user_score = request.rating / 5.0
        difference = abs(automated - user_score)

        if difference > 0.3:
            metadata["calibration_needed"] = True
        else:
            metadata["calibration_needed"] = False

        # Update user_rating and metadata in DB
        await conn.execute(
            """
            UPDATE evaluations
            SET user_rating = $1, metadata = $2::jsonb
            WHERE run_id = $3;
            """,
            request.rating,
            json.dumps(metadata),
            run_uuid
        )

    return {
        "status": "success",
        "run_id": str(run_uuid),
        "user_rating": request.rating,
        "difference": round(difference, 4),
        "calibration_needed": metadata["calibration_needed"]
    }
