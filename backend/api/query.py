import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from opentelemetry import trace
from pydantic import BaseModel

from resilience import SlidingWindowRateLimiter

try:
    from pipelines.generation.generator import RAGGenerator
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from pipelines.generation.generator import RAGGenerator

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

router = APIRouter()
rag_generator = RAGGenerator()

# In-memory dictionary to hold active stream queues
stream_queues: Dict[str, asyncio.Queue] = {}

class QueryRequest(BaseModel):
    query: str
    pipeline_id: str
    stream: bool = True

@router.post("/query")
async def post_query(request: QueryRequest, req: Request, background_tasks: BackgroundTasks):
    # API Rate Limiting Check (60/minute/IP)
    ip = req.client.host if req.client else "127.0.0.1"
    limiter = SlidingWindowRateLimiter()
    is_allowed, retry_after = await limiter.is_allowed(ip, "/query", limit=60, window_seconds=60)
    await limiter.close()
    
    if not is_allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limit_exceeded", "retry_after": retry_after},
            headers={"Retry-After": str(retry_after)}
        )
        
    if not request.query or not request.pipeline_id:
        raise HTTPException(status_code=400, detail="query and pipeline_id must be provided")

    if not request.stream:
        # Synchronous execution
        with tracer.start_as_current_span("generation.pipeline.sync") as span:
            try:
                res = await rag_generator.generate(request.query, request.pipeline_id)
                return {
                    "answer": res.answer,
                    "citations": [c.__dict__ for c in res.citations]
                }
            except Exception as e:
                logger.error(f"Sync query execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    else:
        # Streaming execution
        run_id = str(uuid.uuid4())
        queue = asyncio.Queue()
        stream_queues[run_id] = queue

        # Start the generator task in the background
        async def run_pipeline_task():
            try:
                await rag_generator.generate(request.query, request.pipeline_id, stream_queue=queue, run_id=uuid.UUID(run_id))
            except Exception as e:
                logger.error(f"Background generation failed: {e}")
                await queue.put({"type": "error", "message": str(e)})
                await queue.put(None)

        # Use background_tasks or create_task to launch immediately
        asyncio.create_task(run_pipeline_task())

        return {"run_id": run_id}

@router.get("/query/{run_id}/stream")
async def get_query_stream(run_id: str):
    if run_id not in stream_queues:
        raise HTTPException(status_code=404, detail="Stream queue not found or expired")

    async def event_generator():
        with tracer.start_as_current_span("generation.sse") as span:
            span.set_attribute("run_id", run_id)
            queue = stream_queues.get(run_id)
            if not queue:
                return

            try:
                while True:
                    try:
                        # Wait up to 15 seconds for next chunk
                        item = await asyncio.wait_for(queue.get(), timeout=15.0)
                        if item is None:
                            break
                        
                        yield {
                            "event": "message",
                            "data": json.dumps(item)
                        }
                    except asyncio.TimeoutError:
                        # Yield keepalive
                        yield {
                            "event": "message",
                            "data": json.dumps({"type": "keepalive"})
                        }
            finally:
                # Clean up queue
                if run_id in stream_queues:
                    del stream_queues[run_id]

    return EventSourceResponse(event_generator())
