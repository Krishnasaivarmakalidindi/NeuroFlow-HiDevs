import os
import uuid
import sqlite3
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from pipelines.ingestion.dedup import compute_sha256, check_duplicate, record_document, _get_db
from pipelines.ingestion.queue import enqueue_document_job
from opentelemetry import trace

from resilience import SlidingWindowRateLimiter, BackpressureManager

router = APIRouter()
tracer = trace.get_tracer(__name__)

class UrlIngestRequest(BaseModel):
    url: str

@router.post("/ingest")
async def ingest_document(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    # 1. API Rate Limiting Check (10/hour/IP)
    ip = request.client.host if request.client else "127.0.0.1"
    limiter = SlidingWindowRateLimiter()
    is_allowed, retry_after = await limiter.is_allowed(ip, "/ingest", limit=10, window_seconds=3600)
    await limiter.close()
    
    if not is_allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limit_exceeded", "retry_after": retry_after},
            headers={"Retry-After": str(retry_after)}
        )
        
    # 2. Backpressure Queue Depth Check
    bp_manager = BackpressureManager()
    bp_res = await bp_manager.check_backpressure()
    await bp_manager.close()
    
    if bp_res:
        status_code, body = bp_res
        if status_code == 503:
            return JSONResponse(status_code=503, content=body)
        elif status_code == 202:
            # High queue depth: process document, but return 202 with warning details
            if not file and not url:
                raise HTTPException(status_code=400, detail="Must provide either file or url")
                
            doc_id = str(uuid.uuid4())
            if file:
                file_bytes = await file.read()
                if len(file_bytes) > 100 * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="File size exceeds 100MB limit")
                content_hash = compute_sha256(file_bytes)
                source_type = file.filename.split('.')[-1].lower() if file.filename else 'unknown'
                
                existing_id = check_duplicate(content_hash)
                if existing_id:
                    res_body = body.copy()
                    res_body.update({
                        "document_id": existing_id,
                        "status": "complete",
                        "duplicate": True
                    })
                    return JSONResponse(status_code=202, content=res_body)
                    
                temp_dir = os.path.join(os.path.dirname(__file__), "..", "temp")
                os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, f"{doc_id}.{source_type}")
                with open(file_path, "wb") as f:
                    f.write(file_bytes)
            else:
                content_hash = compute_sha256(url.encode('utf-8'))
                source_type = "url"
                file_path = url
                
                existing_id = check_duplicate(content_hash)
                if existing_id:
                    res_body = body.copy()
                    res_body.update({
                        "document_id": existing_id,
                        "status": "complete",
                        "duplicate": True
                    })
                    return JSONResponse(status_code=202, content=res_body)
                    
            record_document(doc_id, content_hash, "queued")
            await enqueue_document_job(doc_id, file_path, source_type)
            
            res_body = body.copy()
            res_body.update({
                "document_id": doc_id,
                "status": "queued",
                "duplicate": False
            })
            return JSONResponse(status_code=202, content=res_body)
            
    if not file and not url:
        raise HTTPException(status_code=400, detail="Must provide either file or url")
        
    doc_id = str(uuid.uuid4())
    
    if file:
        file_bytes = await file.read()
        if len(file_bytes) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 100MB limit")
            
        content_hash = compute_sha256(file_bytes)
        source_type = file.filename.split('.')[-1].lower() if file.filename else 'unknown'
        
        # Deduplication check
        existing_id = check_duplicate(content_hash)
        if existing_id:
            return {
                "document_id": existing_id,
                "status": "complete", # or whatever status it is in DB
                "duplicate": True
            }
            
        # Save file temporarily
        temp_dir = os.path.join(os.path.dirname(__file__), "..", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, f"{doc_id}.{source_type}")
        with open(file_path, "wb") as f:
            f.write(file_bytes)
            
    else:
        # Handle URL
        content_hash = compute_sha256(url.encode('utf-8'))
        source_type = "url"
        file_path = url # Using file_path as the URL identifier
        
        existing_id = check_duplicate(content_hash)
        if existing_id:
            return {
                "document_id": existing_id,
                "status": "complete",
                "duplicate": True
            }
            
    record_document(doc_id, content_hash, "queued")
    
    # Enqueue job
    await enqueue_document_job(doc_id, file_path, source_type)
    
    return {
        "document_id": doc_id,
        "status": "queued",
        "duplicate": False
    }

@router.get("/documents/{id}")
async def get_document(id: str):
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT status FROM documents WHERE id = ?', (id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # In a full implementation, chunk_count and metadata would be queried from 
    # a chunks or documents metadata table. We simulate the response here.
    return {
        "id": id,
        "status": row[0],
        "chunk_count": 0, # simulated
        "metadata": {}
    }
