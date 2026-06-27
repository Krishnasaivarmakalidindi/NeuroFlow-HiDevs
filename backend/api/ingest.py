import os
import uuid
import sqlite3
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Form
from pydantic import BaseModel
from typing import Optional

from pipelines.ingestion.dedup import compute_sha256, check_duplicate, record_document, _get_db
from pipelines.ingestion.queue import enqueue_document_job
from opentelemetry import trace

router = APIRouter()
tracer = trace.get_tracer(__name__)

class UrlIngestRequest(BaseModel):
    url: str

@router.post("/ingest")
async def ingest_document(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
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
