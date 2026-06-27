import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import os
import json

from api.ingest import router
from pipelines.ingestion.models import ExtractedPage
from pipelines.ingestion.dedup import check_duplicate, compute_sha256, _get_db
from pipelines.ingestion.chunker import get_chunking_strategy, chunk_document
from pipelines.ingestion.worker import process_document
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_db():
    conn = _get_db()
    conn.execute('DELETE FROM documents')
    conn.commit()
    conn.close()
    yield

@pytest.mark.asyncio
async def test_duplicate_uploads():
    with patch("api.ingest.enqueue_document_job", new_callable=AsyncMock) as mock_enqueue:
        # First upload
        response1 = client.post("/ingest", data={"url": "https://example.com"})
        assert response1.status_code == 200
        assert response1.json()["duplicate"] == False
        
        # Second upload with same URL (hash will match)
        response2 = client.post("/ingest", data={"url": "https://example.com"})
        assert response2.status_code == 200
        assert response2.json()["duplicate"] == True
        assert response1.json()["document_id"] == response2.json()["document_id"]
        
        # Queue should only be called once
        mock_enqueue.assert_called_once()

@pytest.mark.asyncio
async def test_chunk_strategy_selection():
    p1 = ExtractedPage(page_number=1, content="Data", content_type="table", metadata={})
    assert get_chunking_strategy(p1) == "fixed"
    
    p2 = ExtractedPage(page_number=1, content="Doc", content_type="text", metadata={"level": "h1", "section": "Intro"})
    assert get_chunking_strategy(p2) == "hierarchical"
    
    p3 = ExtractedPage(page_number=1, content="Long PDF", content_type="text", metadata={})
    assert get_chunking_strategy(p3, total_pages=51) == "semantic"
    
    p4 = ExtractedPage(page_number=1, content="Short PDF", content_type="text", metadata={})
    assert get_chunking_strategy(p4, total_pages=5) == "fixed"

@pytest.mark.asyncio
async def test_worker_processing():
    with patch("pipelines.ingestion.worker.extract_content", new_callable=AsyncMock) as mock_extract:
        mock_extract.return_value = [
            ExtractedPage(page_number=1, content="Test content for document extraction.", content_type="text", metadata={})
        ]
        
        with patch("pipelines.ingestion.worker.NeuroFlowClient") as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.embed = AsyncMock(return_value=[[0.1] * 1536])
            
            ctx = {}
            doc_id = "test-doc-id"
            
            # Record it as queued
            conn = _get_db()
            conn.execute('INSERT INTO documents (id, content_hash, status) VALUES (?, ?, ?)', (doc_id, "hash", "queued"))
            conn.commit()
            
            result = await process_document(ctx, doc_id, "dummy/path.pdf", "pdf")
            
            assert result["status"] == "complete"
            assert result["chunks"] > 0
            
            cursor = conn.cursor()
            cursor.execute('SELECT status FROM documents WHERE id = ?', (doc_id,))
            status = cursor.fetchone()[0]
            assert status == "complete"
            
@pytest.mark.asyncio
async def test_document_status_transitions():
    doc_id = "status-test-id"
    conn = _get_db()
    conn.execute('INSERT INTO documents (id, content_hash, status) VALUES (?, ?, ?)', (doc_id, "hash2", "queued"))
    conn.commit()
    
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
