import logging
import time
from opentelemetry import trace
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from .extractors import extract_content
from .chunker import chunk_document
from .dedup import compute_sha256, check_duplicate, record_document, update_document_status
from providers.client import NeuroFlowClient
from providers.router import RoutingCriteria

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

async def process_document(ctx, document_id: str, file_path: str, source_type: str):
    start_time = time.time()
    update_document_status(document_id, "processing")
    
    with tracer.start_as_current_span("ingestion.process") as span:
        span.set_attribute("document_id", document_id)
        span.set_attribute("source_type", source_type)
        
        try:
            # Extraction
            pages = await extract_content(file_path, source_type)
            page_count = len(pages)
            span.set_attribute("page_count", page_count)
            
            # Chunking
            chunks = chunk_document(pages)
            chunk_count = len(chunks)
            span.set_attribute("chunk_count", chunk_count)
            
            # Embedding
            client = NeuroFlowClient()
            criteria = RoutingCriteria() # Default embedding criteria
            
            # Assuming we batch embeddings
            texts = [c["content"] for c in chunks]
            if texts:
                try:
                    embeddings = await client.embed(texts, criteria)
                    span.set_attribute("embedding_calls", 1)
                except NotImplementedError:
                    # e.g., Anthropic doesn't support embeddings in our mock
                    span.set_attribute("embedding_calls", 0)
            else:
                span.set_attribute("embedding_calls", 0)
                
            # Log completion
            duration_ms = int((time.time() - start_time) * 1000)
            total_tokens = sum(len(c["content"]) // 4 for c in chunks) # rough estimate for logging
            
            logger.info(
                f"event=ingestion_complete document_id={document_id} "
                f"duration_ms={duration_ms} chunks={chunk_count} tokens={total_tokens}"
            )
            
            update_document_status(document_id, "complete")
            return {"status": "complete", "chunks": chunk_count}
            
        except Exception as e:
            span.record_exception(e)
            logger.error(f"Error processing document {document_id}: {str(e)}")
            update_document_status(document_id, "failed")
            raise

class WorkerSettings:
    functions = [process_document]
    # Optionally configure redis settings
