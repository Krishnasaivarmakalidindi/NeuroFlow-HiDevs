import asyncio
import logging
import time
import uuid
import json
from typing import Dict, Any, List, Optional
import tiktoken
from opentelemetry import trace

try:
    from db.pool import DatabasePool
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria
    from providers.base import ChatMessage
    from pipelines.retrieval.pipeline import RetrievalPipeline
    from pipelines.generation.models import GenerationResult, Citation
    from pipelines.generation.prompt_builder import PromptBuilder
    from pipelines.generation.citations import CitationParser
    from pipelines.ingestion.queue import get_redis_pool
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from db.pool import DatabasePool
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria
    from providers.base import ChatMessage
    from pipelines.retrieval.pipeline import RetrievalPipeline
    from pipelines.generation.models import GenerationResult, Citation
    from pipelines.generation.prompt_builder import PromptBuilder
    from pipelines.generation.citations import CitationParser
    from pipelines.ingestion.queue import get_redis_pool

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Streaming stripping helper for CoT
class StreamingCoTStripper:
    def __init__(self):
        self.buffer = ""
        self.in_think = False
        self.think_content = ""
        self.started = False

    def process_chunk(self, chunk: str) -> str:
        if not self.started:
            self.buffer += chunk
            prefix = "<think>"
            if prefix in self.buffer:
                self.in_think = True
                self.started = True
                parts = self.buffer.split(prefix, 1)
                before = parts[0]
                self.buffer = parts[1]
                # Process the rest of the buffer recursively
                return before + self.process_chunk("")
            elif len(self.buffer) < len(prefix) and prefix.startswith(self.buffer):
                return ""
            else:
                self.started = True
                res = self.buffer
                self.buffer = ""
                return res

        if self.in_think:
            self.buffer += chunk
            if "</think>" in self.buffer:
                parts = self.buffer.split("</think>", 1)
                self.think_content += parts[0]
                self.in_think = False
                self.buffer = ""
                return parts[1]
            else:
                suffix = "</think>"
                matched_len = 0
                for i in range(1, len(suffix) + 1):
                    sub = suffix[:i]
                    if self.buffer.endswith(sub):
                        matched_len = i
                
                if matched_len > 0:
                    non_match = self.buffer[:-matched_len]
                    self.think_content += non_match
                    self.buffer = self.buffer[-matched_len:]
                else:
                    self.think_content += self.buffer
                    self.buffer = ""
                return ""
        else:
            return chunk

    def flush(self) -> str:
        if not self.started:
            return self.buffer
        if self.in_think:
            self.think_content += self.buffer
            return ""
        return self.buffer


class RAGGenerator:
    def __init__(self):
        self.client = NeuroFlowClient()
        self.retrieval_pipeline = RetrievalPipeline()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def _ensure_db_schema(self, conn):
        # Alter table to add metadata JSONB column if it does not exist
        try:
            await conn.execute(
                "ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';"
            )
        except Exception as e:
            logger.warning(f"Failed to run schema migration for metadata column: {e}")

    async def _ensure_pipeline_exists(self, conn, pipeline_id: uuid.UUID):
        # Ensure that the pipeline_id exists in the pipelines table to prevent foreign key violation
        try:
            row = await conn.fetchrow("SELECT id FROM pipelines WHERE id = $1;", pipeline_id)
            if not row:
                await conn.execute(
                    "INSERT INTO pipelines (id, name, config) VALUES ($1, $2, $3::jsonb) ON CONFLICT (id) DO NOTHING;",
                    pipeline_id,
                    f"Default Pipeline {str(pipeline_id)[:8]}",
                    "{}"
                )
        except Exception as e:
            logger.warning(f"Failed to ensure pipeline {pipeline_id} exists: {e}")

    async def generate(
        self,
        query: str,
        pipeline_id: str,
        stream_queue: Optional[asyncio.Queue] = None,
        run_id: Optional[uuid.UUID] = None,
        **kwargs
    ) -> GenerationResult:
        if run_id is None:
            run_id = uuid.uuid4()
        else:
            run_id = uuid.UUID(str(run_id)) if not isinstance(run_id, uuid.UUID) else run_id
        pipeline_uuid = uuid.UUID(pipeline_id) if isinstance(pipeline_id, str) else pipeline_id

        # 1. Run Retrieval Pipeline (OTel Span)
        with tracer.start_as_current_span("generation.pipeline") as pipeline_span:
            pipeline_span.set_attribute("run_id", str(run_id))
            pipeline_span.set_attribute("query", query)

            if stream_queue:
                await stream_queue.put({"type": "retrieval_start"})

            retrieval_result = await self.retrieval_pipeline.run(query, **kwargs)
            context = retrieval_result["context"]
            chunks_used = retrieval_result["chunks_used"]
            sources = retrieval_result["sources"]
            query_type = retrieval_result["pipeline_meta"]["query_type"]

            if stream_queue:
                await stream_queue.put({
                    "type": "retrieval_complete",
                    "chunk_count": len(chunks_used),
                    "sources": sources
                })

            # 2. Build Prompt (OTel Span)
            with tracer.start_as_current_span("generation.prompt") as prompt_span:
                prompt = PromptBuilder.build(query, context, query_type)
                prompt_span.set_attribute("prompt_length", len(prompt))

            # 3. DB Insertion: status=running
            db_pool = None
            try:
                db_pool = await DatabasePool.get_pool()
                async with db_pool.acquire() as conn:
                    await self._ensure_db_schema(conn)
                    await self._ensure_pipeline_exists(conn, pipeline_uuid)
                    
                    # Convert chunk ids to UUIDs
                    chunk_uuids = [uuid.UUID(cid) for cid in chunks_used if cid]
                    await conn.execute(
                        """
                        INSERT INTO pipeline_runs (id, pipeline_id, query, retrieved_chunk_ids, status, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6::jsonb);
                        """,
                        run_id,
                        pipeline_uuid,
                        query,
                        chunk_uuids,
                        "running",
                        json.dumps({"prompt": prompt})
                    )
            except Exception as e:
                logger.error(f"Database error during pipeline_runs initial insert: {e}")

            # 4. Stream and Generate Response (OTel Span)
            criteria = RoutingCriteria()
            messages = [ChatMessage(role="user", content=prompt)]
            cot_stripper = StreamingCoTStripper()
            accumulated_answer = ""
            start_time = time.time()

            input_tokens = len(self.tokenizer.encode(prompt))

            with tracer.start_as_current_span("generation.stream") as stream_span:
                stream_span.set_attribute("run_id", str(run_id))
                try:
                    # Request stream from client
                    stream_generator = await self.client.chat(messages, criteria, stream=True)
                    async for chunk in stream_generator:
                        accumulated_answer += chunk
                        
                        # Process CoT stripping
                        clean_chunk = cot_stripper.process_chunk(chunk)
                        if clean_chunk and stream_queue:
                            await stream_queue.put({"type": "token", "delta": clean_chunk})

                    # Flush stripper
                    clean_chunk = cot_stripper.flush()
                    if clean_chunk and stream_queue:
                        await stream_queue.put({"type": "token", "delta": clean_chunk})
                except Exception as e:
                    stream_span.record_exception(e)
                    logger.error(f"Streaming error: {e}")
                    raise e

            latency_ms = (time.time() - start_time) * 1000
            think_content = cot_stripper.think_content.strip()

            # Clean final answer (we strip reasoning out of the returned answer)
            # If the model didn't use <think> tags, it's just the full answer.
            # If it did, cot_stripper extracted think_content, so the final clean answer
            # is the full answer with <think>...</think> block removed.
            final_answer = accumulated_answer
            if think_content:
                # Remove <think>...</think> block from the final answer
                # Handle cases with or without closing tag gracefully
                if "<think>" in final_answer:
                    parts = final_answer.split("<think>", 1)
                    before = parts[0]
                    after = ""
                    if "</think>" in parts[1]:
                        after = parts[1].split("</think>", 1)[1]
                    final_answer = before + after
            
            final_answer = final_answer.strip()

            output_tokens = len(self.tokenizer.encode(final_answer))
            model_used = "llama-3.3-70b-versatile" # default / router model

            # 5. Citations Resolution (OTel Span)
            with tracer.start_as_current_span("generation.citations") as citations_span:
                citations = CitationParser.parse(final_answer, retrieval_result["reranked"])
                citations_span.set_attribute("run_id", str(run_id))
                citations_span.set_attribute("citation_count", len(citations))

            # 6. DB Update: status=complete
            try:
                if db_pool:
                    async with db_pool.acquire() as conn:
                        await conn.execute(
                            """
                            UPDATE pipeline_runs
                            SET generation = $1, latency_ms = $2, input_tokens = $3,
                                output_tokens = $4, model_used = $5, status = $6, metadata = $7::jsonb
                            WHERE id = $8;
                            """,
                            final_answer,
                            int(latency_ms),
                            input_tokens,
                            output_tokens,
                            model_used,
                            "complete",
                            json.dumps({
                                "prompt": prompt,
                                "think_content": think_content,
                                "citations": [c.__dict__ for c in citations]
                            }),
                            run_id
                        )
            except Exception as e:
                logger.error(f"Database error during pipeline_runs final update: {e}")

            # 7. Async Evaluation Enqueue
            try:
                redis_pool = await get_redis_pool()
                await redis_pool.enqueue_job("evaluate_run", str(run_id))
            except Exception as e:
                logger.warning(f"Failed to enqueue evaluation job: {e}")

            if stream_queue:
                await stream_queue.put({
                    "type": "done",
                    "run_id": str(run_id),
                    "citations": [c.__dict__ for c in citations]
                })
                # Indicate end of queue stream
                await stream_queue.put(None)

            # Set top-level span attributes
            pipeline_span.set_attribute("model", model_used)
            pipeline_span.set_attribute("tokens", input_tokens + output_tokens)
            pipeline_span.set_attribute("latency", latency_ms)
            pipeline_span.set_attribute("citation_count", len(citations))

            return GenerationResult(
                answer=final_answer,
                citations=citations,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_used=model_used,
                latency_ms=latency_ms
            )
