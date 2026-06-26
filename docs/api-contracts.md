# NeuroFlow API Contracts

## Authentication

* Authentication Method: JWT Bearer Token
* Rate Limit: 100 requests/minute per authenticated user
* Content-Type: application/json

---

# 1. POST /ingest

## Purpose

Ingest files or URLs into the knowledge base.

### Authentication

Required

### Rate Limit

20 requests/minute

### Request

```json
{
  "source_type": "pdf|docx|image|csv|url",
  "source": "string",
  "metadata": {
    "project": "string",
    "tags": ["string"]
  }
}
```

### Response

```json
{
  "ingestion_id": "uuid",
  "status": "processing",
  "chunks_created": 0,
  "message": "Ingestion started"
}
```

### Errors

| Code | Meaning        |
| ---- | -------------- |
| 400  | Invalid input  |
| 401  | Unauthorized   |
| 413  | File too large |
| 500  | Server error   |

---

# 2. POST /query

## Purpose

Execute RAG query.

### Authentication

Required

### Rate Limit

60 requests/minute

### Request

```json
{
  "query": "What is Retrieval Augmented Generation?",
  "pipeline_id": "uuid",
  "top_k": 10,
  "filters": {
    "project": "research"
  }
}
```

### Response

```json
{
  "query_id": "uuid",
  "answer": "string",
  "sources": [
    {
      "chunk_id": "uuid",
      "score": 0.95
    }
  ],
  "model_used": "gpt-4",
  "latency_ms": 1200
}
```

### Errors

| Code | Meaning             |
| ---- | ------------------- |
| 400  | Bad request         |
| 401  | Unauthorized        |
| 429  | Rate limit exceeded |
| 500  | Server error        |

---

# 3. GET /query/{query_id}/stream

## Purpose

Stream generated response.

### Authentication

Required

### Rate Limit

100 requests/minute

### Response (SSE)

```text
event: token
data: Hello

event: token
data: World

event: done
data: complete
```

### Errors

| Code | Meaning         |
| ---- | --------------- |
| 404  | Query not found |
| 500  | Server error    |

---

# 4. GET /evaluations

## Purpose

Retrieve evaluation results.

### Authentication

Required

### Rate Limit

100 requests/minute

### Response

```json
{
  "page": 1,
  "size": 20,
  "results": [
    {
      "query_id": "uuid",
      "faithfulness": 0.91,
      "answer_relevance": 0.89,
      "context_precision": 0.85,
      "context_recall": 0.87
    }
  ]
}
```

### Errors

| Code | Meaning      |
| ---- | ------------ |
| 401  | Unauthorized |
| 500  | Server error |

---

# 5. GET /evaluations/aggregate

## Purpose

Retrieve rolling quality metrics.

### Authentication

Required

### Response

```json
{
  "avg_faithfulness": 0.89,
  "avg_relevance": 0.91,
  "avg_precision": 0.87,
  "avg_recall": 0.88,
  "evaluated_samples": 12000
}
```

---

# 6. POST /pipelines

## Purpose

Create pipeline configuration.

### Authentication

Required

### Request

```json
{
  "name": "research_pipeline",
  "embedding_model": "bge-large",
  "retrieval_strategy": "hybrid",
  "llm": "gpt-4"
}
```

### Response

```json
{
  "pipeline_id": "uuid",
  "status": "created"
}
```

---

# 7. GET /pipelines/{id}/runs

## Purpose

Retrieve pipeline execution history.

### Authentication

Required

### Response

```json
{
  "pipeline_id": "uuid",
  "runs": [
    {
      "run_id": "uuid",
      "status": "completed",
      "duration": 35
    }
  ]
}
```

---

# 8. POST /finetune/jobs

## Purpose

Submit fine-tuning job.

### Authentication

Required

### Request

```json
{
  "base_model": "llama-3",
  "dataset_id": "uuid",
  "epochs": 3
}
```

### Response

```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

---

# 9. GET /finetune/jobs/{id}

## Purpose

Get fine-tuning job status.

### Authentication

Required

### Response

```json
{
  "job_id": "uuid",
  "status": "running",
  "progress": 75,
  "metrics": {
    "loss": 0.12
  }
}
```

---

# 10. GET /health

## Purpose

Health check endpoint.

### Authentication

Not Required

### Response

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

# 11. GET /metrics

## Purpose

System monitoring metrics.

### Authentication

Required

### Response

```json
{
  "cpu_usage": 35,
  "memory_usage": 62,
  "requests_per_minute": 245,
  "average_latency": 850
}
```

