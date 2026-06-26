# NeuroFlow Data Models

## Overview

NeuroFlow stores data across several domains:

* Document ingestion
* Vector embeddings
* Query execution
* Evaluation metrics
* Pipeline management
* Fine-tuning experiments

---

# 1. Document Model

Represents an uploaded file or URL.

```json
{
  "document_id": "uuid",
  "source_type": "pdf|docx|image|csv|url",
  "source_name": "string",
  "content_type": "string",
  "status": "processing|completed|failed",
  "metadata": {
    "project": "string",
    "tags": ["string"]
  },
  "created_at": "timestamp"
}
```

### Fields

| Field        | Type      |
| ------------ | --------- |
| document_id  | UUID      |
| source_type  | String    |
| source_name  | String    |
| content_type | String    |
| status       | Enum      |
| metadata     | JSON      |
| created_at   | Timestamp |

---

# 2. Chunk Model

Represents a chunk extracted from a document.

```json
{
  "chunk_id": "uuid",
  "document_id": "uuid",
  "chunk_index": 0,
  "text": "string",
  "token_count": 512,
  "embedding_model": "bge-large",
  "metadata": {}
}
```

### Fields

| Field           | Type    |
| --------------- | ------- |
| chunk_id        | UUID    |
| document_id     | UUID    |
| chunk_index     | Integer |
| text            | Text    |
| token_count     | Integer |
| embedding_model | String  |
| metadata        | JSON    |

---

# 3. Vector Embedding Model

Stores vector representations.

```json
{
  "embedding_id": "uuid",
  "chunk_id": "uuid",
  "vector_dimension": 1024,
  "embedding": "vector",
  "created_at": "timestamp"
}
```

### Fields

| Field            | Type      |
| ---------------- | --------- |
| embedding_id     | UUID      |
| chunk_id         | UUID      |
| vector_dimension | Integer   |
| embedding        | Vector    |
| created_at       | Timestamp |

---

# 4. Query Model

Stores user queries.

```json
{
  "query_id": "uuid",
  "user_id": "uuid",
  "query_text": "string",
  "pipeline_id": "uuid",
  "created_at": "timestamp"
}
```

### Fields

| Field       | Type      |
| ----------- | --------- |
| query_id    | UUID      |
| user_id     | UUID      |
| query_text  | Text      |
| pipeline_id | UUID      |
| created_at  | Timestamp |

---

# 5. Generation Model

Stores generated responses.

```json
{
  "generation_id": "uuid",
  "query_id": "uuid",
  "model_name": "gpt-4",
  "prompt": "string",
  "response": "string",
  "latency_ms": 1200,
  "created_at": "timestamp"
}
```

### Fields

| Field         | Type      |
| ------------- | --------- |
| generation_id | UUID      |
| query_id      | UUID      |
| model_name    | String    |
| prompt        | Text      |
| response      | Text      |
| latency_ms    | Integer   |
| created_at    | Timestamp |

---

# 6. Evaluation Model

Stores evaluation metrics.

```json
{
  "evaluation_id": "uuid",
  "query_id": "uuid",
  "faithfulness": 0.92,
  "answer_relevance": 0.89,
  "context_precision": 0.86,
  "context_recall": 0.88,
  "user_rating": 5
}
```

### Fields

| Field             | Type    |
| ----------------- | ------- |
| evaluation_id     | UUID    |
| query_id          | UUID    |
| faithfulness      | Float   |
| answer_relevance  | Float   |
| context_precision | Float   |
| context_recall    | Float   |
| user_rating       | Integer |

---

# 7. Pipeline Model

Stores retrieval pipeline configurations.

```json
{
  "pipeline_id": "uuid",
  "name": "research_pipeline",
  "embedding_model": "bge-large",
  "retrieval_strategy": "hybrid",
  "llm_model": "gpt-4",
  "created_at": "timestamp"
}
```

### Fields

| Field              | Type      |
| ------------------ | --------- |
| pipeline_id        | UUID      |
| name               | String    |
| embedding_model    | String    |
| retrieval_strategy | String    |
| llm_model          | String    |
| created_at         | Timestamp |

---

# 8. Fine-Tuning Job Model

Stores fine-tuning experiments.

```json
{
  "job_id": "uuid",
  "base_model": "llama-3",
  "dataset_id": "uuid",
  "status": "queued|running|completed|failed",
  "experiment_id": "string",
  "metrics": {}
}
```

### Fields

| Field         | Type   |
| ------------- | ------ |
| job_id        | UUID   |
| base_model    | String |
| dataset_id    | UUID   |
| status        | Enum   |
| experiment_id | String |
| metrics       | JSON   |

---

# Entity Relationship Overview

```text
Document
    │
    └──► Chunk
              │
              └──► Embedding

Query
    │
    └──► Generation
              │
              └──► Evaluation

Pipeline
    │
    └──► Query

Fine-Tuning Job
    │
    └──► Experiment
```

