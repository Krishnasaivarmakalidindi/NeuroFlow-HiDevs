# NeuroFlow Architecture

## Overview

NeuroFlow is a production-grade Retrieval-Augmented Generation (RAG) platform that supports document ingestion, hybrid retrieval, LLM generation, evaluation, and continuous model improvement through fine-tuning.

The system consists of five major subsystems:

1. Ingestion Subsystem
2. Retrieval Subsystem
3. Generation Subsystem
4. Evaluation Subsystem
5. Fine-Tuning Subsystem

---

# 1. Ingestion Subsystem

## Purpose

The ingestion subsystem accepts multiple data formats, extracts their contents, transforms them into embeddings, and stores them in the vector database for future retrieval.

## Supported Inputs

* PDF documents
* DOCX files
* Images
* CSV files
* Web URLs

## Workflow

```text
User Upload
     │
     ▼
File Detection
     │
     ▼
Content Extraction
(PDF/DOCX/OCR/Web Parser)
     │
     ▼
Text Cleaning
     │
     ▼
Chunking Engine
     │
     ▼
Embedding Model
     │
     ▼
Vector Store
```

## Components

* File Upload Service
* Document Parser
* OCR Engine
* Chunking Service
* Embedding Service
* Vector Database Writer

---

# 2. Retrieval Subsystem

## Purpose

The retrieval subsystem finds the most relevant information for a user query using multiple retrieval strategies.

## Workflow

```text
User Query
      │
      ▼
Query Embedding
      │
      ├─────────────► Vector Similarity Search
      │
      ├─────────────► Keyword Search
      │
      └─────────────► Metadata Filtering
                          │
                          ▼
                 Reciprocal Rank Fusion
                          │
                          ▼
                  Cross Encoder Reranker
                          │
                          ▼
                    Context Window
```

## Components

* Query Encoder
* Vector Retriever
* Keyword Retriever
* Metadata Filter
* RRF Fusion Engine
* Cross Encoder Reranker

---

# 3. Generation Subsystem

## Purpose

The generation subsystem creates responses using retrieved context and selects the most suitable LLM.

## Workflow

```text
Retrieved Context
        │
        ▼
Prompt Builder
        │
        ▼
Model Router
        │
        ▼
Selected LLM
        │
        ▼
Streaming Response
        │
        ▼
Evaluation Logger
```

## Model Routing Factors

* Cost
* Latency
* Capability
* Domain Expertise

## Components

* Prompt Builder
* Model Router
* LLM Gateway
* Streaming Service
* Logging Service

---

# 4. Evaluation Subsystem

## Purpose

The evaluation subsystem measures the quality of every generated answer automatically.

## Metrics

* Faithfulness
* Answer Relevance
* Context Precision
* Context Recall

## Workflow

```text
Question
Answer
Retrieved Context
        │
        ▼
Evaluation Engine
        │
        ├── Faithfulness
        ├── Relevance
        ├── Context Precision
        └── Context Recall
                │
                ▼
          PostgreSQL
                │
                ▼
       Rolling Analytics
```

## Components

* Evaluation Engine
* Metrics Calculator
* Analytics Engine
* Score Storage

---

# 5. Fine-Tuning Subsystem

## Purpose

The fine-tuning subsystem continuously improves models using high-quality historical interactions.

## Selection Criteria

* Faithfulness > 0.8
* User Rating >= 4

## Workflow

```text
Evaluation Logs
        │
        ▼
Quality Filtering
        │
        ▼
JSONL Dataset Builder
        │
        ▼
Fine-Tuning Pipeline
        │
        ▼
MLflow Tracking
        │
        ▼
Model Registry
        │
        ▼
Future Query Routing
```

## Components

* Data Extractor
* Dataset Generator
* Fine-Tuning Manager
* MLflow Tracker
* Model Registry

---

# System Architecture Overview

```text
               User
                 │
                 ▼
            Frontend
                 │
                 ▼
              Backend
                 │
        ┌────────┼────────┐
        │        │        │
        ▼        ▼        ▼
   Ingestion Retrieval Generation
        │        │        │
        └────────┼────────┘
                 │
                 ▼
            Evaluation
                 │
                 ▼
            Fine-Tuning
```
