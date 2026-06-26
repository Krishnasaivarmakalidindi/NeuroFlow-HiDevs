# ADR-001: Vector Store Selection

## Context

NeuroFlow requires a vector database capable of storing embeddings, performing similarity search, supporting metadata filtering, and integrating well with the existing relational data infrastructure.

The considered alternatives were:

* pgvector
* Pinecone
* Weaviate
* Qdrant

The system also requires:

* Low operational complexity
* Cost efficiency
* SQL support
* Production scalability
* Integration with evaluation and analytics pipelines

---

## Decision

We will use **PostgreSQL with pgvector** as the primary vector database.

Reasons:

* Open source and self-hosted
* Supports vector similarity search
* Native SQL support
* Lower infrastructure cost
* Easy integration with metadata and analytics
* Mature ecosystem
* Simplifies deployment architecture

---

## Consequences

### Positive

* Reduced operational complexity
* Lower cost compared to managed vector databases
* Unified storage layer
* Easier backup and maintenance

### Negative

* Lower scalability compared to specialized vector databases
* Requires manual tuning for large-scale workloads
* May require migration to dedicated vector stores in the future
