# ADR-002: Chunking Strategy

## Context

Document chunking significantly affects retrieval quality.

Three approaches were evaluated:

* Fixed-size chunking
* Sentence-boundary chunking
* Semantic chunking

Requirements:

* High retrieval accuracy
* Efficient processing
* Support for multiple document types
* Adaptability to future workloads

---

## Decision

NeuroFlow will use a hybrid chunking strategy.

### Default Strategy

* Sentence-boundary chunking
* Chunk size: 500 tokens
* Overlap: 100 tokens

### Special Cases

* Structured data → Fixed-size chunking
* Complex documents → Semantic chunking

---

## Consequences

### Positive

* Better context preservation
* Improved retrieval accuracy
* Flexible for different modalities

### Negative

* Increased implementation complexity
* Higher preprocessing cost
* Additional tuning requirements

