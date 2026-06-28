from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Citation:
    reference: str
    chunk_id: str
    document_name: str
    page_number: Optional[int]
    content_preview: str
    invalid_citation: bool = False

@dataclass
class GenerationResult:
    answer: str
    citations: List[Citation]
    input_tokens: int
    output_tokens: int
    model_used: str
    latency_ms: float
