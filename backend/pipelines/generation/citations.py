import re
from typing import List, Any, Union
from pipelines.generation.models import Citation

class CitationParser:
    @staticmethod
    def parse(text: str, retrieved_sources: List[Any]) -> List[Citation]:
        # Find all patterns like [Source N]
        matches = re.findall(r"\[Source (\d+)\]", text)
        citations = []
        seen = set()

        for num_str in matches:
            if num_str in seen:
                continue
            seen.add(num_str)
            
            n = int(num_str)
            ref = f"[Source {n}]"
            
            # Map index
            if 1 <= n <= len(retrieved_sources):
                src = retrieved_sources[n - 1]
                # src could be RetrievalResult or a dict
                if hasattr(src, "chunk_id"):
                    chunk_id = src.chunk_id
                    document_name = src.source
                    content = src.content
                    # Retrieve page number from metadata or chunk_index
                    page_number = None
                    if hasattr(src, "metadata") and src.metadata:
                        page_number = src.metadata.get("page_number")
                    if page_number is None and hasattr(src, "chunk_index"):
                        page_number = src.chunk_index
                else:
                    # Dict fallback
                    chunk_id = src.get("chunk_id", "")
                    document_name = src.get("source", "")
                    content = src.get("content", "")
                    metadata = src.get("metadata", {}) or {}
                    page_number = metadata.get("page_number")
                    if page_number is None:
                        page_number = src.get("chunk_index")

                preview = content[:100] + "..." if len(content) > 100 else content

                citations.append(
                    Citation(
                        reference=ref,
                        chunk_id=chunk_id,
                        document_name=document_name,
                        page_number=page_number,
                        content_preview=preview,
                        invalid_citation=False
                    )
                )
            else:
                # Hallucinated citation
                citations.append(
                    Citation(
                        reference=ref,
                        chunk_id="",
                        document_name="",
                        page_number=None,
                        content_preview="",
                        invalid_citation=True
                    )
                )

        return citations
