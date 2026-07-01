import tiktoken
from typing import List, Dict, Any
from .models import ExtractedPage

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None
    cosine_similarity = None

def get_chunking_strategy(page: ExtractedPage, total_pages: int = 1) -> str:
    if page.content_type == "table":
        return "fixed"
    if page.metadata.get("level") and page.metadata.get("section"):
        return "hierarchical"
    if total_pages > 50:
        return "semantic"
    return "fixed"

def chunk_fixed(text: str, max_tokens: int = 512, overlap: int = 64) -> List[str]:
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        # To avoid splitting mid-sentence we could decode and find the nearest period, 
        # but for fixed size tiktoken chunking we typically decode directly.
        # Simple heuristic to find last period in the chunk to avoid mid-sentence splits
        chunk_text = encoder.decode(chunk_tokens)
        
        # If it's not the last chunk and we can find a period in the last 25% of the text
        if i + max_tokens < len(tokens):
            last_period = chunk_text.rfind('.')
            if last_period > len(chunk_text) * 0.75:
                # Recalculate token index
                chunk_text = chunk_text[:last_period + 1]
                chunk_tokens = encoder.encode(chunk_text)
                
        chunks.append(chunk_text)
        i += max(1, len(chunk_tokens) - overlap)
        
    return chunks

def chunk_semantic(text: str, threshold: float = 0.7) -> List[str]:
    if not SentenceTransformer or not cosine_similarity:
        # Fallback to fixed if ML libs are not installed
        return chunk_fixed(text)
        
    # Split into sentences
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    if not sentences:
        return []
        
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def chunk_hierarchical(page: ExtractedPage) -> List[Dict[str, Any]]:
    # Simple simulation of hierarchical chunking. 
    # Returns chunk dicts with parent_id metadata
    parent_text = f"Header: {page.metadata.get('section', 'Unknown')}"
    parent_id = f"parent_{page.page_number}"
    
    # The parent is its own chunk
    chunks = [{"content": parent_text, "metadata": {"chunk_type": "parent", "parent_id": None, "id": parent_id}}]
    
    # Children are fixed size chunks of the content
    child_texts = chunk_fixed(page.content, max_tokens=256, overlap=32)
    for i, child in enumerate(child_texts):
        chunks.append({
            "content": child,
            "metadata": {"chunk_type": "child", "parent_id": parent_id, "id": f"child_{page.page_number}_{i}"}
        })
        
    return chunks

from monitoring.tracing import trace_sync

@trace_sync("ingestion.chunk")
def chunk_document(pages: List[ExtractedPage]) -> List[Dict[str, Any]]:
    all_chunks = []
    total_pages = len(pages)
    
    for page in pages:
        strategy = get_chunking_strategy(page, total_pages)
        
        if strategy == "fixed":
            text_chunks = chunk_fixed(page.content)
            for c in text_chunks:
                all_chunks.append({"content": c, "metadata": {**page.metadata, "strategy": "fixed"}})
        elif strategy == "semantic":
            text_chunks = chunk_semantic(page.content)
            for c in text_chunks:
                all_chunks.append({"content": c, "metadata": {**page.metadata, "strategy": "semantic"}})
        elif strategy == "hierarchical":
            h_chunks = chunk_hierarchical(page)
            for c in h_chunks:
                c["metadata"].update({k: v for k, v in page.metadata.items() if k not in c["metadata"]})
                c["metadata"]["strategy"] = "hierarchical"
                all_chunks.append(c)
                
    return all_chunks
