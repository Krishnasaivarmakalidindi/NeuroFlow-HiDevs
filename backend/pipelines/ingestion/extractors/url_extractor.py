import httpx
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
import trafilatura
from typing import List

from monitoring.tracing import trace_async

@trace_async("ingestion.extract.url")
async def extract_url(url: str) -> List[ExtractedPage]:
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    # Check robots.txt
    rp = RobotFileParser()
    robots_url = f"{base_url}/robots.txt"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            robots_resp = await client.get(robots_url)
            if robots_resp.status_code == 200:
                rp.parse(robots_resp.text.splitlines())
    except Exception:
        # If we can't fetch robots.txt, we proceed with caution or assume allowed.
        rp.allow_all = True
        
    if not getattr(rp, 'allow_all', False) and not rp.can_fetch("*", url):
        raise ValueError(f"Fetching {url} is disallowed by robots.txt")

    # Fetch content
    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        
    html_content = response.text
    
    # Extract
    extracted_text = trafilatura.extract(
        html_content, 
        include_tables=True,
        output_format="txt"
    )
    
    if not extracted_text:
        extracted_text = ""

    # Metadata extraction
    metadata_obj = trafilatura.extract_metadata(html_content, default_url=url)
    
    meta_dict = {}
    if metadata_obj:
        meta_dict = {
            "title": metadata_obj.title,
            "author": metadata_obj.author,
            "canonical_url": metadata_obj.url,
            "publish_date": metadata_obj.date
        }
        
    return [ExtractedPage(
        page_number=1,
        content=extracted_text,
        content_type="text",
        metadata=meta_dict
    )]
