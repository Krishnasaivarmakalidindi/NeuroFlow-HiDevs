import pytesseract
from PIL import Image
from typing import List
import base64
from io import BytesIO

from ..models import ExtractedPage
import sys
import os

# Add providers to path if needed or just absolute import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from providers.client import NeuroFlowClient
from providers.router import RoutingCriteria
from providers.base import ChatMessage

async def extract_image(file_path: str) -> List[ExtractedPage]:
    with Image.open(file_path) as img:
        # Convert to RGB if needed
        if img.mode not in ('L', 'RGB'):
            img = img.convert('RGB')
            
        # Resize if max dimension > 1024
        max_dim = 1024
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
        # OCR
        ocr_text = pytesseract.image_to_string(img)
        
        # Vision LLM
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        try:
            client = NeuroFlowClient()
            criteria = RoutingCriteria(require_vision=True)
            
            messages = [
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "Describe this image in detail."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                    ]
                )
            ]
            
            result = await client.chat(messages, criteria)
            vision_description = result.content
        except Exception as e:
            # Fallback if no vision model available or configured properly
            vision_description = f"Vision description unavailable. ({str(e)})"
            
        final_content = f"{vision_description}\n\nText found in image:\n{ocr_text}"
        
        return [ExtractedPage(
            page_number=1,
            content=final_content,
            content_type="image_description",
            metadata={}
        )]
