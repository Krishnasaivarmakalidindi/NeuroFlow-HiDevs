import pypdfium2 as pdfium
import pdfplumber
import pytesseract
from PIL import Image
from typing import List

from ..models import ExtractedPage

def extract_pdf(file_path: str) -> List[ExtractedPage]:
    pages = []
    
    # Extract text using pypdfium2
    pdf = pdfium.PdfDocument(file_path)
    
    for i, page in enumerate(pdf):
        text_page = page.get_textpage()
        text = text_page.get_text_range()
        
        # Scanned PDF check
        if len(text.strip()) < 50:
            bitmap = page.render(scale=2.0)
            pil_image = bitmap.to_pil()
            text = pytesseract.image_to_string(pil_image, config="--psm 6")
        
        if text.strip():
            pages.append(ExtractedPage(
                page_number=i + 1,
                content=text,
                content_type="text",
                metadata={"page": i + 1}
            ))
            
    # Extract tables using pdfplumber
    with pdfplumber.open(file_path) as plumber_pdf:
        for i, p_page in enumerate(plumber_pdf.pages):
            tables = p_page.extract_tables()
            for table in tables:
                if not table:
                    continue
                # Convert table to markdown
                md_table = []
                for row in table:
                    cleaned_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                    md_table.append("| " + " | ".join(cleaned_row) + " |")
                
                # Add markdown header separator
                if md_table:
                    headers = md_table[0]
                    separator = "|---" * headers.count("|") + "|"
                    md_table.insert(1, separator)
                
                table_text = "\n".join(md_table)
                pages.append(ExtractedPage(
                    page_number=i + 1,
                    content=table_text,
                    content_type="table",
                    metadata={"page": i + 1}
                ))

    return pages
