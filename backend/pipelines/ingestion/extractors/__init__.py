from .pdf_extractor import extract_pdf
from .docx_extractor import extract_docx
from .image_extractor import extract_image
from .csv_extractor import extract_csv
from .url_extractor import extract_url
from .pptx_extractor import extract_pptx

async def extract_content(file_path_or_url: str, source_type: str):
    if source_type == "pdf":
        return extract_pdf(file_path_or_url)
    elif source_type == "docx":
        return extract_docx(file_path_or_url)
    elif source_type in ["jpg", "jpeg", "png", "webp"]:
        return await extract_image(file_path_or_url)
    elif source_type == "csv":
        return extract_csv(file_path_or_url)
    elif source_type == "url":
        return await extract_url(file_path_or_url)
    elif source_type == "pptx":
        return await extract_pptx(file_path_or_url)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")
