"""Document loaders for various file formats."""

from pathlib import Path

import structlog

logger = structlog.get_logger()


async def load_pdf(file_path: Path) -> str:
    """Load text content from PDF file."""
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(file_path)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_parts.append(f"[Page {page_num + 1}]\n{text}")
        
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error("Failed to load PDF", path=str(file_path), error=str(e))
        raise


async def load_docx(file_path: Path) -> str:
    """Load text content from DOCX file."""
    try:
        from docx import Document
        
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        logger.error("Failed to load DOCX", path=str(file_path), error=str(e))
        raise


async def load_html(file_path: Path) -> str:
    """Load text content from HTML file."""
    try:
        from bs4 import BeautifulSoup
        
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator="\n")
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        return "\n".join(line for line in lines if line)
    except Exception as e:
        logger.error("Failed to load HTML", path=str(file_path), error=str(e))
        raise


async def load_text(file_path: Path) -> str:
    """Load content from plain text or markdown file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


LOADERS = {
    "application/pdf": load_pdf,
    "text/plain": load_text,
    "text/markdown": load_text,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": load_docx,
    "text/html": load_html,
}


async def load_document(file_path: Path, content_type: str) -> str:
    """
    Load document content based on file type.
    
    Args:
        file_path: Path to the document file
        content_type: MIME type of the document
    
    Returns:
        Extracted text content
    """
    loader = LOADERS.get(content_type)
    
    if not loader:
        # Fallback to text loader
        logger.warning(
            "No specific loader for content type, using text loader",
            content_type=content_type,
        )
        loader = load_text
    
    logger.info("Loading document", path=str(file_path), content_type=content_type)
    return await loader(file_path)
