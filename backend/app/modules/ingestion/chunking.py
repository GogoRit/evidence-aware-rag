"""Document chunking strategies."""

import re
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger()


def split_by_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitting - handles common cases
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


def split_by_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_by_tokens(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[str]:
    """
    Chunk text by approximate token count.
    
    Uses tiktoken for accurate token counting when available,
    falls back to word-based approximation otherwise.
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap
        
        return chunks
    except ImportError:
        # Fallback to word-based chunking (rough approximation: 1 token ≈ 0.75 words)
        words = text.split()
        word_chunk_size = int(chunk_size * 0.75)
        word_overlap = int(overlap * 0.75)
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + word_chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            start = end - word_overlap
        
        return chunks


def semantic_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[str]:
    """
    Semantic chunking that respects document structure.
    
    Tries to split on:
    1. Section headers (markdown-style)
    2. Paragraph boundaries
    3. Sentence boundaries
    4. Falls back to token-based if needed
    """
    # Try to identify sections
    section_pattern = re.compile(r'^#{1,6}\s+.+$', re.MULTILINE)
    sections = section_pattern.split(text)
    section_headers = section_pattern.findall(text)
    
    chunks = []
    
    if len(sections) > 1:
        # Document has sections
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            # Add header to section if available
            header = section_headers[i - 1] if i > 0 and i <= len(section_headers) else ""
            section_text = f"{header}\n{section}".strip() if header else section.strip()
            
            # If section is too large, chunk it further
            if len(section_text.split()) > chunk_size:
                sub_chunks = chunk_by_tokens(section_text, chunk_size, overlap)
                chunks.extend(sub_chunks)
            else:
                chunks.append(section_text)
    else:
        # No clear sections, use paragraph-aware chunking
        paragraphs = split_by_paragraphs(text)
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para.split())
            
            if current_size + para_size > chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Keep overlap
                overlap_text = current_chunk[-1] if current_chunk else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_size = len(overlap_text.split()) if overlap_text else 0
            
            current_chunk.append(para)
            current_size += para_size
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
    
    return [c for c in chunks if c.strip()]


def chunk_document(
    content: str,
    document_id: str,
    chunk_size: int = 512,
    overlap: int = 50,
    strategy: str = "semantic",
) -> list[dict[str, Any]]:
    """
    Chunk a document into smaller pieces for embedding.
    
    Args:
        content: Raw text content
        document_id: ID of the source document
        chunk_size: Target chunk size in tokens
        overlap: Number of overlapping tokens between chunks
        strategy: Chunking strategy ("semantic", "token", "paragraph")
    
    Returns:
        List of chunk dictionaries with metadata
    """
    logger.info(
        "Chunking document",
        document_id=document_id,
        content_length=len(content),
        strategy=strategy,
    )
    
    if strategy == "token":
        raw_chunks = chunk_by_tokens(content, chunk_size, overlap)
    elif strategy == "paragraph":
        raw_chunks = split_by_paragraphs(content)
    else:
        raw_chunks = semantic_chunk(content, chunk_size, overlap)
    
    chunks = []
    for i, chunk_content in enumerate(raw_chunks):
        chunk = {
            "chunk_id": str(uuid4()),
            "document_id": document_id,
            "content": chunk_content,
            "index": i,
            "metadata": {
                "chunk_size": len(chunk_content.split()),
                "position": f"{i + 1}/{len(raw_chunks)}",
            },
        }
        chunks.append(chunk)
    
    logger.info(
        "Document chunked",
        document_id=document_id,
        num_chunks=len(chunks),
    )
    
    return chunks
