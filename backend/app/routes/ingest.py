"""Document ingestion endpoints."""

import hashlib
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import aiofiles
import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import get_settings
from app.models.schemas import DocumentMetadata, IngestResponse
from app.modules.ingestion import (
    load_document,
    chunk_document,
    register_document,
    unregister_document,
    append_chunks,
    load_manifest,
    rebuild_chunks_file,
    get_active_chunk_ids,
)
from app.modules.retrieval.faiss_index import add_chunks_to_index, rebuild_index
from app.modules.observability.latency import track_latency

router = APIRouter()
logger = structlog.get_logger()

ALLOWED_CONTENT_TYPES = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/html": ".html",
}


def compute_checksum(content: bytes) -> str:
    """Compute SHA-256 checksum of content."""
    return hashlib.sha256(content).hexdigest()


async def save_uploaded_file(
    workspace_path: Path,
    document_id: str,
    filename: str,
    content: bytes,
    version: int,
) -> Path:
    """Save uploaded file to workspace with versioning."""
    # Create document directory
    doc_dir = workspace_path / "documents" / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw file with version
    versioned_filename = f"v{version}_{filename}"
    file_path = doc_dir / versioned_filename
    
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)
    
    return file_path


@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
@track_latency("ingest")
async def ingest_document(
    file: UploadFile = File(..., description="Document to ingest"),
    workspace_id: str = Form(..., description="Target workspace ID"),
    chunk_size: int | None = Form(None, ge=50, le=4000),
    chunk_overlap: int | None = Form(None, ge=0, le=500),
    chunk_strategy: str | None = Form(None, description="Chunking strategy: semantic, token, or paragraph"),
) -> IngestResponse:
    """
    Ingest a document into a workspace.
    
    Supported formats: PDF, TXT, MD, DOCX, HTML
    
    The document will be:
    1. Validated and stored in documents/{doc_id}/
    2. Parsed and chunked → appended to chunks.jsonl
    3. Indexed → vectors added to index.faiss with mapping in index_meta.json
    4. Registered in manifest.json
    """
    start_time = time.perf_counter()
    settings = get_settings()
    
    # Validate content type
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type}. Allowed: {list(ALLOWED_CONTENT_TYPES.keys())}",
        )
    
    # Read file content
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded",
        )
    
    # Setup paths
    workspace_path = settings.workspaces_dir / workspace_id
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    # Generate document ID and compute checksum
    document_id = str(uuid4())
    checksum = compute_checksum(content)
    filename = file.filename or "unknown"
    
    logger.info(
        "Ingesting document",
        workspace_id=workspace_id,
        document_id=document_id,
        filename=filename,
        size_bytes=len(content),
    )
    
    # Check if this exact file already exists (same filename + checksum)
    manifest = await load_manifest(workspace_path)
    for doc_id, doc_info in manifest.get("documents", {}).items():
        if doc_info.get("filename") == filename and doc_info.get("checksum") == checksum:
            # Document unchanged - return existing info
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.info("Document unchanged, skipping", filename=filename)
            
            return IngestResponse(
                workspace_id=workspace_id,
                document_id=doc_id,
                filename=filename,
                chunks_created=doc_info.get("chunk_count", 0),
                version=doc_info.get("version", 1),
                processing_time_ms=round(processing_time, 2),
                metadata=DocumentMetadata(
                    filename=filename,
                    content_type=doc_info.get("content_type", content_type),
                    size_bytes=doc_info.get("size_bytes", len(content)),
                    version=doc_info.get("version", 1),
                    checksum=checksum,
                ),
            )
    
    # Save original file (version will be determined after registration)
    # Use version=1 for now, will be updated
    temp_version = 1
    file_path = await save_uploaded_file(
        workspace_path,
        document_id,
        filename,
        content,
        temp_version,
    )
    
    # Load and parse document
    text_content = await load_document(file_path, content_type)
    
    # Chunk document
    effective_chunk_size = chunk_size or settings.chunk_size
    effective_overlap = chunk_overlap or settings.chunk_overlap
    effective_strategy = chunk_strategy or "semantic"
    
    chunks = chunk_document(
        text_content,
        document_id=document_id,
        chunk_size=effective_chunk_size,
        overlap=effective_overlap,
        strategy=effective_strategy,
    )
    
    # Add filename to chunk metadata
    for chunk in chunks:
        chunk["metadata"]["filename"] = filename
        chunk["metadata"]["workspace_id"] = workspace_id
    
    # Append chunks to chunks.jsonl
    await append_chunks(workspace_path, chunks)
    
    # Add chunks to FAISS index
    await add_chunks_to_index(workspace_path, chunks)
    
    # Register document in manifest
    chunk_ids = [c["chunk_id"] for c in chunks]
    version = await register_document(
        workspace_path=workspace_path,
        document_id=document_id,
        filename=filename,
        checksum=checksum,
        chunk_ids=chunk_ids,
        content_type=content_type,
        size_bytes=len(content),
    )
    
    # Rename file with correct version if needed
    if version != temp_version:
        new_file_path = file_path.parent / f"v{version}_{filename}"
        file_path.rename(new_file_path)
    
    processing_time = (time.perf_counter() - start_time) * 1000
    
    # Create metadata
    metadata = DocumentMetadata(
        filename=filename,
        content_type=content_type,
        size_bytes=len(content),
        created_at=datetime.utcnow(),
        version=version,
        checksum=checksum,
    )
    
    logger.info(
        "Document ingested successfully",
        document_id=document_id,
        chunks_created=len(chunks),
        processing_time_ms=round(processing_time, 2),
    )
    
    return IngestResponse(
        workspace_id=workspace_id,
        document_id=document_id,
        filename=filename,
        chunks_created=len(chunks),
        version=version,
        processing_time_ms=round(processing_time, 2),
        metadata=metadata,
    )


@router.get("/{workspace_id}/documents")
async def list_documents(workspace_id: str) -> dict:
    """List all documents in a workspace from manifest."""
    settings = get_settings()
    workspace_path = settings.workspaces_dir / workspace_id
    
    if not workspace_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found",
        )
    
    manifest = await load_manifest(workspace_path)
    
    documents = []
    for doc_id, doc_info in manifest.get("documents", {}).items():
        documents.append({
            "document_id": doc_id,
            "filename": doc_info.get("filename"),
            "version": doc_info.get("version"),
            "chunk_count": doc_info.get("chunk_count"),
            "size_bytes": doc_info.get("size_bytes"),
            "content_type": doc_info.get("content_type"),
            "created_at": doc_info.get("created_at"),
            "updated_at": doc_info.get("updated_at"),
        })
    
    return {
        "workspace_id": workspace_id,
        "documents": documents,
        "stats": manifest.get("stats", {}),
    }


@router.delete("/{workspace_id}/documents/{document_id}")
async def delete_document(workspace_id: str, document_id: str) -> dict:
    """Delete a document from a workspace and rebuild index."""
    import shutil
    
    settings = get_settings()
    workspace_path = settings.workspaces_dir / workspace_id
    
    if not workspace_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found",
        )
    
    # Unregister from manifest (returns chunk IDs to remove)
    removed_chunks = await unregister_document(workspace_path, document_id)
    
    if not removed_chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in workspace '{workspace_id}'",
        )
    
    # Delete document files
    doc_path = workspace_path / "documents" / document_id
    if doc_path.exists():
        shutil.rmtree(doc_path)
    
    # Get remaining active chunk IDs
    active_chunk_ids = await get_active_chunk_ids(workspace_path)
    
    # Rebuild chunks.jsonl (remove deleted document's chunks)
    await rebuild_chunks_file(workspace_path, active_chunk_ids)
    
    # Rebuild FAISS index
    await rebuild_index(workspace_path)
    
    logger.info(
        "Document deleted",
        workspace_id=workspace_id,
        document_id=document_id,
        chunks_removed=len(removed_chunks),
    )
    
    return {
        "status": "deleted",
        "document_id": document_id,
        "chunks_removed": len(removed_chunks),
    }


@router.post("/{workspace_id}/rebuild-index")
async def rebuild_workspace_index(workspace_id: str) -> dict:
    """Rebuild the FAISS index for a workspace."""
    settings = get_settings()
    workspace_path = settings.workspaces_dir / workspace_id
    
    if not workspace_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found",
        )
    
    vector_count = await rebuild_index(workspace_path)
    
    return {
        "status": "rebuilt",
        "workspace_id": workspace_id,
        "vector_count": vector_count,
    }


@router.get("/{workspace_id}/stats")
async def get_workspace_statistics(workspace_id: str) -> dict:
    """Get workspace statistics including index info."""
    from app.modules.ingestion.storage import get_workspace_stats
    
    settings = get_settings()
    workspace_path = settings.workspaces_dir / workspace_id
    
    if not workspace_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found",
        )
    
    stats = await get_workspace_stats(workspace_path)
    return stats
