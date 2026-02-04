"""Workspace storage schema management.

Standard workspace structure:
    {workspace_id}/
    ├── manifest.json      # Document registry and metadata
    ├── chunks.jsonl       # All chunks (one JSON object per line)
    ├── index.faiss        # FAISS vector index
    ├── index_meta.json    # Vector ID → chunk_id mapping
    └── documents/         # Original uploaded files
        └── {doc_id}/
            └── v{n}_{filename}
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import structlog

logger = structlog.get_logger()


# ============ Manifest ============

async def load_manifest(workspace_path: Path) -> dict[str, Any]:
    """Load or create workspace manifest."""
    manifest_path = workspace_path / "manifest.json"
    
    if manifest_path.exists():
        async with aiofiles.open(manifest_path, "r") as f:
            content = await f.read()
            return json.loads(content)
    
    return {
        "schema_version": "1.0",
        "workspace_id": workspace_path.name,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "documents": {},
        "stats": {
            "total_documents": 0,
            "total_chunks": 0,
            "index_built": False,
        },
    }


async def save_manifest(workspace_path: Path, manifest: dict[str, Any]) -> None:
    """Save workspace manifest."""
    manifest_path = workspace_path / "manifest.json"
    manifest["updated_at"] = datetime.utcnow().isoformat()
    
    # Update stats
    manifest["stats"]["total_documents"] = len(manifest.get("documents", {}))
    manifest["stats"]["total_chunks"] = sum(
        doc.get("chunk_count", 0) for doc in manifest.get("documents", {}).values()
    )
    
    async with aiofiles.open(manifest_path, "w") as f:
        await f.write(json.dumps(manifest, indent=2))
    
    logger.debug("Manifest saved", workspace=workspace_path.name)


async def register_document(
    workspace_path: Path,
    document_id: str,
    filename: str,
    checksum: str,
    chunk_ids: list[str],
    content_type: str,
    size_bytes: int,
) -> int:
    """
    Register a document in the manifest.
    
    Returns:
        Version number assigned to this document
    """
    manifest = await load_manifest(workspace_path)
    
    # Check for existing document with same filename
    existing_doc = None
    for doc_id, doc_info in manifest["documents"].items():
        if doc_info.get("filename") == filename:
            existing_doc = (doc_id, doc_info)
            break
    
    if existing_doc:
        doc_id, doc_info = existing_doc
        # Check if content changed
        if doc_info.get("checksum") == checksum:
            logger.info("Document unchanged", filename=filename, version=doc_info["version"])
            return doc_info["version"]
        
        # New version - mark old chunks as superseded
        new_version = doc_info.get("version", 1) + 1
        doc_info["previous_versions"] = doc_info.get("previous_versions", [])
        doc_info["previous_versions"].append({
            "version": doc_info["version"],
            "checksum": doc_info["checksum"],
            "chunk_ids": doc_info["chunk_ids"],
            "superseded_at": datetime.utcnow().isoformat(),
        })
        
        # Update to new version
        doc_info["version"] = new_version
        doc_info["checksum"] = checksum
        doc_info["chunk_ids"] = chunk_ids
        doc_info["chunk_count"] = len(chunk_ids)
        doc_info["updated_at"] = datetime.utcnow().isoformat()
        
        logger.info("Document updated", filename=filename, version=new_version)
    else:
        # New document
        new_version = 1
        manifest["documents"][document_id] = {
            "document_id": document_id,
            "filename": filename,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "checksum": checksum,
            "version": new_version,
            "chunk_ids": chunk_ids,
            "chunk_count": len(chunk_ids),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "previous_versions": [],
        }
        logger.info("Document registered", filename=filename, document_id=document_id)
    
    await save_manifest(workspace_path, manifest)
    return new_version


async def unregister_document(workspace_path: Path, document_id: str) -> list[str]:
    """
    Remove a document from manifest.
    
    Returns:
        List of chunk_ids that should be removed from the index
    """
    manifest = await load_manifest(workspace_path)
    
    if document_id not in manifest["documents"]:
        return []
    
    doc_info = manifest["documents"].pop(document_id)
    chunk_ids = doc_info.get("chunk_ids", [])
    
    # Also collect chunk_ids from previous versions
    for prev in doc_info.get("previous_versions", []):
        chunk_ids.extend(prev.get("chunk_ids", []))
    
    await save_manifest(workspace_path, manifest)
    
    logger.info("Document unregistered", document_id=document_id, chunks_removed=len(chunk_ids))
    return chunk_ids


# ============ Chunks (JSONL) ============

async def append_chunks(workspace_path: Path, chunks: list[dict[str, Any]]) -> None:
    """Append chunks to chunks.jsonl file."""
    chunks_path = workspace_path / "chunks.jsonl"
    
    async with aiofiles.open(chunks_path, "a") as f:
        for chunk in chunks:
            await f.write(json.dumps(chunk) + "\n")
    
    logger.debug("Appended chunks", count=len(chunks), workspace=workspace_path.name)


async def load_all_chunks(workspace_path: Path) -> list[dict[str, Any]]:
    """Load all chunks from chunks.jsonl."""
    chunks_path = workspace_path / "chunks.jsonl"
    
    if not chunks_path.exists():
        return []
    
    chunks = []
    async with aiofiles.open(chunks_path, "r") as f:
        async for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    
    return chunks


async def load_chunks_by_ids(workspace_path: Path, chunk_ids: set[str]) -> list[dict[str, Any]]:
    """Load specific chunks by their IDs."""
    chunks_path = workspace_path / "chunks.jsonl"
    
    if not chunks_path.exists():
        return []
    
    chunks = []
    async with aiofiles.open(chunks_path, "r") as f:
        async for line in f:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                if chunk.get("chunk_id") in chunk_ids:
                    chunks.append(chunk)
    
    return chunks


async def rebuild_chunks_file(workspace_path: Path, valid_chunk_ids: set[str]) -> int:
    """
    Rebuild chunks.jsonl keeping only valid chunks.
    Used after document deletion to clean up orphaned chunks.
    
    Returns:
        Number of chunks retained
    """
    chunks_path = workspace_path / "chunks.jsonl"
    temp_path = workspace_path / "chunks.jsonl.tmp"
    
    if not chunks_path.exists():
        return 0
    
    retained = 0
    async with aiofiles.open(chunks_path, "r") as f_in:
        async with aiofiles.open(temp_path, "w") as f_out:
            async for line in f_in:
                line = line.strip()
                if line:
                    chunk = json.loads(line)
                    if chunk.get("chunk_id") in valid_chunk_ids:
                        await f_out.write(json.dumps(chunk) + "\n")
                        retained += 1
    
    # Atomic replace
    temp_path.replace(chunks_path)
    
    logger.info("Rebuilt chunks file", retained=retained, workspace=workspace_path.name)
    return retained


# ============ Index Metadata ============

async def load_index_meta(workspace_path: Path) -> dict[str, Any]:
    """Load index metadata (vector_id → chunk_id mapping)."""
    meta_path = workspace_path / "index_meta.json"
    
    if not meta_path.exists():
        return {
            "version": 1,
            "created_at": None,
            "updated_at": None,
            "vector_count": 0,
            "id_to_chunk": {},  # str(vector_id) → chunk_id
            "chunk_to_id": {},  # chunk_id → vector_id
        }
    
    async with aiofiles.open(meta_path, "r") as f:
        content = await f.read()
        return json.loads(content)


async def save_index_meta(workspace_path: Path, meta: dict[str, Any]) -> None:
    """Save index metadata."""
    meta_path = workspace_path / "index_meta.json"
    meta["updated_at"] = datetime.utcnow().isoformat()
    
    async with aiofiles.open(meta_path, "w") as f:
        await f.write(json.dumps(meta, indent=2))
    
    logger.debug("Index meta saved", vector_count=meta.get("vector_count", 0))


async def update_index_meta(
    workspace_path: Path,
    new_mappings: list[tuple[int, str]],  # (vector_id, chunk_id)
) -> dict[str, Any]:
    """
    Update index metadata with new vector→chunk mappings.
    
    Args:
        workspace_path: Workspace path
        new_mappings: List of (vector_id, chunk_id) tuples
    
    Returns:
        Updated metadata
    """
    meta = await load_index_meta(workspace_path)
    
    if meta["created_at"] is None:
        meta["created_at"] = datetime.utcnow().isoformat()
    
    for vector_id, chunk_id in new_mappings:
        meta["id_to_chunk"][str(vector_id)] = chunk_id
        meta["chunk_to_id"][chunk_id] = vector_id
    
    meta["vector_count"] = len(meta["id_to_chunk"])
    
    await save_index_meta(workspace_path, meta)
    return meta


def get_chunk_id_for_vector(meta: dict[str, Any], vector_id: int) -> str | None:
    """Get chunk_id for a vector index."""
    return meta.get("id_to_chunk", {}).get(str(vector_id))


# ============ Workspace Utilities ============

async def get_workspace_stats(workspace_path: Path) -> dict[str, Any]:
    """Get workspace statistics."""
    manifest = await load_manifest(workspace_path)
    meta = await load_index_meta(workspace_path)
    
    index_path = workspace_path / "index.faiss"
    
    return {
        "workspace_id": workspace_path.name,
        "total_documents": manifest["stats"]["total_documents"],
        "total_chunks": manifest["stats"]["total_chunks"],
        "indexed_vectors": meta.get("vector_count", 0),
        "index_exists": index_path.exists(),
        "index_size_bytes": index_path.stat().st_size if index_path.exists() else 0,
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
    }


async def get_active_chunk_ids(workspace_path: Path) -> set[str]:
    """Get all active (non-superseded) chunk IDs from manifest."""
    manifest = await load_manifest(workspace_path)
    
    chunk_ids = set()
    for doc_info in manifest.get("documents", {}).values():
        chunk_ids.update(doc_info.get("chunk_ids", []))
    
    return chunk_ids
