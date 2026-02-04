"""FAISS vector index management with persistent storage."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from app.modules.ingestion.storage import (
    load_index_meta,
    update_index_meta,
    load_chunks_by_ids,
    get_chunk_id_for_vector,
)
from app.modules.retrieval.embeddings import (
    get_embedding,
    get_embeddings_batch,
    get_embedding_dimension,
)

logger = structlog.get_logger()


def create_index(dimension: int = 1536, index_type: str = "Flat") -> Any:
    """
    Create a new FAISS index.
    
    Args:
        dimension: Vector dimension (1536 for OpenAI ada-002)
        index_type: Index type - "Flat" (exact), "IVFFlat" (approximate)
    
    Returns:
        FAISS index object
    """
    import faiss
    
    if index_type == "Flat":
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    elif index_type == "IVFFlat":
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    logger.info("Created FAISS index", dimension=dimension, index_type=index_type)
    return index


def save_index(index: Any, workspace_path: Path) -> None:
    """Save FAISS index to workspace."""
    import faiss
    
    index_path = workspace_path / "index.faiss"
    faiss.write_index(index, str(index_path))
    logger.info("Saved FAISS index", path=str(index_path), ntotal=index.ntotal)


def load_index(workspace_path: Path) -> Any | None:
    """Load FAISS index from workspace."""
    import faiss
    
    index_path = workspace_path / "index.faiss"
    
    if not index_path.exists():
        logger.debug("No index found", path=str(index_path))
        return None
    
    index = faiss.read_index(str(index_path))
    logger.info("Loaded FAISS index", path=str(index_path), ntotal=index.ntotal)
    return index


async def add_chunks_to_index(
    workspace_path: Path,
    chunks: list[dict[str, Any]],
) -> int:
    """
    Add chunks to the workspace FAISS index.
    
    Args:
        workspace_path: Path to workspace
        chunks: List of chunk dictionaries with 'chunk_id' and 'content'
    
    Returns:
        Number of vectors added
    """
    import faiss
    from app.config import get_settings
    
    settings = get_settings()
    
    if not chunks:
        return 0
    
    # Get embedding dimension from the configured backend
    embed_dim = get_embedding_dimension()
    
    # Load or create index
    index = load_index(workspace_path)
    if index is None:
        index = create_index(embed_dim, settings.faiss_index_type)
    
    # Get current vector count (this will be the starting ID for new vectors)
    start_id = index.ntotal
    
    # Get embeddings for chunks
    texts = [chunk["content"][:1000] for chunk in chunks]  # Truncate for embedding
    embeddings = await get_embeddings_batch(texts)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Train index if needed (IVF indices)
    if hasattr(index, "is_trained") and not index.is_trained:
        if len(embeddings) >= 100:  # Need enough vectors to train
            logger.info("Training index", n_vectors=len(embeddings))
            index.train(embeddings)
        else:
            # Fall back to flat index if not enough vectors
            logger.warning("Not enough vectors to train IVF, using Flat index")
            index = create_index(embed_dim, "Flat")
    
    # Add vectors to index
    index.add(embeddings)
    
    # Create ID mappings
    mappings = [
        (start_id + i, chunk["chunk_id"])
        for i, chunk in enumerate(chunks)
    ]
    
    # Update index metadata
    await update_index_meta(workspace_path, mappings)
    
    # Save index
    save_index(index, workspace_path)
    
    logger.info(
        "Added chunks to index",
        chunks_added=len(chunks),
        total_vectors=index.ntotal,
    )
    
    return len(chunks)


async def search_similar(
    workspace_path: Path,
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Search for similar documents using the persisted FAISS index.
    
    Args:
        workspace_path: Path to workspace
        query: Search query
        top_k: Number of results to return
    
    Returns:
        List of matching chunks with scores
    """
    import faiss
    
    # Load index
    index = load_index(workspace_path)
    if index is None or index.ntotal == 0:
        logger.warning("No index or empty index", path=str(workspace_path))
        return []
    
    # Load index metadata for ID mapping
    meta = await load_index_meta(workspace_path)
    
    if not meta.get("id_to_chunk"):
        logger.warning("No index metadata found", path=str(workspace_path))
        return []
    
    # Get query embedding
    query_embedding = await get_embedding(query)
    query_norm = query_embedding.reshape(1, -1).copy()
    faiss.normalize_L2(query_norm)
    
    # Search
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_norm, k)
    
    # Map vector IDs to chunk IDs
    chunk_ids_to_fetch = set()
    vector_results = []  # (score, vector_id)
    
    for score, vector_id in zip(scores[0], indices[0]):
        if vector_id >= 0:
            chunk_id = get_chunk_id_for_vector(meta, int(vector_id))
            if chunk_id:
                chunk_ids_to_fetch.add(chunk_id)
                vector_results.append((float(score), chunk_id))
    
    if not chunk_ids_to_fetch:
        return []
    
    # Load chunk data
    chunks = await load_chunks_by_ids(workspace_path, chunk_ids_to_fetch)
    chunk_map = {c["chunk_id"]: c for c in chunks}
    
    # Build results with scores
    results = []
    for score, chunk_id in vector_results:
        if chunk_id in chunk_map:
            chunk = chunk_map[chunk_id]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "content": chunk["content"],
                "score": score,
                "metadata": chunk.get("metadata", {}),
            })
    
    logger.info(
        "Search completed",
        query_length=len(query),
        results_found=len(results),
        index_size=index.ntotal,
    )
    
    return results


async def rebuild_index(workspace_path: Path) -> int:
    """
    Rebuild the entire FAISS index from chunks.jsonl.
    
    Use after document deletion or index corruption.
    
    Returns:
        Number of vectors in rebuilt index
    """
    import faiss
    from app.config import get_settings
    from app.modules.ingestion.storage import load_all_chunks, get_active_chunk_ids
    
    settings = get_settings()
    
    # Get active chunk IDs from manifest
    active_ids = await get_active_chunk_ids(workspace_path)
    
    # Load all chunks and filter to active ones
    all_chunks = await load_all_chunks(workspace_path)
    active_chunks = [c for c in all_chunks if c["chunk_id"] in active_ids]
    
    if not active_chunks:
        logger.warning("No active chunks to index")
        # Remove old index
        index_path = workspace_path / "index.faiss"
        if index_path.exists():
            index_path.unlink()
        return 0
    
    # Get embedding dimension from the configured backend
    embed_dim = get_embedding_dimension()
    
    # Create fresh index
    index = create_index(embed_dim, "Flat")  # Use Flat for rebuild
    
    # Get embeddings
    texts = [chunk["content"][:1000] for chunk in active_chunks]
    embeddings = await get_embeddings_batch(texts)
    faiss.normalize_L2(embeddings)
    
    # Add to index
    index.add(embeddings)
    
    # Create fresh metadata
    mappings = [
        (i, chunk["chunk_id"])
        for i, chunk in enumerate(active_chunks)
    ]
    
    # Clear and rebuild index_meta
    meta = {
        "version": 1,
        "created_at": None,
        "updated_at": None,
        "vector_count": 0,
        "id_to_chunk": {},
        "chunk_to_id": {},
    }
    from app.modules.ingestion.storage import save_index_meta
    await save_index_meta(workspace_path, meta)
    await update_index_meta(workspace_path, mappings)
    
    # Save index
    save_index(index, workspace_path)
    
    logger.info("Index rebuilt", total_vectors=index.ntotal)
    return index.ntotal
