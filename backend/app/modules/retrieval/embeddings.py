"""Embedding backends for vector search.

Supports:
- Local: sentence-transformers (all-MiniLM-L6-v2) - no API key needed
- OpenAI: text-embedding-ada-002 - requires OPENAI_API_KEY
"""

from functools import lru_cache
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()

# Global model cache for local embeddings
_local_model: Any = None


def _get_local_model():
    """Lazily load and cache the local embedding model."""
    global _local_model
    
    if _local_model is None:
        from app.config import get_settings
        settings = get_settings()
        
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = settings.local_embedding_model
            logger.info("Loading local embedding model", model=model_name)
            
            _local_model = SentenceTransformer(model_name)
            
            logger.info(
                "Local embedding model loaded",
                model=model_name,
                dimension=_local_model.get_sentence_embedding_dimension(),
            )
        except ImportError as e:
            logger.error(
                "sentence-transformers not installed. Run: pip install sentence-transformers",
                error=str(e),
            )
            raise
    
    return _local_model


def get_embedding_local(text: str) -> np.ndarray:
    """Get embedding using local sentence-transformers model."""
    model = _get_local_model()
    
    # Truncate to model's max sequence length (typically 256-512 tokens)
    truncated = text[:2000]  # Rough character limit
    
    embedding = model.encode(truncated, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.astype(np.float32)


def get_embeddings_batch_local(texts: list[str]) -> np.ndarray:
    """Get embeddings for multiple texts using local model."""
    model = _get_local_model()
    
    # Truncate texts
    truncated = [t[:2000] for t in texts]
    
    embeddings = model.encode(
        truncated,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def get_embedding_openai(text: str, api_key: str) -> np.ndarray:
    """Get embedding using OpenAI API."""
    import openai
    
    client = openai.OpenAI(api_key=api_key)
    
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text[:8000],  # API limit
    )
    
    return np.array(response.data[0].embedding, dtype=np.float32)


def get_embeddings_batch_openai(texts: list[str], api_key: str) -> np.ndarray:
    """Get embeddings for multiple texts using OpenAI API."""
    import openai
    
    client = openai.OpenAI(api_key=api_key)
    
    # Truncate texts to API limit
    truncated = [t[:8000] for t in texts]
    
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=truncated,
    )
    
    embeddings = [np.array(d.embedding, dtype=np.float32) for d in response.data]
    return np.array(embeddings)


async def get_embedding(text: str) -> np.ndarray:
    """
    Get embedding for text using the configured backend.
    
    Automatically selects backend based on:
    - EMBEDDINGS_BACKEND setting
    - Availability of OPENAI_API_KEY
    """
    from app.config import get_settings
    
    settings = get_settings()
    backend = settings.effective_embeddings_backend
    
    if backend == "openai":
        try:
            return get_embedding_openai(text, settings.openai_api_key)
        except Exception as e:
            logger.error("OpenAI embedding failed, falling back to local", error=str(e))
            return get_embedding_local(text)
    else:
        return get_embedding_local(text)


async def get_embeddings_batch(texts: list[str]) -> np.ndarray:
    """
    Get embeddings for multiple texts using the configured backend.
    """
    from app.config import get_settings
    
    settings = get_settings()
    backend = settings.effective_embeddings_backend
    
    logger.debug(
        "Getting batch embeddings",
        backend=backend,
        count=len(texts),
    )
    
    if backend == "openai":
        try:
            return get_embeddings_batch_openai(texts, settings.openai_api_key)
        except Exception as e:
            logger.error("OpenAI batch embedding failed, falling back to local", error=str(e))
            return get_embeddings_batch_local(texts)
    else:
        return get_embeddings_batch_local(texts)


def get_embedding_dimension() -> int:
    """Get the embedding dimension for the current backend."""
    from app.config import get_settings
    return get_settings().effective_embedding_dim
