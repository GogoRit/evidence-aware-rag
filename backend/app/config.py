"""Application configuration using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "RAG System"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Data paths (relative to where uvicorn runs, typically backend/)
    data_dir: Path = Path("data")
    workspaces_dir: Path = Path("data/workspaces")

    # Chunking defaults
    chunk_size: int = 512
    chunk_overlap: int = 50

    # LLM Configuration
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    # Generation
    generation_enabled: bool = False  # Set to True to enable LLM generation
    
    # Confidence & Refusal
    confidence_threshold: float = 0.4  # Minimum confidence to answer (0-1)
    refusal_threshold: float = 0.25  # Below this, refuse entirely even with citations
    refusal_message: str = "I don't have enough confidence in the available information to answer this question accurately."
    
    # Model routing
    cheap_model: str = "gpt-3.5-turbo"
    expensive_model: str = "gpt-4-turbo-preview"
    complexity_threshold: float = 0.7  # Route to expensive model above this

    # Embeddings
    # "local" = sentence-transformers (all-MiniLM-L6-v2), "openai" = OpenAI ada-002
    # Defaults to "local" if OPENAI_API_KEY is not set
    embeddings_backend: Literal["local", "openai"] = "local"
    local_embedding_model: str = "all-MiniLM-L6-v2"  # 384 dimensions
    
    # FAISS
    embedding_dim: int = 384  # Default for local model; 1536 for OpenAI ada-002
    faiss_index_type: str = "Flat"
    
    @property
    def effective_embeddings_backend(self) -> str:
        """Return actual backend to use based on config and available keys."""
        if self.embeddings_backend == "openai" and self.openai_api_key:
            return "openai"
        return "local"
    
    @property
    def effective_embedding_dim(self) -> int:
        """Return embedding dimension based on effective backend."""
        if self.effective_embeddings_backend == "openai":
            return 1536  # OpenAI ada-002
        return 384  # all-MiniLM-L6-v2

    # Observability
    enable_metrics: bool = True
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
