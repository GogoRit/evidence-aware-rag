"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ============ Health ============

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    name: str
    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    status: HealthStatus
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: list[ComponentHealth] = []


# ============ Ingestion ============

class DocumentMetadata(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    checksum: str | None = None


class IngestRequest(BaseModel):
    workspace_id: str = Field(..., description="Target workspace identifier")
    chunk_size: int | None = Field(None, ge=100, le=4000, description="Override chunk size")
    chunk_overlap: int | None = Field(None, ge=0, le=500, description="Override chunk overlap")


class IngestResponse(BaseModel):
    workspace_id: str
    document_id: str
    filename: str
    chunks_created: int
    version: int
    processing_time_ms: float
    metadata: DocumentMetadata


# ============ Retrieval ============

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = {}
    embedding: list[float] | None = None


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    rank: int


# ============ Chat ============

class ChatRequest(BaseModel):
    workspace_id: str
    query: str = Field(..., min_length=1, max_length=10000)
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    include_sources: bool = True
    model_preference: str | None = Field(None, description="Override model routing")


class SourceDocument(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = {}


class ConfidenceInfo(BaseModel):
    """Confidence assessment details."""
    level: str  # "high", "low", "insufficient"
    top_score: float
    mean_score: float
    threshold: float
    doc_top_score: float | None = None  # Document-level aggregated confidence (retrieval-only)
    doc_hit_count: int | None = None  # Chunks from best matching document (for diagnostics)
    lexical_match: bool = False  # True if lexical fallback was used
    raw_top_score: float | None = None  # Raw FAISS similarity score (for diagnostics)
    entity_fact_lookup: bool = False  # True if query is an entity-fact lookup intent


class RouterDecisionInfo(BaseModel):
    """Deterministic routing decision details."""
    mode: str  # "generation", "retrieval_only", "refused", "none"
    confidence: float  # Confidence score used in decision
    complexity: float  # Query complexity score (0-1)
    effective_complexity: float  # Combined score for routing
    chosen_model: str  # Selected model or mode name
    reason: str  # Human-readable explanation
    cost_estimate_usd: float  # Estimated cost for this request
    factors: dict[str, Any] = {}  # Detailed breakdown


class ChatResponse(BaseModel):
    answer: str | None = None
    workspace_id: str
    sources: list[SourceDocument] = []
    model_used: str
    latency_ms: float
    cost_usd: float | None = None
    tokens_used: dict[str, int] = {}
    retrieval_only: bool = False  # True when generation is disabled
    refused: bool = False  # True when confidence too low
    refusal_reason: str | None = None  # Explanation for refusal
    confidence: ConfidenceInfo | None = None  # Confidence details
    router_decision: RouterDecisionInfo | None = None  # Full routing decision


# ============ Metrics ============

class CostMetrics(BaseModel):
    total_usd: float
    by_model: dict[str, float] = {}
    period_start: datetime
    period_end: datetime


class LatencyMetrics(BaseModel):
    p50_ms: float
    p90_ms: float
    p99_ms: float
    avg_ms: float


class MetricsResponse(BaseModel):
    requests_total: int
    requests_by_endpoint: dict[str, int] = {}
    latency: LatencyMetrics | None = None
    costs: CostMetrics | None = None
    active_workspaces: int
    documents_ingested: int
    chunks_indexed: int
