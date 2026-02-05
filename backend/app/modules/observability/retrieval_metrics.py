"""
Retrieval metrics and observability for RAG pipelines.

This module provides lightweight in-memory counters and structured logging
for monitoring retrieval behavior in production. No external services required.

Metrics tracked:
- Query outcomes: refused vs allowed, with confidence breakdown
- Confidence distribution: HIGH vs LOW vs INSUFFICIENT rates
- Guardrail triggers: entity-fact lookup rate, lexical fallback rate
- Document aggregation: doc_hit_count distribution

Usage:
    from app.modules.observability.retrieval_metrics import (
        RetrievalMetrics,
        get_retrieval_metrics,
        log_retrieval_event,
    )
    
    # Log a retrieval event
    log_retrieval_event(
        workspace_id="demo",
        query="What is the vacation policy?",
        confidence_level="high",
        refused=False,
        doc_top_score=0.75,
        chunk_top_score=0.72,
        doc_hit_count=3,
        lexical_match=False,
        entity_fact_lookup=False,
    )
    
    # Get current metrics
    metrics = get_retrieval_metrics()
    print(f"Refusal rate: {metrics.refusal_rate:.1%}")

Thread Safety:
    All counters use threading.Lock for safe concurrent access.
    Reset operations are atomic.

Note:
    Metrics are in-memory only and reset on server restart.
    For persistent metrics, consider exporting to external systems.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class RetrievalMetrics:
    """
    In-memory counters for retrieval behavior metrics.
    
    All counters are thread-safe and can be safely accessed
    from multiple request handlers.
    
    Attributes:
        total_queries: Total number of queries processed
        refused_queries: Number of queries that were refused
        high_confidence: Number of HIGH confidence results
        low_confidence: Number of LOW confidence results
        insufficient_confidence: Number of INSUFFICIENT confidence results
        entity_fact_lookups: Number of queries classified as entity-fact lookups
        lexical_fallbacks: Number of times lexical match was used as fallback
        total_doc_hits: Sum of doc_hit_count across all queries (for averaging)
    """
    total_queries: int = 0
    refused_queries: int = 0
    high_confidence: int = 0
    low_confidence: int = 0
    insufficient_confidence: int = 0
    entity_fact_lookups: int = 0
    entity_fact_refusals: int = 0
    lexical_fallbacks: int = 0
    total_doc_hits: int = 0
    
    # Internal lock for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    @property
    def refusal_rate(self) -> float:
        """Percentage of queries that were refused."""
        if self.total_queries == 0:
            return 0.0
        return self.refused_queries / self.total_queries
    
    @property
    def high_confidence_rate(self) -> float:
        """Percentage of queries with HIGH confidence."""
        if self.total_queries == 0:
            return 0.0
        return self.high_confidence / self.total_queries
    
    @property
    def low_confidence_rate(self) -> float:
        """Percentage of queries with LOW confidence."""
        if self.total_queries == 0:
            return 0.0
        return self.low_confidence / self.total_queries
    
    @property
    def insufficient_confidence_rate(self) -> float:
        """Percentage of queries with INSUFFICIENT confidence."""
        if self.total_queries == 0:
            return 0.0
        return self.insufficient_confidence / self.total_queries
    
    @property
    def entity_fact_lookup_rate(self) -> float:
        """Percentage of queries classified as entity-fact lookups."""
        if self.total_queries == 0:
            return 0.0
        return self.entity_fact_lookups / self.total_queries
    
    @property
    def lexical_fallback_rate(self) -> float:
        """Percentage of queries where lexical match was used as fallback."""
        if self.total_queries == 0:
            return 0.0
        return self.lexical_fallbacks / self.total_queries
    
    @property
    def avg_doc_hit_count(self) -> float:
        """Average doc_hit_count per query."""
        if self.total_queries == 0:
            return 0.0
        return self.total_doc_hits / self.total_queries
    
    def to_dict(self) -> dict[str, Any]:
        """
        Export metrics as a dictionary for JSON serialization.
        
        Returns:
            Dictionary with all metrics and computed rates
        """
        return {
            "total_queries": self.total_queries,
            "refused_queries": self.refused_queries,
            "refusal_rate": round(self.refusal_rate, 4),
            "confidence_distribution": {
                "high": self.high_confidence,
                "high_rate": round(self.high_confidence_rate, 4),
                "low": self.low_confidence,
                "low_rate": round(self.low_confidence_rate, 4),
                "insufficient": self.insufficient_confidence,
                "insufficient_rate": round(self.insufficient_confidence_rate, 4),
            },
            "guardrails": {
                "entity_fact_lookups": self.entity_fact_lookups,
                "entity_fact_lookup_rate": round(self.entity_fact_lookup_rate, 4),
                "entity_fact_refusals": self.entity_fact_refusals,
                "lexical_fallbacks": self.lexical_fallbacks,
                "lexical_fallback_rate": round(self.lexical_fallback_rate, 4),
            },
            "document_aggregation": {
                "total_doc_hits": self.total_doc_hits,
                "avg_doc_hit_count": round(self.avg_doc_hit_count, 2),
            },
        }


# Global metrics instance
_metrics = RetrievalMetrics()


def get_retrieval_metrics() -> RetrievalMetrics:
    """
    Get the global retrieval metrics instance.
    
    Returns:
        The singleton RetrievalMetrics instance
    """
    return _metrics


def reset_retrieval_metrics() -> None:
    """
    Reset all retrieval metrics to zero.
    
    Thread-safe operation that atomically resets all counters.
    """
    global _metrics
    with _metrics._lock:
        _metrics.total_queries = 0
        _metrics.refused_queries = 0
        _metrics.high_confidence = 0
        _metrics.low_confidence = 0
        _metrics.insufficient_confidence = 0
        _metrics.entity_fact_lookups = 0
        _metrics.entity_fact_refusals = 0
        _metrics.lexical_fallbacks = 0
        _metrics.total_doc_hits = 0
    
    logger.info("Retrieval metrics reset")


def record_retrieval_event(
    confidence_level: str,
    refused: bool,
    entity_fact_lookup: bool = False,
    lexical_fallback: bool = False,
    doc_hit_count: int = 0,
) -> None:
    """
    Record a retrieval event in the metrics counters.
    
    Thread-safe operation that increments appropriate counters.
    
    Args:
        confidence_level: "high", "low", or "insufficient"
        refused: Whether the query was refused
        entity_fact_lookup: Whether query was classified as entity-fact lookup
        lexical_fallback: Whether lexical match was used as fallback
        doc_hit_count: Number of chunks from best matching document
    """
    with _metrics._lock:
        _metrics.total_queries += 1
        
        if refused:
            _metrics.refused_queries += 1
        
        level_lower = confidence_level.lower()
        if level_lower == "high":
            _metrics.high_confidence += 1
        elif level_lower == "low":
            _metrics.low_confidence += 1
        elif level_lower == "insufficient":
            _metrics.insufficient_confidence += 1
        
        if entity_fact_lookup:
            _metrics.entity_fact_lookups += 1
            if refused:
                _metrics.entity_fact_refusals += 1
        
        if lexical_fallback:
            _metrics.lexical_fallbacks += 1
        
        _metrics.total_doc_hits += doc_hit_count


def log_retrieval_event(
    workspace_id: str,
    query: str,
    confidence_level: str,
    refused: bool,
    doc_top_score: float | None = None,
    chunk_top_score: float | None = None,
    raw_top_score: float | None = None,
    doc_hit_count: int | None = None,
    lexical_match: bool = False,
    entity_fact_lookup: bool = False,
    num_sources: int = 0,
) -> None:
    """
    Log a retrieval event with structured fields and update metrics.
    
    This is the primary entry point for recording retrieval behavior.
    It both logs the event (for external analysis) and updates in-memory
    metrics (for real-time monitoring).
    
    Args:
        workspace_id: Workspace the query was executed against
        query: The user's query (truncated in logs for privacy)
        confidence_level: "high", "low", or "insufficient"
        refused: Whether the query was refused
        doc_top_score: Document-level aggregated confidence score
        chunk_top_score: Best chunk confidence score
        raw_top_score: Raw FAISS similarity score
        doc_hit_count: Number of chunks from best matching document
        lexical_match: Whether lexical overlap check passed
        entity_fact_lookup: Whether query was classified as entity-fact lookup
        num_sources: Number of sources returned
    """
    # Determine if lexical was used as a fallback (saved a refusal)
    lexical_fallback = (
        lexical_match 
        and confidence_level.lower() in ("low", "insufficient")
        and not refused
    )
    
    # Update in-memory metrics
    record_retrieval_event(
        confidence_level=confidence_level,
        refused=refused,
        entity_fact_lookup=entity_fact_lookup,
        lexical_fallback=lexical_fallback,
        doc_hit_count=doc_hit_count or 0,
    )
    
    # Structured log for external analysis
    log_data = {
        "event_type": "retrieval_query",
        "timestamp": datetime.utcnow().isoformat(),
        "workspace_id": workspace_id,
        "query_preview": query[:100] + "..." if len(query) > 100 else query,
        "query_length": len(query),
        "confidence_level": confidence_level,
        "refused": refused,
        "num_sources": num_sources,
    }
    
    # Add optional score fields
    if doc_top_score is not None:
        log_data["doc_top_score"] = round(doc_top_score, 4)
    if chunk_top_score is not None:
        log_data["chunk_top_score"] = round(chunk_top_score, 4)
    if raw_top_score is not None:
        log_data["raw_top_score"] = round(raw_top_score, 4)
    if doc_hit_count is not None:
        log_data["doc_hit_count"] = doc_hit_count
    
    # Add flags
    log_data["lexical_match"] = lexical_match
    log_data["lexical_fallback"] = lexical_fallback
    log_data["entity_fact_lookup"] = entity_fact_lookup
    
    # Log at appropriate level
    if refused:
        logger.warning("RETRIEVAL_REFUSED", **log_data)
    else:
        logger.info("RETRIEVAL_SUCCESS", **log_data)


def get_metrics_summary() -> dict[str, Any]:
    """
    Get a summary of retrieval metrics suitable for API responses.
    
    Returns:
        Dictionary with metrics summary
    """
    return _metrics.to_dict()
