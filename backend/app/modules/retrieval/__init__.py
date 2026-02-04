"""Document retrieval module."""

from app.modules.retrieval.faiss_index import (
    create_index,
    add_chunks_to_index,
    search_similar,
    load_index,
    save_index,
    rebuild_index,
)
from app.modules.retrieval.scoring import (
    compute_confidence,
    rerank_results,
    filter_by_threshold,
    assess_confidence,
    log_refusal_event,
    ConfidenceLevel,
    ConfidenceAssessment,
)

__all__ = [
    "create_index",
    "add_chunks_to_index",
    "search_similar",
    "load_index",
    "save_index",
    "rebuild_index",
    "compute_confidence",
    "rerank_results",
    "filter_by_threshold",
    "assess_confidence",
    "log_refusal_event",
    "ConfidenceLevel",
    "ConfidenceAssessment",
]
