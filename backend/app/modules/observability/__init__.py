"""Observability module for monitoring and metrics."""

import logging

import structlog

# Map string log levels to Python logging constants
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging."""
    # Import processors inside function to avoid import-time issues
    from structlog.processors import TimeStamper
    
    level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
    
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer() if log_level != "DEBUG" else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


from app.modules.observability.latency import (
    track_latency,
    get_latency_stats,
    reset_latency_stats,
)
from app.modules.observability.cost import (
    track_cost,
    estimate_cost,
    get_cost_summary,
    reset_cost_tracking,
)
from app.modules.observability.workspace_stats import (
    get_workspace_aggregate_stats,
    count_lines_in_file,
    count_documents_in_manifest,
)
from app.modules.observability.retrieval_metrics import (
    RetrievalMetrics,
    get_retrieval_metrics,
    reset_retrieval_metrics,
    record_retrieval_event,
    log_retrieval_event,
    get_metrics_summary,
)

__all__ = [
    "setup_logging",
    "track_latency",
    "get_latency_stats",
    "reset_latency_stats",
    "track_cost",
    "estimate_cost",
    "get_cost_summary",
    "reset_cost_tracking",
    "get_workspace_aggregate_stats",
    "count_lines_in_file",
    "count_documents_in_manifest",
    # Retrieval metrics
    "RetrievalMetrics",
    "get_retrieval_metrics",
    "reset_retrieval_metrics",
    "record_retrieval_event",
    "log_retrieval_event",
    "get_metrics_summary",
]
