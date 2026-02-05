"""Metrics and observability endpoints."""

from datetime import datetime

import structlog
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.config import get_settings
from app.models.schemas import MetricsResponse, LatencyMetrics, CostMetrics
from app.modules.observability.latency import get_latency_stats
from app.modules.observability.cost import get_cost_summary
from app.modules.observability.workspace_stats import get_workspace_aggregate_stats
from app.modules.observability.retrieval_metrics import (
    get_metrics_summary,
    reset_retrieval_metrics,
)

router = APIRouter()
logger = structlog.get_logger()


@router.get("", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """
    Get system metrics including:
    - Request counts by endpoint
    - Latency percentiles
    - Cost tracking
    - Index statistics
    """
    settings = get_settings()
    
    # Get latency stats
    latency_stats = get_latency_stats()
    latency = None
    if latency_stats["count"] > 0:
        latency = LatencyMetrics(
            p50_ms=latency_stats["p50"],
            p90_ms=latency_stats["p90"],
            p99_ms=latency_stats["p99"],
            avg_ms=latency_stats["avg"],
        )
    
    # Get cost summary
    cost_data = get_cost_summary()
    costs = None
    if cost_data["total"] > 0:
        costs = CostMetrics(
            total_usd=cost_data["total"],
            by_model=cost_data["by_model"],
            period_start=cost_data["period_start"],
            period_end=datetime.utcnow(),
        )
    
    # Count workspaces, documents, and chunks using correct storage contract
    workspace_stats = get_workspace_aggregate_stats(settings.workspaces_dir)
    active_workspaces = workspace_stats["active_workspaces"]
    documents_ingested = workspace_stats["documents_ingested"]
    chunks_indexed = workspace_stats["chunks_indexed"]
    
    # Extract just the counts from by_endpoint (schema expects dict[str, int])
    by_endpoint_raw = latency_stats.get("by_endpoint", {})
    requests_by_endpoint = {
        endpoint: data["count"] if isinstance(data, dict) else data
        for endpoint, data in by_endpoint_raw.items()
    }
    
    return MetricsResponse(
        requests_total=latency_stats["count"],
        requests_by_endpoint=requests_by_endpoint,
        latency=latency,
        costs=costs,
        active_workspaces=active_workspaces,
        documents_ingested=documents_ingested,
        chunks_indexed=chunks_indexed,
    )


@router.get("/prometheus", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    """
    Prometheus-compatible metrics endpoint.
    
    Exposes metrics in Prometheus text format for scraping.
    """
    try:
        from prometheus_client import generate_latest, REGISTRY
        return generate_latest(REGISTRY).decode("utf-8")
    except ImportError:
        return "# prometheus_client not available\n"


@router.get("/health/detailed")
async def detailed_health() -> dict:
    """
    Detailed health metrics for monitoring dashboards.
    """
    settings = get_settings()
    latency_stats = get_latency_stats()
    
    return {
        "status": "healthy",
        "uptime_checks": {
            "filesystem": settings.workspaces_dir.exists(),
            "config_loaded": True,
        },
        "performance": {
            "avg_latency_ms": latency_stats["avg"],
            "requests_last_hour": latency_stats["count"],
        },
        "configuration": {
            "environment": settings.app_env,
            "debug": settings.debug,
            "chunk_size": settings.chunk_size,
            "embedding_dim": settings.embedding_dim,
        },
    }


@router.get("/retrieval")
async def get_retrieval_metrics() -> dict:
    """
    Get retrieval-specific metrics including:
    - Refusal rate and confidence distribution
    - Entity-fact lookup rate
    - Lexical fallback rate
    - Document aggregation statistics
    """
    return get_metrics_summary()


@router.post("/retrieval/reset")
async def reset_retrieval_metrics_endpoint() -> dict:
    """
    Reset retrieval metrics counters.
    
    Safe for development use. In production, consider gating behind
    an environment variable or admin authentication.
    """
    reset_retrieval_metrics()
    logger.warning("Retrieval metrics reset by user")
    return {"status": "reset", "timestamp": datetime.utcnow().isoformat()}


@router.delete("/reset")
async def reset_metrics() -> dict:
    """Reset all collected metrics. Use with caution."""
    from app.modules.observability.latency import reset_latency_stats
    from app.modules.observability.cost import reset_cost_tracking
    
    reset_latency_stats()
    reset_cost_tracking()
    
    logger.warning("Metrics reset by user")
    return {"status": "reset", "timestamp": datetime.utcnow().isoformat()}
