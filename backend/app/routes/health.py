"""Health check endpoints."""

import time
from datetime import datetime

import structlog
from fastapi import APIRouter

from app import __version__
from app.config import get_settings
from app.models.schemas import HealthResponse, HealthStatus, ComponentHealth

router = APIRouter()
logger = structlog.get_logger()


async def check_filesystem() -> ComponentHealth:
    """Check filesystem accessibility."""
    settings = get_settings()
    start = time.perf_counter()
    
    try:
        # Check if workspaces dir is accessible
        settings.workspaces_dir.mkdir(parents=True, exist_ok=True)
        test_file = settings.workspaces_dir / ".health_check"
        test_file.write_text("ok")
        test_file.unlink()
        
        latency = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            name="filesystem",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            name="filesystem",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 2),
            message=str(e),
        )


async def check_faiss() -> ComponentHealth:
    """Check FAISS availability."""
    start = time.perf_counter()
    
    try:
        import faiss
        import numpy as np
        
        # Quick sanity check - create tiny index
        d = 64
        index = faiss.IndexFlatL2(d)
        xb = np.random.random((10, d)).astype("float32")
        index.add(xb)
        
        latency = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            name="faiss",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            name="faiss",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 2),
            message=str(e),
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Comprehensive health check endpoint.
    
    Returns status of all system components including:
    - Filesystem access
    - FAISS vector store
    """
    settings = get_settings()
    
    # Run component checks
    components = [
        await check_filesystem(),
        await check_faiss(),
    ]
    
    # Determine overall status
    statuses = [c.status for c in components]
    if all(s == HealthStatus.HEALTHY for s in statuses):
        overall = HealthStatus.HEALTHY
    elif any(s == HealthStatus.UNHEALTHY for s in statuses):
        overall = HealthStatus.UNHEALTHY
    else:
        overall = HealthStatus.DEGRADED
    
    response = HealthResponse(
        status=overall,
        version=__version__,
        environment=settings.app_env,
        timestamp=datetime.utcnow(),
        components=components,
    )
    
    logger.info("Health check completed", status=overall.value)
    return response


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe - always returns ok if server is running."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness() -> dict[str, str]:
    """Kubernetes readiness probe - checks critical dependencies."""
    fs_check = await check_filesystem()
    
    if fs_check.status == HealthStatus.UNHEALTHY:
        return {"status": "not_ready", "reason": "filesystem unavailable"}
    
    return {"status": "ready"}
