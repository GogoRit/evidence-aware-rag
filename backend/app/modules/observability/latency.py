"""Latency tracking and metrics."""

import time
from collections import defaultdict
from datetime import datetime
from functools import wraps
from typing import Any, Callable

import structlog

logger = structlog.get_logger()

# In-memory metrics store (use Redis/Prometheus in production)
_latency_samples: dict[str, list[float]] = defaultdict(list)
_request_counts: dict[str, int] = defaultdict(int)
_start_time = datetime.utcnow()


def track_latency(endpoint: str) -> Callable:
    """
    Decorator to track endpoint latency.
    
    Usage:
        @track_latency("ingest")
        async def ingest_document(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                latency_ms = (time.perf_counter() - start) * 1000
                record_latency(endpoint, latency_ms)
        
        return wrapper
    return decorator


def record_latency(endpoint: str, latency_ms: float) -> None:
    """Record a latency measurement."""
    _latency_samples[endpoint].append(latency_ms)
    _request_counts[endpoint] += 1
    
    # Keep only last 10000 samples per endpoint
    if len(_latency_samples[endpoint]) > 10000:
        _latency_samples[endpoint] = _latency_samples[endpoint][-10000:]
    
    logger.debug(
        "Latency recorded",
        endpoint=endpoint,
        latency_ms=round(latency_ms, 2),
    )


def get_percentile(samples: list[float], percentile: float) -> float:
    """Calculate percentile from samples."""
    if not samples:
        return 0.0
    
    sorted_samples = sorted(samples)
    index = int(len(sorted_samples) * percentile / 100)
    return sorted_samples[min(index, len(sorted_samples) - 1)]


def get_latency_stats() -> dict[str, Any]:
    """
    Get latency statistics.
    
    Returns:
        Dictionary with p50, p90, p99, avg, and per-endpoint stats
    """
    all_samples = []
    for samples in _latency_samples.values():
        all_samples.extend(samples)
    
    if not all_samples:
        return {
            "count": 0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "avg": 0.0,
            "by_endpoint": {},
        }
    
    by_endpoint = {}
    for endpoint, samples in _latency_samples.items():
        if samples:
            by_endpoint[endpoint] = {
                "count": _request_counts[endpoint],
                "p50": round(get_percentile(samples, 50), 2),
                "p90": round(get_percentile(samples, 90), 2),
                "p99": round(get_percentile(samples, 99), 2),
                "avg": round(sum(samples) / len(samples), 2),
            }
    
    return {
        "count": sum(_request_counts.values()),
        "p50": round(get_percentile(all_samples, 50), 2),
        "p90": round(get_percentile(all_samples, 90), 2),
        "p99": round(get_percentile(all_samples, 99), 2),
        "avg": round(sum(all_samples) / len(all_samples), 2),
        "by_endpoint": by_endpoint,
        "since": _start_time.isoformat(),
    }


def reset_latency_stats() -> None:
    """Reset all latency statistics."""
    global _start_time
    _latency_samples.clear()
    _request_counts.clear()
    _start_time = datetime.utcnow()
    logger.info("Latency stats reset")


class LatencyTracker:
    """Context manager for tracking latency of code blocks."""
    
    def __init__(self, operation: str):
        self.operation = operation
        self.start_time: float = 0
        self.elapsed_ms: float = 0
    
    def __enter__(self) -> "LatencyTracker":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        record_latency(self.operation, self.elapsed_ms)
    
    async def __aenter__(self) -> "LatencyTracker":
        return self.__enter__()
    
    async def __aexit__(self, *args: Any) -> None:
        self.__exit__()
