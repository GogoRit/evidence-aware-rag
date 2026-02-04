"""Cost tracking for LLM usage."""

from collections import defaultdict
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger()

# Model pricing (per 1000 tokens)
MODEL_PRICING = {
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-32k": {"input": 0.06, "output": 0.12},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
}

# In-memory cost tracking (use database in production)
_cost_by_model: dict[str, float] = defaultdict(float)
_token_usage: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})
_period_start = datetime.utcnow()


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Estimate cost for a model invocation.
    
    Args:
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    
    Returns:
        Estimated cost in USD
    """
    pricing = MODEL_PRICING.get(model)
    
    if not pricing:
        # Try to match partial model name
        for known_model, price in MODEL_PRICING.items():
            if known_model in model or model in known_model:
                pricing = price
                break
    
    if not pricing:
        logger.warning("Unknown model for cost estimation", model=model)
        return 0.0
    
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    
    return round(input_cost + output_cost, 6)


def track_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Track cost for a model invocation.
    
    Args:
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    
    Returns:
        Cost in USD
    """
    cost = estimate_cost(model, input_tokens, output_tokens)
    
    _cost_by_model[model] += cost
    _token_usage[model]["input"] += input_tokens
    _token_usage[model]["output"] += output_tokens
    
    logger.debug(
        "Cost tracked",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )
    
    return cost


def get_cost_summary() -> dict[str, Any]:
    """
    Get cost summary for the current period.
    
    Returns:
        Dictionary with total cost, breakdown by model, and token usage
    """
    total = sum(_cost_by_model.values())
    
    return {
        "total": round(total, 6),
        "by_model": {
            model: round(cost, 6)
            for model, cost in _cost_by_model.items()
        },
        "token_usage": dict(_token_usage),
        "period_start": _period_start,
    }


def reset_cost_tracking() -> None:
    """Reset cost tracking for a new period."""
    global _period_start
    _cost_by_model.clear()
    _token_usage.clear()
    _period_start = datetime.utcnow()
    logger.info("Cost tracking reset")


def get_budget_status(budget_usd: float) -> dict[str, Any]:
    """
    Check current spend against budget.
    
    Args:
        budget_usd: Budget limit in USD
    
    Returns:
        Budget status including remaining amount and percentage used
    """
    total_spent = sum(_cost_by_model.values())
    remaining = max(0, budget_usd - total_spent)
    percentage_used = (total_spent / budget_usd * 100) if budget_usd > 0 else 0
    
    status = "ok"
    if percentage_used >= 100:
        status = "exceeded"
    elif percentage_used >= 90:
        status = "critical"
    elif percentage_used >= 75:
        status = "warning"
    
    return {
        "budget_usd": budget_usd,
        "spent_usd": round(total_spent, 6),
        "remaining_usd": round(remaining, 6),
        "percentage_used": round(percentage_used, 2),
        "status": status,
    }


def project_monthly_cost() -> float:
    """
    Project monthly cost based on current usage.
    
    Returns:
        Projected monthly cost in USD
    """
    elapsed = (datetime.utcnow() - _period_start).total_seconds()
    
    if elapsed < 3600:  # Less than an hour of data
        return 0.0
    
    total_spent = sum(_cost_by_model.values())
    seconds_in_month = 30 * 24 * 3600
    
    projected = (total_spent / elapsed) * seconds_in_month
    return round(projected, 2)
