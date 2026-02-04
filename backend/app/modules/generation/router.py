"""Deterministic query routing to appropriate models based on complexity."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import structlog

from app.config import get_settings

logger = structlog.get_logger()


class RoutingMode(str, Enum):
    """Routing mode for the request."""
    GENERATION = "generation"
    RETRIEVAL_ONLY = "retrieval_only"
    REFUSED = "refused"
    NONE = "none"  # No results


@dataclass
class RouterDecision:
    """Complete routing decision with all factors."""
    mode: RoutingMode
    chosen_model: str
    confidence_score: float
    complexity_score: float
    effective_complexity: float  # Combined score used for decision
    reason: str
    cost_estimate_usd: float
    factors: dict  # Detailed breakdown of decision factors


def estimate_query_complexity(query: str) -> tuple[float, dict]:
    """
    Estimate query complexity on a scale of 0-1.
    
    Returns:
        Tuple of (complexity_score, factors_breakdown)
    """
    factors = {
        "length": 0.0,
        "complex_keywords": 0.0,
        "multi_part": 0.0,
        "enumeration": 0.0,
        "technical": 0.0,
    }
    
    words = query.split()
    word_count = len(words)
    
    # Length factor (longer queries tend to be more complex)
    if word_count > 50:
        factors["length"] = 0.3
    elif word_count > 20:
        factors["length"] = 0.15
    elif word_count > 10:
        factors["length"] = 0.05
    
    # Complex keywords - deterministic matching
    complex_patterns = [
        (r'\bcompare\b', 0.15),
        (r'\bcontrast\b', 0.15),
        (r'\banalyze\b', 0.12),
        (r'\bevaluate\b', 0.12),
        (r'\bsynthesize\b', 0.15),
        (r'\bcritique\b', 0.12),
        (r'\bexplain\s+why\b', 0.10),
        (r'\bhow\s+does\s+.+\s+relate\s+to\b', 0.15),
        (r'\bwhat\s+are\s+the\s+implications\b', 0.12),
        (r'\bpros\s+and\s+cons\b', 0.15),
        (r'\btrade-?offs?\b', 0.12),
        (r'\bdifferences?\s+between\b', 0.10),
        (r'\badvantages?\s+and\s+disadvantages?\b', 0.12),
    ]
    
    query_lower = query.lower()
    keyword_score = 0.0
    matched_patterns = []
    for pattern, weight in complex_patterns:
        if re.search(pattern, query_lower):
            keyword_score += weight
            matched_patterns.append(pattern)
    factors["complex_keywords"] = min(0.4, keyword_score)  # Cap at 0.4
    
    # Multi-part questions
    question_marks = query.count('?')
    if question_marks > 2:
        factors["multi_part"] = 0.25
    elif question_marks > 1:
        factors["multi_part"] = 0.15
    
    # Numbered lists or enumeration
    if re.search(r'\b(first|second|third|fourth|fifth)\b', query_lower):
        factors["enumeration"] = 0.1
    if re.search(r'\b[1-5]\.\s', query):
        factors["enumeration"] = max(factors["enumeration"], 0.15)
    
    # Technical terminology density
    technical_words = [
        'algorithm', 'architecture', 'implementation', 'optimization',
        'framework', 'methodology', 'paradigm', 'infrastructure',
        'scalability', 'latency', 'throughput', 'concurrent',
    ]
    tech_count = sum(1 for word in technical_words if word in query_lower)
    factors["technical"] = min(0.2, tech_count * 0.05)
    
    # Sum all factors and cap at 1.0
    total = sum(factors.values())
    return min(1.0, total), factors


def estimate_cost(
    model: str,
    estimated_input_tokens: int = 2000,
    estimated_output_tokens: int = 500,
) -> float:
    """
    Estimate cost for a model invocation.
    
    Uses conservative estimates for token counts.
    """
    # Model pricing per 1000 tokens
    pricing = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    }
    
    model_pricing = pricing.get(model)
    if not model_pricing:
        # Try partial match
        for known_model, price in pricing.items():
            if known_model in model or model in known_model:
                model_pricing = price
                break
    
    if not model_pricing:
        return 0.0
    
    input_cost = (estimated_input_tokens / 1000) * model_pricing["input"]
    output_cost = (estimated_output_tokens / 1000) * model_pricing["output"]
    
    return round(input_cost + output_cost, 6)


def compute_effective_complexity(
    complexity: float,
    confidence: float,
    complexity_weight: float = 0.7,
) -> float:
    """
    Compute effective complexity combining query complexity and context quality.
    
    High complexity OR low confidence → higher effective complexity → expensive model
    
    Formula: effective = complexity * weight + (1 - confidence) * (1 - weight)
    """
    context_factor = 1.0 - confidence  # Low confidence = high factor
    effective = complexity * complexity_weight + context_factor * (1.0 - complexity_weight)
    return round(effective, 4)


def make_routing_decision(
    query: str,
    confidence_score: float,
    generation_enabled: bool,
    is_refused: bool,
    model_preference: str | None = None,
) -> RouterDecision:
    """
    Make a deterministic routing decision.
    
    Args:
        query: User query
        confidence_score: Top confidence score from retrieval
        generation_enabled: Whether LLM generation is enabled
        is_refused: Whether the request was refused due to low confidence
        model_preference: Optional user override for model selection
    
    Returns:
        RouterDecision with all routing details
    """
    settings = get_settings()
    
    # Determine mode
    if is_refused:
        mode = RoutingMode.REFUSED
    elif not generation_enabled:
        mode = RoutingMode.RETRIEVAL_ONLY
    else:
        mode = RoutingMode.GENERATION
    
    # Calculate complexity
    complexity_score, complexity_factors = estimate_query_complexity(query)
    
    # Calculate effective complexity
    effective_complexity = compute_effective_complexity(
        complexity_score,
        confidence_score,
    )
    
    # Determine model
    if mode != RoutingMode.GENERATION:
        chosen_model = mode.value  # "refused" or "retrieval_only"
        reason = f"Mode is {mode.value}, no model selection needed"
        cost_estimate = 0.0
    elif model_preference:
        chosen_model = model_preference
        reason = f"User specified model preference: {model_preference}"
        cost_estimate = estimate_cost(model_preference)
    elif effective_complexity > settings.complexity_threshold:
        chosen_model = settings.expensive_model
        reason = (
            f"Effective complexity {effective_complexity:.2f} > threshold {settings.complexity_threshold:.2f}. "
            f"Query complexity: {complexity_score:.2f}, Context confidence: {confidence_score:.2f}"
        )
        cost_estimate = estimate_cost(settings.expensive_model)
    else:
        chosen_model = settings.cheap_model
        reason = (
            f"Effective complexity {effective_complexity:.2f} <= threshold {settings.complexity_threshold:.2f}. "
            f"Query complexity: {complexity_score:.2f}, Context confidence: {confidence_score:.2f}"
        )
        cost_estimate = estimate_cost(settings.cheap_model)
    
    decision = RouterDecision(
        mode=mode,
        chosen_model=chosen_model,
        confidence_score=round(confidence_score, 4),
        complexity_score=round(complexity_score, 4),
        effective_complexity=effective_complexity,
        reason=reason,
        cost_estimate_usd=cost_estimate,
        factors={
            "complexity_breakdown": complexity_factors,
            "complexity_threshold": settings.complexity_threshold,
            "confidence_threshold": settings.confidence_threshold,
            "generation_enabled": generation_enabled,
            "model_preference_used": model_preference is not None,
        },
    )
    
    # Log the routing decision
    logger.info(
        "ROUTING_DECISION",
        mode=decision.mode.value,
        chosen_model=decision.chosen_model,
        confidence_score=decision.confidence_score,
        complexity_score=decision.complexity_score,
        effective_complexity=decision.effective_complexity,
        cost_estimate_usd=decision.cost_estimate_usd,
        reason=decision.reason,
    )
    
    return decision


# Legacy function for backward compatibility
async def route_query(
    query: str,
    context_quality: float = 0.5,
) -> str:
    """
    Route a query to the appropriate model.
    
    Deprecated: Use make_routing_decision() for full decision details.
    """
    decision = make_routing_decision(
        query=query,
        confidence_score=context_quality,
        generation_enabled=True,
        is_refused=False,
    )
    return decision.chosen_model


def get_available_models() -> dict[str, dict]:
    """Get information about available models."""
    return {
        "gpt-3.5-turbo": {
            "provider": "openai",
            "tier": "cheap",
            "context_window": 16385,
            "cost_per_1k_input": 0.0005,
            "cost_per_1k_output": 0.0015,
        },
        "gpt-4-turbo-preview": {
            "provider": "openai",
            "tier": "expensive",
            "context_window": 128000,
            "cost_per_1k_input": 0.01,
            "cost_per_1k_output": 0.03,
        },
        "claude-3-haiku-20240307": {
            "provider": "anthropic",
            "tier": "cheap",
            "context_window": 200000,
            "cost_per_1k_input": 0.00025,
            "cost_per_1k_output": 0.00125,
        },
        "claude-3-opus-20240229": {
            "provider": "anthropic",
            "tier": "expensive",
            "context_window": 200000,
            "cost_per_1k_input": 0.015,
            "cost_per_1k_output": 0.075,
        },
    }
