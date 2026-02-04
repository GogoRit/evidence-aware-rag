"""Response generation module."""

from app.modules.generation.router import (
    route_query,
    make_routing_decision,
    estimate_query_complexity,
    RouterDecision,
    RoutingMode,
    get_available_models,
)
from app.modules.generation.cheap_backend import generate_cheap
from app.modules.generation.expensive_backend import generate_expensive

__all__ = [
    "route_query",
    "make_routing_decision",
    "estimate_query_complexity",
    "RouterDecision",
    "RoutingMode",
    "get_available_models",
    "generate_cheap",
    "generate_expensive",
]


async def generate_response(
    query: str,
    context: str,
    model: str,
) -> dict:
    """
    Generate a response using the specified model.
    
    Args:
        query: User question
        context: Retrieved document context
        model: Model identifier
    
    Returns:
        Dictionary with answer and token usage
    """
    from app.config import get_settings
    
    settings = get_settings()
    
    # Route to appropriate backend based on model
    if model in [settings.cheap_model, "gpt-3.5-turbo", "claude-3-haiku"]:
        return await generate_cheap(query, context, model)
    else:
        return await generate_expensive(query, context, model)
