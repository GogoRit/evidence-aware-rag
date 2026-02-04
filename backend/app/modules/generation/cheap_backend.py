"""Cheap/fast model backend for simple queries."""

import structlog

from app.config import get_settings

logger = structlog.get_logger()


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Only use information from the provided context to answer
2. If the context doesn't contain enough information, say so clearly
3. Be concise and direct in your answers
4. Cite specific parts of the context when relevant

Context:
{context}
"""


async def generate_cheap(
    query: str,
    context: str,
    model: str | None = None,
) -> dict:
    """
    Generate response using cheap/fast model.
    
    Optimized for:
    - Simple factual queries
    - Good quality retrieved context
    - Cost-sensitive applications
    
    Args:
        query: User question
        context: Retrieved document context
        model: Specific model to use (optional)
    
    Returns:
        Dictionary with answer and token usage
    """
    settings = get_settings()
    model = model or settings.cheap_model
    
    logger.info("Generating with cheap backend", model=model)
    
    # Try OpenAI first
    if settings.openai_api_key and "gpt" in model:
        return await _generate_openai(query, context, model)
    
    # Try Anthropic
    if settings.anthropic_api_key and "claude" in model:
        return await _generate_anthropic(query, context, model)
    
    # Fallback response
    logger.warning("No API keys configured, using fallback response")
    return _fallback_response(query, context)


async def _generate_openai(query: str, context: str, model: str) -> dict:
    """Generate using OpenAI API."""
    from app.config import get_settings
    
    settings = get_settings()
    
    try:
        import openai
        
        client = openai.OpenAI(api_key=settings.openai_api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        return {
            "answer": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "model": model,
        }
    except Exception as e:
        logger.error("OpenAI generation failed", error=str(e))
        return _fallback_response(query, context)


async def _generate_anthropic(query: str, context: str, model: str) -> dict:
    """Generate using Anthropic API."""
    from app.config import get_settings
    
    settings = get_settings()
    
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            system=SYSTEM_PROMPT.format(context=context),
            messages=[
                {"role": "user", "content": query},
            ],
        )
        
        return {
            "answer": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": model,
        }
    except Exception as e:
        logger.error("Anthropic generation failed", error=str(e))
        return _fallback_response(query, context)


def _fallback_response(query: str, context: str) -> dict:
    """Fallback response when no API is available."""
    # Extract first relevant sentence from context as basic answer
    context_sentences = context.split('.')[:3]
    basic_answer = '. '.join(context_sentences) + '.'
    
    return {
        "answer": f"Based on the available context: {basic_answer}\n\n(Note: Full AI generation unavailable - API key not configured)",
        "input_tokens": len(context.split()),
        "output_tokens": len(basic_answer.split()),
        "model": "fallback",
    }
