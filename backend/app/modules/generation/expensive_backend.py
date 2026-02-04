"""Expensive/powerful model backend for complex queries."""

import structlog

from app.config import get_settings

logger = structlog.get_logger()


SYSTEM_PROMPT = """You are an expert assistant that provides thorough, well-reasoned answers based on the provided context.

Your approach:
1. Carefully analyze the provided context for relevant information
2. Synthesize information from multiple sources when applicable
3. Provide comprehensive answers with clear reasoning
4. Acknowledge limitations or gaps in the available information
5. Structure complex answers with clear organization

When the context is insufficient:
- Clearly state what information is missing
- Explain what would be needed to fully answer the question
- Provide what partial answer is possible from available context

Context:
{context}
"""


async def generate_expensive(
    query: str,
    context: str,
    model: str | None = None,
) -> dict:
    """
    Generate response using expensive/powerful model.
    
    Optimized for:
    - Complex multi-part queries
    - Queries requiring synthesis and reasoning
    - Cases where context quality is lower
    
    Args:
        query: User question
        context: Retrieved document context
        model: Specific model to use (optional)
    
    Returns:
        Dictionary with answer and token usage
    """
    settings = get_settings()
    model = model or settings.expensive_model
    
    logger.info("Generating with expensive backend", model=model)
    
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
    """Generate using OpenAI API with extended capabilities."""
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
            temperature=0.5,  # Slightly higher for more nuanced responses
            max_tokens=2000,  # More tokens for complex answers
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
    """Generate using Anthropic API with extended capabilities."""
    from app.config import get_settings
    
    settings = get_settings()
    
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        
        response = client.messages.create(
            model=model,
            max_tokens=2000,
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
    # Provide a more detailed fallback for complex queries
    context_preview = context[:1000] + "..." if len(context) > 1000 else context
    
    answer = f"""I apologize, but I cannot provide a full AI-generated response as no LLM API keys are configured.

However, here is the relevant context that was retrieved for your question:

---
{context_preview}
---

To enable full AI-powered responses, please configure either:
- OPENAI_API_KEY for GPT models
- ANTHROPIC_API_KEY for Claude models

Your question was: "{query}"
"""
    
    return {
        "answer": answer,
        "input_tokens": len(context.split()),
        "output_tokens": len(answer.split()),
        "model": "fallback",
    }
