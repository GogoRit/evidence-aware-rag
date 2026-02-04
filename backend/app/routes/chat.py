"""Chat and query endpoints."""

import time
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, status

from app.config import get_settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    SourceDocument,
    ConfidenceInfo,
    RouterDecisionInfo,
)
from app.modules.retrieval import (
    search_similar,
    compute_confidence,
    assess_confidence,
    log_refusal_event,
)
from app.modules.generation import generate_response
from app.modules.generation.router import make_routing_decision, RoutingMode
from app.modules.observability.latency import track_latency
from app.modules.observability.cost import estimate_cost

router = APIRouter()
logger = structlog.get_logger()


def build_sources(
    scored_results: list[tuple[dict, float]],
    max_content_length: int = 500,
) -> list[SourceDocument]:
    """Build source documents from scored results."""
    return [
        SourceDocument(
            document_id=r["document_id"],
            chunk_id=r["chunk_id"],
            content=r["content"][:max_content_length] + "..." if len(r["content"]) > max_content_length else r["content"],
            score=confidence,
            metadata={
                **r.get("metadata", {}),
                "raw_score": r["score"],
                "rank": idx + 1,
            },
        )
        for idx, (r, confidence) in enumerate(scored_results)
    ]


def build_router_decision_info(decision) -> RouterDecisionInfo:
    """Convert RouterDecision dataclass to Pydantic model."""
    return RouterDecisionInfo(
        mode=decision.mode.value,
        confidence=decision.confidence_score,
        complexity=decision.complexity_score,
        effective_complexity=decision.effective_complexity,
        chosen_model=decision.chosen_model,
        reason=decision.reason,
        cost_estimate_usd=decision.cost_estimate_usd,
        factors=decision.factors,
    )


@router.post("", response_model=ChatResponse)
@track_latency("chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Query the RAG system with a natural language question.
    
    Process:
    1. Retrieve relevant document chunks from the workspace
    2. Assess confidence - refuse if below threshold
    3. Make deterministic routing decision
    4. If generation mode: route to LLM and generate answer
    5. Return response with full router_decision details
    
    The router_decision field contains:
    - mode: generation/retrieval_only/refused/none
    - confidence: retrieval confidence score
    - complexity: query complexity score
    - effective_complexity: combined score used for model selection
    - chosen_model: selected model or mode name
    - reason: human-readable explanation
    - cost_estimate_usd: estimated cost
    """
    start_time = time.perf_counter()
    settings = get_settings()
    
    # Verify workspace exists
    workspace_path = settings.workspaces_dir / request.workspace_id
    if not workspace_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{request.workspace_id}' not found",
        )
    
    logger.info(
        "Processing chat request",
        workspace_id=request.workspace_id,
        query_length=len(request.query),
        top_k=request.top_k,
        generation_enabled=settings.generation_enabled,
    )
    
    # Retrieve relevant chunks
    retrieval_results = await search_similar(
        workspace_path=workspace_path,
        query=request.query,
        top_k=request.top_k,
    )
    
    # Handle no results
    if not retrieval_results:
        logger.warning("No relevant documents found", workspace_id=request.workspace_id)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        # Make routing decision for no-results case
        routing_decision = make_routing_decision(
            query=request.query,
            confidence_score=0.0,
            generation_enabled=settings.generation_enabled,
            is_refused=True,
            model_preference=request.model_preference,
        )
        
        # Log refusal event
        from app.modules.retrieval.scoring import ConfidenceAssessment, ConfidenceLevel
        empty_assessment = ConfidenceAssessment(
            level=ConfidenceLevel.INSUFFICIENT,
            top_score=0.0,
            mean_score=0.0,
            should_refuse=True,
            should_warn=False,
            reason="No documents found in workspace",
        )
        log_refusal_event(request.workspace_id, request.query, empty_assessment)
        
        return ChatResponse(
            answer=None,
            workspace_id=request.workspace_id,
            sources=[],
            model_used="none",
            latency_ms=latency_ms,
            retrieval_only=not settings.generation_enabled,
            refused=True,
            refusal_reason="No relevant documents found in the workspace.",
            confidence=ConfidenceInfo(
                level="insufficient",
                top_score=0.0,
                mean_score=0.0,
                threshold=settings.confidence_threshold,
            ),
            router_decision=build_router_decision_info(routing_decision),
        )
    
    # Compute confidence scores for each result
    scored_results = [
        (result, compute_confidence(result["score"]))
        for result in retrieval_results
    ]
    
    # Assess overall confidence (mode-aware)
    confidence_scores = [conf for _, conf in scored_results]
    assessment = assess_confidence(
        scores=confidence_scores,
        confidence_threshold=settings.confidence_threshold,
        refusal_threshold=settings.refusal_threshold,
        generation_enabled=settings.generation_enabled,
        query=request.query,
        top_chunks=retrieval_results[:5],  # Pass top chunks for lexical check
    )
    
    # Build confidence info for response
    confidence_info = ConfidenceInfo(
        level=assessment.level.value,
        top_score=round(assessment.top_score, 4),
        mean_score=round(assessment.mean_score, 4),
        threshold=settings.confidence_threshold,
    )
    
    # Prepare sources (always include for transparency)
    sources = build_sources(scored_results)
    
    # Make routing decision (deterministic)
    routing_decision = make_routing_decision(
        query=request.query,
        confidence_score=assessment.top_score,
        generation_enabled=settings.generation_enabled,
        is_refused=assessment.should_refuse,
        model_preference=request.model_preference,
    )
    
    router_decision_info = build_router_decision_info(routing_decision)
    
    # Check if we should refuse
    if assessment.should_refuse:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        # Log the refusal event
        log_refusal_event(
            workspace_id=request.workspace_id,
            query=request.query,
            assessment=assessment,
            top_chunks=retrieval_results[:3],
        )
        
        # Determine refusal message based on confidence level
        if assessment.level.value == "insufficient":
            refusal_reason = f"{settings.refusal_message} The retrieved information has very low relevance (confidence: {assessment.top_score:.1%})."
        else:
            refusal_reason = f"{settings.refusal_message} However, here are the closest matches found (confidence: {assessment.top_score:.1%})."
        
        logger.info(
            "Request refused due to low confidence",
            workspace_id=request.workspace_id,
            confidence_level=assessment.level.value,
            top_score=assessment.top_score,
            threshold=settings.confidence_threshold,
        )
        
        return ChatResponse(
            answer=None,
            workspace_id=request.workspace_id,
            sources=sources,
            model_used="refused",
            latency_ms=latency_ms,
            cost_usd=0.0,
            tokens_used={},
            retrieval_only=not settings.generation_enabled,
            refused=True,
            refusal_reason=refusal_reason,
            confidence=confidence_info,
            router_decision=router_decision_info,
        )
    
    # Confidence is sufficient - proceed with response
    
    # Retrieval-only mode: return results without LLM generation
    if not settings.generation_enabled:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        logger.info(
            "Retrieval-only response",
            workspace_id=request.workspace_id,
            chunks_retrieved=len(sources),
            top_score=assessment.top_score,
            complexity=routing_decision.complexity_score,
            latency_ms=latency_ms,
        )
        
        return ChatResponse(
            answer=None,
            workspace_id=request.workspace_id,
            sources=sources,
            model_used="retrieval-only",
            latency_ms=latency_ms,
            cost_usd=0.0,
            tokens_used={},
            retrieval_only=True,
            refused=False,
            confidence=confidence_info,
            router_decision=router_decision_info,
        )
    
    # Generation mode: use the routed model
    model_to_use = routing_decision.chosen_model
    
    # Build context from retrieved chunks
    context = "\n\n---\n\n".join([
        f"[Source: {r['metadata'].get('filename', 'Unknown')}]\n{r['content']}"
        for r, _ in scored_results
    ])
    
    # Generate response
    generation_result = await generate_response(
        query=request.query,
        context=context,
        model=model_to_use,
    )
    
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    
    # Calculate actual cost
    actual_cost = estimate_cost(
        model=model_to_use,
        input_tokens=generation_result.get("input_tokens", 0),
        output_tokens=generation_result.get("output_tokens", 0),
    )
    
    logger.info(
        "Chat request completed",
        workspace_id=request.workspace_id,
        model=model_to_use,
        confidence_level=assessment.level.value,
        complexity=routing_decision.complexity_score,
        effective_complexity=routing_decision.effective_complexity,
        latency_ms=latency_ms,
        cost_estimate_usd=routing_decision.cost_estimate_usd,
        cost_actual_usd=actual_cost,
    )
    
    return ChatResponse(
        answer=generation_result["answer"],
        workspace_id=request.workspace_id,
        sources=sources,
        model_used=model_to_use,
        latency_ms=latency_ms,
        cost_usd=actual_cost,
        tokens_used={
            "input": generation_result.get("input_tokens", 0),
            "output": generation_result.get("output_tokens", 0),
        },
        retrieval_only=False,
        refused=False,
        confidence=confidence_info,
        router_decision=router_decision_info,
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> Any:
    """
    Stream chat responses for real-time UI updates.
    
    Note: Streaming implementation placeholder - requires SSE setup.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Streaming not yet implemented. Use POST /chat for synchronous responses.",
    )
