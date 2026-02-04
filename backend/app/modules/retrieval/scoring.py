"""Confidence scoring, refusal logic, and result ranking."""

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()

# Common English stopwords for lexical overlap check
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "and", "but", "if", "or", "because", "until", "while", "about",
    "against", "this", "that", "these", "those", "what", "which", "who",
    "whom", "its", "it", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "their",
})

# Organization-related terms that shouldn't count as meaningful keywords
# These often appear in headers/footers but don't indicate topical relevance
ORG_TERMS = frozenset({
    "company", "corporation", "corp", "inc", "llc", "ltd", "organization",
    "enterprise", "business", "firm", "group", "acme",  # Add known org names
})


class ConfidenceLevel(str, Enum):
    """Confidence level classification."""
    HIGH = "high"        # >= confidence_threshold
    LOW = "low"          # >= refusal_threshold but < confidence_threshold  
    INSUFFICIENT = "insufficient"  # < refusal_threshold


@dataclass
class ConfidenceAssessment:
    """Result of confidence assessment."""
    level: ConfidenceLevel
    top_score: float
    mean_score: float
    should_refuse: bool
    should_warn: bool
    reason: str | None = None
    lexical_match: bool = False  # True if lexical sanity check passed


def extract_query_keywords(query: str) -> set[str]:
    """
    Extract meaningful keywords from a query (lowercase, no stopwords, no org terms).
    """
    words = set(query.lower().split())
    # Remove stopwords, org terms, and short words
    keywords = {
        w.strip("?.,!\"'") 
        for w in words 
        if w not in STOPWORDS 
        and w.strip("?.,!\"'") not in ORG_TERMS
        and len(w) > 2
    }
    return keywords


def has_lexical_overlap(
    query: str,
    chunks: list[dict[str, Any]],
    min_matches: int = 1,
    top_n: int = 3,
) -> bool:
    """
    Check if any of the top N chunks contain keywords from the query.
    
    This is a lightweight sanity check to prevent refusal when evidence
    is clearly present but vector similarity is modest.
    
    Args:
        query: The user query
        chunks: Retrieved chunks with 'content' field
        min_matches: Minimum keyword matches required
        top_n: Number of top chunks to check
        
    Returns:
        True if lexical overlap found, False otherwise
    """
    if not query or not chunks:
        return False
    
    query_keywords = extract_query_keywords(query)
    if not query_keywords:
        return False
    
    for chunk in chunks[:top_n]:
        content = chunk.get("content", "").lower()
        matches = sum(1 for kw in query_keywords if kw in content)
        if matches >= min_matches:
            logger.debug(
                "Lexical overlap found",
                query_keywords=list(query_keywords),
                matches=matches,
                chunk_id=chunk.get("chunk_id", "")[:8],
            )
            return True
    
    return False


# Threshold configurations for different modes
RETRIEVAL_ONLY_THRESHOLDS = {
    "high": 0.45,      # top_score >= 0.45 -> HIGH confidence
    "low": 0.20,       # top_score >= 0.20 -> LOW confidence
    "refuse": 0.10,    # top_score < 0.10 -> REFUSE (very strict)
}

GENERATION_THRESHOLDS = {
    "high": 0.40,      # top_score >= 0.40 -> HIGH confidence
    "low": 0.25,       # top_score >= 0.25 -> LOW confidence  
    "refuse": 0.25,    # top_score < 0.25 -> REFUSE (same as low boundary)
}


def assess_confidence(
    scores: list[float],
    confidence_threshold: float = 0.4,
    refusal_threshold: float = 0.25,
    generation_enabled: bool = True,
    query: str | None = None,
    top_chunks: list[dict[str, Any]] | None = None,
) -> ConfidenceAssessment:
    """
    Assess overall confidence from retrieval scores.
    
    Thresholds are mode-aware:
    - Retrieval-only mode: Relaxed thresholds (refuse only < 0.10)
    - Generation mode: Stricter thresholds (refuse < 0.25)
    
    Additionally, in retrieval-only mode, a lexical sanity check can
    prevent refusal when keywords from the query appear in retrieved chunks.
    
    Args:
        scores: List of confidence scores (already normalized 0-1)
        confidence_threshold: Base minimum score to answer confidently
        refusal_threshold: Base minimum score to provide any answer
        generation_enabled: Whether LLM generation is enabled
        query: Original query (for lexical check in retrieval-only mode)
        top_chunks: Top retrieved chunks (for lexical check)
    
    Returns:
        ConfidenceAssessment with level, scores, and decision
    """
    if not scores:
        return ConfidenceAssessment(
            level=ConfidenceLevel.INSUFFICIENT,
            top_score=0.0,
            mean_score=0.0,
            should_refuse=True,
            should_warn=False,
            reason="No retrieval results",
        )
    
    top_score = max(scores)
    mean_score = sum(scores) / len(scores)
    
    # Select thresholds based on mode
    if generation_enabled:
        thresholds = GENERATION_THRESHOLDS
    else:
        thresholds = RETRIEVAL_ONLY_THRESHOLDS
    
    high_threshold = thresholds["high"]
    low_threshold = thresholds["low"]
    refuse_threshold = thresholds["refuse"]
    
    # Determine confidence level
    if top_score >= high_threshold:
        level = ConfidenceLevel.HIGH
    elif top_score >= low_threshold:
        level = ConfidenceLevel.LOW
    else:
        level = ConfidenceLevel.INSUFFICIENT
    
    # Determine if we should refuse
    # In generation mode: refuse if LOW or INSUFFICIENT
    # In retrieval-only mode: more nuanced approach
    lexical_match = False
    
    if generation_enabled:
        # Generation mode: stricter - refuse unless HIGH
        should_refuse = level != ConfidenceLevel.HIGH
        should_warn = level == ConfidenceLevel.LOW
    else:
        # Retrieval-only mode: nuanced approach
        # - HIGH confidence: never refuse
        # - LOW confidence: don't refuse (user can see sources)
        # - INSUFFICIENT confidence: check lexical evidence
        
        if level == ConfidenceLevel.HIGH:
            should_refuse = False
            should_warn = False
        elif level == ConfidenceLevel.LOW:
            # Low but not insufficient - show results with warning
            should_refuse = False
            should_warn = True
        else:
            # INSUFFICIENT: require lexical evidence to NOT refuse
            # This prevents showing irrelevant results while allowing
            # relevant results with low vector similarity
            if query and top_chunks:
                lexical_match = has_lexical_overlap(query, top_chunks, min_matches=2, top_n=3)
                if lexical_match:
                    # Evidence is present lexically - don't refuse
                    should_refuse = False
                    should_warn = True
                    logger.info(
                        "Lexical sanity check prevented refusal",
                        top_score=round(top_score, 4),
                        query_preview=query[:50],
                    )
                else:
                    # No lexical evidence - refuse
                    should_refuse = True
                    should_warn = False
            else:
                should_refuse = True
                should_warn = False
    
    # Build reason string
    reason = None
    if should_refuse:
        if level == ConfidenceLevel.INSUFFICIENT:
            reason = f"Top confidence {top_score:.2%} below refuse threshold {refuse_threshold:.2%}"
        else:
            reason = f"Top confidence {top_score:.2%} below high threshold {high_threshold:.2%}"
    elif should_warn:
        reason = f"Top confidence {top_score:.2%} is moderate (threshold: {high_threshold:.2%})"
    
    return ConfidenceAssessment(
        level=level,
        top_score=top_score,
        mean_score=mean_score,
        should_refuse=should_refuse,
        should_warn=should_warn,
        reason=reason,
        lexical_match=lexical_match,
    )


def log_refusal_event(
    workspace_id: str,
    query: str,
    assessment: ConfidenceAssessment,
    top_chunks: list[dict[str, Any]] | None = None,
) -> None:
    """
    Log a refusal event for monitoring and analysis.
    
    All refusal events are logged at WARNING level for easy filtering.
    """
    chunk_info = []
    if top_chunks:
        chunk_info = [
            {
                "chunk_id": c.get("chunk_id", "")[:8],
                "document_id": c.get("document_id", "")[:8],
                "score": c.get("score", 0),
            }
            for c in top_chunks[:3]  # Log top 3 chunks
        ]
    
    logger.warning(
        "REFUSAL_EVENT",
        event_type="confidence_refusal",
        timestamp=datetime.utcnow().isoformat(),
        workspace_id=workspace_id,
        query_preview=query[:100] + "..." if len(query) > 100 else query,
        query_length=len(query),
        confidence_level=assessment.level.value,
        top_score=round(assessment.top_score, 4),
        mean_score=round(assessment.mean_score, 4),
        reason=assessment.reason,
        nearest_chunks=chunk_info,
    )


def compute_confidence(raw_score: float, method: str = "sigmoid") -> float:
    """
    Convert raw similarity score to confidence score (0-1).
    
    Args:
        raw_score: Raw similarity score from vector search
        method: Normalization method ("sigmoid", "linear", "minmax")
    
    Returns:
        Normalized confidence score between 0 and 1
    """
    if method == "sigmoid":
        # Sigmoid transformation for smooth 0-1 mapping
        # Assumes scores typically range from 0.3 to 0.9
        centered = (raw_score - 0.6) * 10
        return 1 / (1 + math.exp(-centered))
    
    elif method == "linear":
        # Simple linear clipping
        return max(0.0, min(1.0, raw_score))
    
    elif method == "minmax":
        # Min-max normalization assuming typical range
        min_score, max_score = 0.2, 0.95
        normalized = (raw_score - min_score) / (max_score - min_score)
        return max(0.0, min(1.0, normalized))
    
    else:
        return raw_score


def rerank_results(
    results: list[dict[str, Any]],
    query: str,
    method: str = "mmr",
    lambda_param: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Rerank search results for diversity and relevance.
    
    Args:
        results: Initial search results
        query: Original query
        method: Reranking method ("mmr" for Maximal Marginal Relevance)
        lambda_param: Trade-off between relevance and diversity (0-1)
    
    Returns:
        Reranked results
    """
    if not results or len(results) <= 1:
        return results
    
    if method == "mmr":
        return _mmr_rerank(results, lambda_param)
    
    # Default: sort by score
    return sorted(results, key=lambda x: x.get("score", 0), reverse=True)


def _mmr_rerank(
    results: list[dict[str, Any]],
    lambda_param: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Maximal Marginal Relevance reranking.
    
    Balances relevance with diversity to avoid redundant results.
    """
    if not results:
        return []
    
    # Start with highest scoring result
    reranked = [results[0]]
    remaining = results[1:]
    
    while remaining:
        best_score = -float("inf")
        best_idx = 0
        
        for i, candidate in enumerate(remaining):
            relevance = candidate.get("score", 0)
            
            # Compute max similarity to already selected results
            # Using simple content overlap as proxy
            max_redundancy = 0
            for selected in reranked:
                overlap = _content_overlap(
                    candidate.get("content", ""),
                    selected.get("content", ""),
                )
                max_redundancy = max(max_redundancy, overlap)
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_redundancy
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        reranked.append(remaining[best_idx])
        remaining.pop(best_idx)
    
    return reranked


def _content_overlap(text1: str, text2: str) -> float:
    """Compute simple content overlap between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def filter_by_threshold(
    results: list[dict[str, Any]],
    min_confidence: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Filter results below confidence threshold.
    
    Args:
        results: Search results
        min_confidence: Minimum confidence score to include
    
    Returns:
        Filtered results
    """
    filtered = [
        r for r in results
        if compute_confidence(r.get("score", 0)) >= min_confidence
    ]
    
    logger.info(
        "Filtered results by threshold",
        original_count=len(results),
        filtered_count=len(filtered),
        threshold=min_confidence,
    )
    
    return filtered


def aggregate_document_scores(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Aggregate chunk scores to document-level scores.
    
    Useful for document-level ranking when multiple chunks
    from the same document are retrieved.
    """
    doc_scores: dict[str, list[float]] = {}
    
    for result in results:
        doc_id = result.get("document_id", "unknown")
        score = result.get("score", 0)
        
        if doc_id not in doc_scores:
            doc_scores[doc_id] = []
        doc_scores[doc_id].append(score)
    
    # Aggregate using max score (could also use mean, sum, etc.)
    return {
        doc_id: max(scores)
        for doc_id, scores in doc_scores.items()
    }
