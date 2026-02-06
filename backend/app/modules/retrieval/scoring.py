"""Confidence scoring, refusal logic, and result ranking."""

import math
import re
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
    doc_top_score: float | None = None  # Document-level aggregated confidence (retrieval-only)
    doc_summary: dict[str, dict] | None = None  # Per-document confidence breakdown
    entity_fact_lookup: bool = False  # True if query is an entity-fact lookup intent
    numeric_constraint_lookup: bool = False  # True if numeric/time-slice intent detected
    numeric_constraint_refused: bool = False  # True if refused due to missing constraint in evidence


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


# Patterns for entity-fact lookup intents that require high-precision answers
# These queries ask about specific organizational/entity facts that should only
# be answered if explicitly documented. Conservative patterns to minimize false positives.
ENTITY_FACT_PATTERNS = [
    r"\bwho\s+is\s+(the\s+)?ceo\b",
    r"\bceo\s+of\b",
    r"\bchief\s+executive\s+officer\b",
    r"\bfounder\s+of\b",
    r"\bco-?founder\b",
    r"\bfounded\s+by\b",
    r"\bwho\s+founded\b",
    r"\bheadquarters?\b",
    r"\bhq\s+of\b",
    r"\bwhere\s+is\s+.+\s+(located|based|headquartered)\b",
    r"\brevenue\s+of\b",
    r"\bvaluation\s+of\b",
    r"\bnet\s+worth\b",
    r"\bpublicly\s+traded\b",
    r"\bticker\s+symbol\b",
    r"\bstock\s+ticker\b",
    r"\baddress\s+of\b",
    r"\bwho\s+(owns|acquired|bought)\b",
    r"\bwhen\s+was\s+.+\s+(founded|established|incorporated)\b",
]

# Compiled regex for efficiency
_ENTITY_FACT_REGEX = re.compile(
    "|".join(ENTITY_FACT_PATTERNS),
    re.IGNORECASE
)

# Numeric/time-slice constraint patterns (quarters, years, months, phone, dollar amounts)
_QUARTER_ABBREV = re.compile(r"\bq([1-4])\b", re.IGNORECASE)
_QUARTER_ORDINAL = re.compile(
    r"\b(first|second|third|fourth)\s+quarter\b",
    re.IGNORECASE
)
_ORDINAL_TO_Q = {"first": "q1", "second": "q2", "third": "q3", "fourth": "q4"}
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_PHONE = re.compile(r"\b1-\d{3}-\d{3}-\d{4}\b|\b\d{3}-\d{3}-\d{4}\b")
# Intent: query mentions quarter (Q1-Q4 or ordinal), year, revenue, phone, or dollar amount
_NUMERIC_INTENT = re.compile(
    r"\bq[1-4]\b|"
    r"\b(first|second|third|fourth)\s+quarter\b|"
    r"\b(19|20)\d{2}\b|"
    r"\brevenue\b|\bquarter\b|"
    r"\b\d+(\.\d+)?\s*(million|billion|k|m)\b|"
    r"\$\d+|"
    r"1-\d{3}-\d{3}-\d{4}|\d{3}-\d{3}-\d{4}",
    re.IGNORECASE
)


def extract_numeric_constraint_tokens(query: str) -> set[str]:
    """
    Extract constraint tokens from a query for numeric/time-slice guardrail.
    Normalizes synonyms (e.g. Q2, second quarter -> q2). Returns empty set if
    no numeric/time-slice constraints found.
    """
    if not query:
        return set()
    if not _NUMERIC_INTENT.search(query):
        return set()
    tokens = set()
    q = query.lower()
    # Quarter abbreviations: Q1..Q4 (regex + defensive substring for reliability)
    for m in _QUARTER_ABBREV.finditer(query):
        tokens.add("q" + m.group(1).lower())
    for quarter in ("q1", "q2", "q3", "q4"):
        if quarter in q:
            tokens.add(quarter)
    # Quarter ordinals: first/second/third/fourth quarter
    for m in _QUARTER_ORDINAL.finditer(query):
        ord_word = m.group(1).lower()
        if ord_word in _ORDINAL_TO_Q:
            tokens.add(_ORDINAL_TO_Q[ord_word])
            tokens.add(m.group(0).lower())
    # Years
    for m in _YEAR.finditer(query):
        tokens.add(m.group(0))
    # Phone numbers
    for m in _PHONE.finditer(query):
        tokens.add(m.group(0))
    return tokens


def constraint_tokens_in_chunks(
    constraint_tokens: set[str],
    top_chunks: list[dict[str, Any]],
    top_n: int = 5,
) -> bool:
    """Return True if any top chunk content contains at least one constraint token."""
    if not constraint_tokens or not top_chunks:
        return False
    combined = " ".join(c.get("content", "") for c in top_chunks[:top_n]).lower()
    for t in constraint_tokens:
        if t.lower() in combined:
            return True
    return False


def is_entity_fact_lookup(query: str) -> bool:
    """
    Detect if a query is asking for specific entity/organizational facts.
    
    These intents require high-precision answers: if evidence is weak,
    we should refuse rather than show potentially incorrect information.
    
    Examples of entity-fact lookups:
    - "Who is the CEO of ACME?"
    - "Where is the company headquartered?"
    - "What is the revenue of XYZ Corp?"
    
    Args:
        query: The user's query string
        
    Returns:
        True if the query matches entity-fact lookup patterns
    """
    if not query:
        return False
    
    match = _ENTITY_FACT_REGEX.search(query)
    if match:
        logger.debug(
            "Entity-fact lookup detected",
            query_preview=query[:50],
            matched_pattern=match.group(),
        )
        return True
    
    return False


def aggregate_document_confidence(
    top_chunks: list[dict[str, Any]],
    chunk_confidences: list[float],
) -> dict[str, Any]:
    """
    Aggregate chunk-level confidence scores to document-level scores.
    
    Groups results by document_id (or filename if available) and computes:
    - doc_max_confidence: Maximum chunk confidence within the document
    - doc_mean_confidence: Rank-weighted mean confidence
    - doc_hit_count: Number of chunks retrieved from the document
    
    Args:
        top_chunks: List of retrieved chunks with document_id and metadata
        chunk_confidences: Corresponding confidence scores for each chunk
        
    Returns:
        Dictionary containing:
        - best_doc_id: ID of the document with highest aggregated confidence
        - best_doc_confidence: Maximum document-level confidence score
        - doc_summary: Per-document breakdown {doc_id: {max, mean, count}}
    """
    if not top_chunks or not chunk_confidences:
        return {
            "best_doc_id": None,
            "best_doc_confidence": 0.0,
            "doc_summary": {},
        }
    
    # Group chunks by document
    doc_data: dict[str, dict[str, Any]] = {}
    
    for idx, (chunk, confidence) in enumerate(zip(top_chunks, chunk_confidences)):
        # Use document_id, fall back to filename from metadata
        doc_id = chunk.get("document_id") or chunk.get("metadata", {}).get("filename", "unknown")
        
        if doc_id not in doc_data:
            doc_data[doc_id] = {
                "confidences": [],
                "ranks": [],  # 1-indexed ranks for weighting
            }
        
        doc_data[doc_id]["confidences"].append(confidence)
        doc_data[doc_id]["ranks"].append(idx + 1)  # 1-indexed rank
    
    # Compute per-document aggregated scores
    doc_summary: dict[str, dict] = {}
    best_doc_id = None
    best_doc_confidence = 0.0
    
    for doc_id, data in doc_data.items():
        confidences = data["confidences"]
        ranks = data["ranks"]
        
        # Max confidence from any chunk in this document
        doc_max = max(confidences)
        
        # Rank-weighted mean: higher weight for higher-ranked (lower rank number) chunks
        # Weight = 1/rank, normalized
        weights = [1.0 / r for r in ranks]
        weight_sum = sum(weights)
        doc_mean = sum(c * w for c, w in zip(confidences, weights)) / weight_sum if weight_sum > 0 else 0.0
        
        doc_summary[doc_id] = {
            "max_confidence": round(doc_max, 4),
            "mean_confidence": round(doc_mean, 4),
            "hit_count": len(confidences),
        }
        
        # Track best document by max confidence
        if doc_max > best_doc_confidence:
            best_doc_confidence = doc_max
            best_doc_id = doc_id
    
    logger.debug(
        "Document confidence aggregation",
        num_docs=len(doc_summary),
        best_doc_id=best_doc_id[:8] if best_doc_id else None,
        best_doc_confidence=round(best_doc_confidence, 4),
    )
    
    return {
        "best_doc_id": best_doc_id,
        "best_doc_confidence": best_doc_confidence,
        "doc_summary": doc_summary,
    }


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
    - Retrieval-only mode: Uses document-level aggregation for more stable confidence.
      Refuses only when both doc_level_top and chunk_level_top are below threshold
      AND lexical_match is false.
    - Generation mode: Stricter thresholds (refuse < 0.25)
    
    Args:
        scores: List of confidence scores (already normalized 0-1)
        confidence_threshold: Base minimum score to answer confidently
        refusal_threshold: Base minimum score to provide any answer
        generation_enabled: Whether LLM generation is enabled
        query: Original query (for lexical check in retrieval-only mode)
        top_chunks: Top retrieved chunks (for lexical check and doc aggregation)
    
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
    
    # Chunk-level metrics (baseline)
    chunk_level_top = max(scores)
    mean_score = sum(scores) / len(scores)
    
    # Select thresholds based on mode
    if generation_enabled:
        thresholds = GENERATION_THRESHOLDS
    else:
        thresholds = RETRIEVAL_ONLY_THRESHOLDS
    
    high_threshold = thresholds["high"]
    low_threshold = thresholds["low"]
    refuse_threshold = thresholds["refuse"]
    
    # Initialize tracking variables
    doc_top_score = None
    doc_summary = None
    lexical_match = False
    entity_fact_lookup = False
    numeric_constraint_lookup = False
    numeric_constraint_refused = False
    reason = None

    if generation_enabled:
        # Generation mode: use chunk-level confidence only
        primary_score = chunk_level_top
        
        # Determine confidence level
        if primary_score >= high_threshold:
            level = ConfidenceLevel.HIGH
        elif primary_score >= low_threshold:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.INSUFFICIENT
        
        # Generation mode: stricter - refuse unless HIGH
        should_refuse = level != ConfidenceLevel.HIGH
        should_warn = level == ConfidenceLevel.LOW
        
    else:
        # Retrieval-only mode: use document-level aggregation for more stable confidence
        if top_chunks:
            doc_agg = aggregate_document_confidence(top_chunks, scores)
            doc_level_top = doc_agg["best_doc_confidence"]
            doc_top_score = doc_level_top
            doc_summary = doc_agg["doc_summary"]
        else:
            doc_level_top = chunk_level_top
            doc_top_score = doc_level_top
        
        # Use doc_level_top as the primary confidence score for retrieval-only mode
        # This provides more stable confidence when a document clearly contains the answer
        primary_score = doc_level_top
        
        logger.debug(
            "Retrieval-only confidence assessment",
            chunk_level_top=round(chunk_level_top, 4),
            doc_level_top=round(doc_level_top, 4),
            primary_score=round(primary_score, 4),
        )
        
        # Determine confidence level based on doc_level_top
        if primary_score >= high_threshold:
            level = ConfidenceLevel.HIGH
        elif primary_score >= low_threshold:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.INSUFFICIENT

        should_refuse = False
        should_warn = False

        # Retrieval-only mode: production-defensible refusal logic
        # HIGH: allow, no warning (unless entity-fact lookup without evidence)
        # LOW: allow with warning (handles paraphrases, synonyms without lexical match)
        # INSUFFICIENT: refuse unless lexical_match saves it
        # 
        # Entity-fact lookup guardrail: for high-precision intents (CEO, headquarters, etc.),
        # require lexical evidence regardless of confidence level. These queries match org names
        # in headers but the actual fact (CEO, HQ address) may not be in the document.
        
        # Detect entity-fact lookup intent
        entity_fact_lookup = is_entity_fact_lookup(query) if query else False
        
        # Compute lexical overlap:
        # - Always compute for entity-fact lookups (guardrail needs this)
        # - Also compute for LOW/INSUFFICIENT cases
        if entity_fact_lookup or level in (ConfidenceLevel.LOW, ConfidenceLevel.INSUFFICIENT):
            if query and top_chunks:
                lexical_match = has_lexical_overlap(query, top_chunks, min_matches=2, top_n=3)
        
        # Entity-fact lookup guardrail applies at ALL confidence levels
        # These intents require explicit lexical evidence in the document
        if entity_fact_lookup and not lexical_match:
            should_refuse = True
            should_warn = False
            logger.debug(
                "Entity-fact lookup refused - no lexical evidence",
                level=level.value,
                doc_score=round(primary_score, 4),
                query_preview=query[:50] if query else "",
            )
        else:
            # Numeric/time-slice constraint guardrail: require constraint in evidence
            constraint_tokens = extract_numeric_constraint_tokens(query) if query else set()
            if constraint_tokens:
                numeric_constraint_lookup = True
                if not constraint_tokens_in_chunks(constraint_tokens, top_chunks):
                    numeric_constraint_refused = True
                    should_refuse = True
                    should_warn = False
                    level = ConfidenceLevel.INSUFFICIENT
                    # Use first missing token for reason (e.g. "Q2")
                    missing_display = next(iter(sorted(constraint_tokens))).upper()
                    reason = f"Missing query constraint in evidence ({missing_display} not found)"
                    logger.debug(
                        "Numeric constraint refused - constraint not in top chunks",
                        constraint_tokens=list(constraint_tokens),
                        query_preview=query[:50] if query else "",
                    )
        if not should_refuse and not numeric_constraint_refused:
            if level == ConfidenceLevel.HIGH:
                should_refuse = False
                should_warn = False
            elif level == ConfidenceLevel.LOW:
                # Normal LOW confidence: allow with warning
                # This is production-defensible for paraphrases/synonyms
                should_refuse = False
                should_warn = True
                logger.debug(
                    "LOW confidence - allowing with warning",
                    doc_score=round(primary_score, 4),
                    lexical_match=lexical_match,
                    entity_fact_lookup=entity_fact_lookup,
                    query_preview=query[:50] if query else "",
                )
            else:
                # INSUFFICIENT: refuse unless lexical evidence provides a safety net
                if lexical_match:
                    # Lexical evidence found - allow but with low confidence
                    should_refuse = False
                    should_warn = True
                    # Boost primary_score to low threshold when lexical match found
                    primary_score = max(primary_score, low_threshold)
                    level = ConfidenceLevel.LOW
                    logger.info(
                        "Lexical match boosted confidence",
                        original_doc_score=round(doc_level_top, 4),
                        boosted_score=round(primary_score, 4),
                        query_preview=query[:50] if query else "",
                    )
                else:
                    # No lexical evidence and insufficient semantic match - refuse
                    should_refuse = True
                    should_warn = False
    
    # Build reason string (may already be set by numeric constraint guardrail)
    if should_refuse and reason is None:
        if entity_fact_lookup and not lexical_match:
            reason = "Entity-fact lookup requires explicit evidence in documents"
        elif numeric_constraint_refused:
            pass  # reason already set in guardrail block
        elif level == ConfidenceLevel.INSUFFICIENT:
            reason = f"Top confidence {primary_score:.2%} below refuse threshold {refuse_threshold:.2%}"
        else:
            reason = f"Top confidence {primary_score:.2%} below high threshold {high_threshold:.2%}"
    elif should_warn and reason is None:
        reason = f"Top confidence {primary_score:.2%} is moderate (threshold: {high_threshold:.2%})"
    
    return ConfidenceAssessment(
        level=level,
        top_score=primary_score,  # Use doc-level as primary in retrieval-only mode
        mean_score=mean_score,    # Keep chunk-level mean for reference
        should_refuse=should_refuse,
        should_warn=should_warn,
        reason=reason,
        lexical_match=lexical_match,
        doc_top_score=doc_top_score,
        doc_summary=doc_summary,
        entity_fact_lookup=entity_fact_lookup,
        numeric_constraint_lookup=numeric_constraint_lookup,
        numeric_constraint_refused=numeric_constraint_refused,
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
    
    For single-score conversion. For batch/relative computation,
    use compute_confidence_batch() with method="relative".
    
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


def compute_confidence_batch(
    raw_scores: list[float],
    method: str = "relative",
) -> list[float]:
    """
    Compute confidence scores for a batch of raw similarity scores.
    
    The "relative" method computes confidence based on score distribution
    within the batch, combined with absolute score signals. This avoids
    hard-coded assumptions about score ranges while still penalizing
    queries where raw scores indicate poor matches.
    
    Args:
        raw_scores: List of raw similarity scores (higher = more similar)
        method: "relative" (rank-based) or "sigmoid" (absolute mapping)
    
    Returns:
        List of confidence scores in [0, 1]
    
    The "relative" method uses:
    - margin: gap between top score and second score (distinctiveness)
    - separation: gap between top score and mean (overall relevance)
    - absolute_factor: penalty for low raw scores (prevents false confidence)
    """
    if not raw_scores:
        return []
    
    if method == "sigmoid":
        # Fall back to per-score sigmoid transformation
        return [compute_confidence(s, method="sigmoid") for s in raw_scores]
    
    if method != "relative":
        # Unknown method, use linear
        return [compute_confidence(s, method="linear") for s in raw_scores]
    
    # Relative confidence: data-driven with absolute constraints
    n = len(raw_scores)
    if n == 1:
        # Single result: use score directly but bounded
        return [min(1.0, max(0.0, raw_scores[0]))]
    
    top = raw_scores[0]  # Highest score (assuming sorted descending)
    second = raw_scores[1] if n > 1 else top
    mean_score = sum(raw_scores) / n
    min_score = min(raw_scores)
    
    # Margin: distinctiveness of top result vs second
    margin = top - second
    
    # Separation: how much better is top vs average
    separation = top - mean_score
    
    # Score range in this batch (for normalization)
    score_range = top - min_score
    eps = 1e-6
    
    # Normalized margin and separation relative to score range
    norm_margin = margin / (score_range + eps) if score_range > eps else 0.5
    norm_separation = separation / (score_range + eps) if score_range > eps else 0.5
    
    # Combine relative signals
    alpha = 3.0  # Margin sensitivity
    beta = 2.0   # Separation sensitivity
    
    margin_conf = 1 / (1 + math.exp(-alpha * (norm_margin - 0.3)))
    separation_conf = 1 / (1 + math.exp(-beta * (norm_separation - 0.2)))
    
    relative_conf = 0.5 * margin_conf + 0.5 * separation_conf
    
    # Absolute score factor: penalize confidence when raw scores are low
    # This prevents high confidence for queries that don't match well
    # Threshold calibrated for typical embedding similarity scores:
    # - scores >= 0.55 get full credit (strong semantic match)
    # - scores < 0.30 are heavily penalized (weak match)
    # - scores in between scale with emphasis on higher threshold
    abs_threshold_high = 0.55  # Above this, no penalty
    abs_threshold_low = 0.30   # Below this, significant penalty
    
    if top >= abs_threshold_high:
        absolute_factor = 1.0
    elif top <= abs_threshold_low:
        # Low raw score -> cap confidence regardless of relative metrics
        # Maps [0, abs_threshold_low] -> [0.15, 0.4]
        absolute_factor = 0.15 + 0.25 * (top / abs_threshold_low)
    else:
        # Non-linear interpolation (quadratic) - penalizes mid-range more
        # Maps [abs_threshold_low, abs_threshold_high] -> [0.4, 1.0]
        range_size = abs_threshold_high - abs_threshold_low
        position = (top - abs_threshold_low) / range_size
        # Quadratic curve: slower growth for lower positions
        absolute_factor = 0.4 + 0.6 * (position ** 1.5)
    
    # Final confidence: relative metrics weighted by absolute factor
    # This ensures high relative confidence can't overcome low absolute scores
    top_conf = relative_conf * absolute_factor
    
    # Floor: ensure minimum confidence based on raw top score
    # If raw score is high (> 0.5), don't let relative method push too low
    raw_floor = min(1.0, max(0.0, top)) * 0.4
    top_conf = max(top_conf, raw_floor)
    
    # Cap at 1.0
    top_conf = min(1.0, top_conf)
    
    logger.debug(
        "Relative confidence computation",
        top=round(top, 4),
        second=round(second, 4),
        mean=round(mean_score, 4),
        margin=round(margin, 4),
        norm_margin=round(norm_margin, 4),
        norm_separation=round(norm_separation, 4),
        relative_conf=round(relative_conf, 4),
        absolute_factor=round(absolute_factor, 4),
        top_conf=round(top_conf, 4),
    )
    
    # For other results: scale down based on their relative position
    confidences = []
    for i, score in enumerate(raw_scores):
        if i == 0:
            confidences.append(round(top_conf, 4))
        else:
            # Scale confidence by relative score position
            if score_range > eps:
                rel_pos = (score - min_score) / score_range
            else:
                rel_pos = 0.5
            # Confidence decays from top_conf based on relative position
            conf = top_conf * rel_pos * 0.9
            confidences.append(round(max(0.0, conf), 4))
    
    return confidences


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


