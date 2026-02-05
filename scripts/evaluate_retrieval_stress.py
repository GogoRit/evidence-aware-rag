#!/usr/bin/env python3
"""
Retrieval Stress Test Evaluation Script

This script evaluates retrieval behavior under challenging document conditions:

1. SPARSE DOCUMENTS (1-2 chunks):
   - Tests whether confidence remains meaningful when documents have minimal content
   - Validates that refusal logic works correctly with limited evidence
   - Expected: Lower doc_hit_count, potentially lower confidence

2. REDUNDANT DOCUMENTS (many near-duplicate chunks):
   - Tests whether document aggregation correctly handles repeated content
   - Validates that confidence doesn't artificially inflate from redundancy
   - Expected: Higher doc_hit_count, stable confidence (not inflated)

3. CONFLICTING DOCUMENTS (contradictory statements):
   - Tests whether retrieval surfaces conflicting evidence appropriately
   - Validates confidence behavior when document contains contradictions
   - Expected: System should still return results but may show lower confidence

For each scenario, the script reports:
- confidence_level (HIGH / LOW / INSUFFICIENT)
- refusal_decision (True / False)
- doc_hit_count (number of chunks from best document)
- doc_top_score vs chunk_top_score (aggregated vs raw)
- lexical_match status
- entity_fact_lookup detection

Usage:
    python scripts/evaluate_retrieval_stress.py [--workspace WORKSPACE]

Requirements:
    - Backend server running on localhost:8000
    - Test documents ingested into the workspace (see setup instructions below)

Setup Instructions:
    Before running, create test documents in the workspace:
    
    1. Sparse document: A policy with only 1-2 sentences
    2. Redundant document: A document that repeats the same information multiple times
    3. Conflicting document: A document with contradictory statements
    
    See the TEST_SCENARIOS dict for expected document content.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Any

import httpx

BASE_URL = "http://localhost:8000"
DEFAULT_WORKSPACE = "stress-test"


@dataclass
class StressTestResult:
    """Result of a single stress test query."""
    scenario: str
    query: str
    confidence_level: str
    refused: bool
    doc_hit_count: int | None
    doc_top_score: float | None
    chunk_top_score: float | None
    raw_top_score: float | None
    lexical_match: bool
    entity_fact_lookup: bool
    num_sources: int
    error: str | None = None


# =============================================================================
# TEST SCENARIOS
# =============================================================================
# Each scenario defines:
# - description: What the test is validating
# - queries: List of queries to test against the scenario
# - expected_behavior: What we expect to see (for documentation, not assertions)
#
# NOTE: The actual test documents need to be created/ingested separately.
# These scenarios define the QUERIES, not the documents themselves.
# =============================================================================

TEST_SCENARIOS = {
    # -------------------------------------------------------------------------
    # Scenario 1: SPARSE DOCUMENTS
    # -------------------------------------------------------------------------
    # Documents with minimal content (1-2 chunks). Tests whether confidence
    # scoring works correctly when there's limited evidence to aggregate.
    # -------------------------------------------------------------------------
    "sparse": {
        "description": (
            "Sparse document scenario: Tests retrieval against documents with "
            "very few chunks (1-2). Validates that confidence doesn't artificially "
            "inflate when there's minimal content, and that refusal logic works "
            "correctly with limited evidence."
        ),
        "queries": [
            {
                "query": "What is the vacation policy?",
                "reason": "Simple factual query against sparse document",
                "expected_behavior": "Should return result if content matches, with lower doc_hit_count",
            },
            {
                "query": "How many vacation days do employees get?",
                "reason": "Specific detail query - may not be in sparse document",
                "expected_behavior": "May refuse if detail not present in sparse content",
            },
        ],
    },
    
    # -------------------------------------------------------------------------
    # Scenario 2: REDUNDANT DOCUMENTS
    # -------------------------------------------------------------------------
    # Documents with repeated/near-duplicate content. Tests whether document
    # aggregation correctly handles redundancy without inflating confidence.
    # -------------------------------------------------------------------------
    "redundant": {
        "description": (
            "Redundant document scenario: Tests retrieval against documents with "
            "many near-duplicate chunks. Validates that document-level aggregation "
            "doesn't artificially inflate confidence from repeated content, and that "
            "the system correctly identifies this as a single source of evidence."
        ),
        "queries": [
            {
                "query": "What is the expense reimbursement process?",
                "reason": "Query against document with repeated expense info",
                "expected_behavior": "High doc_hit_count but confidence should not exceed normal levels",
            },
            {
                "query": "What expenses can be reimbursed?",
                "reason": "Variant query to test redundancy handling",
                "expected_behavior": "Similar confidence regardless of how many duplicate chunks match",
            },
        ],
    },
    
    # -------------------------------------------------------------------------
    # Scenario 3: CONFLICTING DOCUMENTS
    # -------------------------------------------------------------------------
    # Documents containing contradictory statements. Tests whether retrieval
    # surfaces conflicting evidence appropriately.
    # -------------------------------------------------------------------------
    "conflicting": {
        "description": (
            "Conflicting document scenario: Tests retrieval against documents "
            "containing contradictory statements. Validates that the system can "
            "still retrieve relevant content but may show appropriate uncertainty."
        ),
        "queries": [
            {
                "query": "Is remote work allowed?",
                "reason": "Query where document may have conflicting answers",
                "expected_behavior": "Should return results; confidence may vary based on which chunks match",
            },
            {
                "query": "What are the office attendance requirements?",
                "reason": "Another query that may surface conflicting statements",
                "expected_behavior": "May return multiple chunks with different information",
            },
        ],
    },
}


def check_backend_health() -> bool:
    """Check if the backend server is running and healthy."""
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def run_query(workspace: str, query: str) -> dict[str, Any]:
    """
    Execute a query against the chat endpoint and return the full response.
    
    Args:
        workspace: Workspace ID to query
        query: The query string
        
    Returns:
        Full response dictionary from the API
    """
    try:
        response = httpx.post(
            f"{BASE_URL}/chat",
            json={"workspace_id": workspace, "query": query},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


def extract_stress_test_result(
    scenario: str,
    query: str,
    response: dict[str, Any],
) -> StressTestResult:
    """
    Extract stress test metrics from an API response.
    
    Args:
        scenario: Name of the test scenario
        query: The query that was executed
        response: API response dictionary
        
    Returns:
        StressTestResult with extracted metrics
    """
    if "error" in response:
        return StressTestResult(
            scenario=scenario,
            query=query,
            confidence_level="ERROR",
            refused=True,
            doc_hit_count=None,
            doc_top_score=None,
            chunk_top_score=None,
            raw_top_score=None,
            lexical_match=False,
            entity_fact_lookup=False,
            num_sources=0,
            error=response["error"],
        )
    
    confidence = response.get("confidence", {})
    
    return StressTestResult(
        scenario=scenario,
        query=query,
        confidence_level=confidence.get("level", "unknown"),
        refused=response.get("refused", False),
        doc_hit_count=_extract_doc_hit_count(response),
        doc_top_score=confidence.get("doc_top_score"),
        chunk_top_score=confidence.get("top_score"),
        raw_top_score=confidence.get("raw_top_score"),
        lexical_match=confidence.get("lexical_match", False),
        entity_fact_lookup=_is_entity_fact_lookup(response),
        num_sources=len(response.get("sources", [])),
    )


def _extract_doc_hit_count(response: dict[str, Any]) -> int | None:
    """
    Extract doc_hit_count from response.
    
    This requires the doc_summary to be exposed in the response.
    If not available, we estimate from sources.
    """
    # If doc_summary is exposed in confidence info (future enhancement)
    confidence = response.get("confidence", {})
    doc_summary = confidence.get("doc_summary")
    if doc_summary and isinstance(doc_summary, dict):
        # Return hit count of best document
        max_count = 0
        for doc_info in doc_summary.values():
            if isinstance(doc_info, dict):
                max_count = max(max_count, doc_info.get("hit_count", 0))
        return max_count if max_count > 0 else None
    
    # Fallback: count sources (approximation)
    sources = response.get("sources", [])
    if sources:
        # Group by document_id
        doc_ids = set()
        for source in sources:
            doc_id = source.get("document_id", source.get("chunk_id", "")[:8])
            doc_ids.add(doc_id)
        # If all from same document, return source count
        if len(doc_ids) == 1:
            return len(sources)
    
    return None


def _is_entity_fact_lookup(response: dict[str, Any]) -> bool:
    """Check if the query was classified as an entity-fact lookup."""
    # This would need to be exposed in the response
    # For now, check if it's in confidence info or metadata
    confidence = response.get("confidence", {})
    return confidence.get("entity_fact_lookup", False)


def print_result(result: StressTestResult, verbose: bool = False) -> None:
    """Print a single stress test result."""
    status = "ERROR" if result.error else ("REFUSED" if result.refused else "OK")
    status_color = {
        "OK": "\033[92m",      # Green
        "REFUSED": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
    }
    reset = "\033[0m"
    
    print(f"\n  Query: \"{result.query[:60]}{'...' if len(result.query) > 60 else ''}\"")
    print(f"  Status: {status_color.get(status, '')}{status}{reset}")
    
    if result.error:
        print(f"  Error: {result.error}")
        return
    
    print(f"  Confidence Level: {result.confidence_level.upper()}")
    print(f"  Refused: {result.refused}")
    
    # Score comparison
    if result.chunk_top_score is not None:
        print(f"  Chunk Top Score: {result.chunk_top_score:.2%}")
    if result.doc_top_score is not None:
        print(f"  Doc Top Score:   {result.doc_top_score:.2%}")
    if result.raw_top_score is not None:
        print(f"  Raw FAISS Score: {result.raw_top_score:.4f}")
    
    # Document aggregation info
    if result.doc_hit_count is not None:
        print(f"  Doc Hit Count: {result.doc_hit_count}")
    print(f"  Num Sources: {result.num_sources}")
    
    # Flags
    flags = []
    if result.lexical_match:
        flags.append("lexical_match")
    if result.entity_fact_lookup:
        flags.append("entity_fact_lookup")
    if flags:
        print(f"  Flags: {', '.join(flags)}")


def run_scenario(
    workspace: str,
    scenario_name: str,
    scenario: dict[str, Any],
    verbose: bool = False,
) -> list[StressTestResult]:
    """
    Run all queries in a scenario and collect results.
    
    Args:
        workspace: Workspace to query
        scenario_name: Name of the scenario
        scenario: Scenario configuration
        verbose: Whether to print verbose output
        
    Returns:
        List of StressTestResult for each query
    """
    print(f"\n{'=' * 80}")
    print(f"  Scenario: {scenario_name.upper()}")
    print(f"{'=' * 80}")
    print(f"\n{scenario['description']}\n")
    
    results = []
    for query_config in scenario["queries"]:
        query = query_config["query"]
        
        if verbose:
            print(f"\n  [Running] {query}")
            print(f"  Reason: {query_config['reason']}")
            print(f"  Expected: {query_config['expected_behavior']}")
        
        response = run_query(workspace, query)
        result = extract_stress_test_result(scenario_name, query, response)
        results.append(result)
        
        print_result(result, verbose)
    
    return results


def print_summary(all_results: list[StressTestResult]) -> None:
    """Print a summary table of all results."""
    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}\n")
    
    # Group by scenario
    scenarios = {}
    for result in all_results:
        if result.scenario not in scenarios:
            scenarios[result.scenario] = []
        scenarios[result.scenario].append(result)
    
    # Print table
    print(f"{'Scenario':<12} {'Query':<40} {'Level':<8} {'Refused':<8} {'DocHits':<8}")
    print("-" * 80)
    
    for scenario_name, results in scenarios.items():
        for result in results:
            query_short = result.query[:37] + "..." if len(result.query) > 40 else result.query
            doc_hits = str(result.doc_hit_count) if result.doc_hit_count else "-"
            print(
                f"{scenario_name:<12} "
                f"{query_short:<40} "
                f"{result.confidence_level:<8} "
                f"{str(result.refused):<8} "
                f"{doc_hits:<8}"
            )
    
    # Aggregate stats
    print(f"\n{'-' * 80}")
    total = len(all_results)
    refused = sum(1 for r in all_results if r.refused)
    errors = sum(1 for r in all_results if r.error)
    high = sum(1 for r in all_results if r.confidence_level == "high")
    low = sum(1 for r in all_results if r.confidence_level == "low")
    insufficient = sum(1 for r in all_results if r.confidence_level == "insufficient")
    
    print(f"\nTotal queries: {total}")
    print(f"  Refused: {refused} ({refused/total*100:.1f}%)")
    print(f"  Errors: {errors}")
    print(f"  HIGH confidence: {high} ({high/total*100:.1f}%)")
    print(f"  LOW confidence: {low} ({low/total*100:.1f}%)")
    print(f"  INSUFFICIENT confidence: {insufficient} ({insufficient/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval behavior under stress conditions"
    )
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        help=f"Workspace to test against (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "--scenario",
        choices=list(TEST_SCENARIOS.keys()) + ["all"],
        default="all",
        help="Scenario to run (default: all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("  Retrieval Stress Test Evaluation")
    print("=" * 80)
    
    # Check backend health
    if not check_backend_health():
        print("\nERROR: Backend is not running or not healthy")
        print("Please start the backend with: make dev")
        sys.exit(1)
    
    print(f"\nBackend: Online")
    print(f"Workspace: {args.workspace}")
    
    # Run scenarios
    all_results = []
    scenarios_to_run = (
        TEST_SCENARIOS.items() 
        if args.scenario == "all" 
        else [(args.scenario, TEST_SCENARIOS[args.scenario])]
    )
    
    for scenario_name, scenario in scenarios_to_run:
        results = run_scenario(args.workspace, scenario_name, scenario, args.verbose)
        all_results.extend(results)
    
    # Print summary
    print_summary(all_results)
    
    print(f"\n{'=' * 80}")
    print("  Evaluation Complete")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
