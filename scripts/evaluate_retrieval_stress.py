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
import json
import os
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
    
    # -------------------------------------------------------------------------
    # Scenario 4: LONG DOCUMENTS
    # -------------------------------------------------------------------------
    # Documents with extensive content (50-150 chunks). Tests whether retrieval
    # and confidence scoring remain effective with large document collections.
    # -------------------------------------------------------------------------
    "long_doc": {
        "description": (
            "Long document scenario: Tests retrieval against documents with "
            "extensive content (50-150 chunks). Validates that the system can "
            "effectively retrieve relevant information from large documents and "
            "that confidence scoring remains meaningful despite document size."
        ),
        "queries": [
            {
                "query": "What is the company's dress code policy?",
                "reason": "Query targeting specific section in long document",
                "expected_behavior": "Should retrieve relevant section; confidence should reflect relevance, not document length",
            },
            {
                "query": "What are the remote work guidelines?",
                "reason": "Another specific query in long document",
                "expected_behavior": "Should find relevant section among many chunks",
            },
        ],
    },
    
    # -------------------------------------------------------------------------
    # Scenario 5: NEEDLE IN HAYSTACK
    # -------------------------------------------------------------------------
    # Documents where a single sentence contains the answer, buried in
    # extensive unrelated content. Tests precision of retrieval.
    # -------------------------------------------------------------------------
    "needle_in_haystack": {
        "description": (
            "Needle in haystack scenario: Tests retrieval when the answer is "
            "contained in a single sentence buried within extensive unrelated content. "
            "Validates that the system can precisely locate specific information "
            "even when it represents a small fraction of the total document."
        ),
        "queries": [
            {
                "query": "What is the emergency contact number for after-hours IT support?",
                "reason": "Very specific query targeting a single sentence in a long document",
                "expected_behavior": "Should retrieve the exact sentence containing the phone number",
            },
            {
                "query": "What was the Q3 revenue total?",
                "reason": "Specific numerical fact buried in financial report",
                "expected_behavior": "Should find the specific revenue figure among extensive financial details",
            },
        ],
    },
    # -------------------------------------------------------------------------
    # Scenario 6: ENTITY-FACT LOOKUP (guardrail)
    # -------------------------------------------------------------------------
    # Query that triggers entity_fact_lookup; should be refused when evidence
    # is not present (no lexical match for CEO/ACME in stress docs).
    # -------------------------------------------------------------------------
    "entity_fact": {
        "description": (
            "Entity-fact lookup scenario: Query that matches entity-fact intent "
            "(e.g. CEO of company). Should trigger entity_fact_lookup and be refused "
            "when the fact is not present in the corpus."
        ),
        "queries": [
            {
                "query": "Who is the CEO of ACME?",
                "reason": "Entity-fact lookup with no evidence in stress docs",
                "expected_behavior": "Should trigger entity_fact_lookup and be refused (no lexical evidence)",
            },
        ],
    },
    # -------------------------------------------------------------------------
    # Scenario 7: PARAPHRASE (synonyms / indirect wording)
    # -------------------------------------------------------------------------
    # Queries that rephrase long_doc / needle content with no direct lexical
    # overlap. Goal: at least one lands LOW (not refused) to validate warning behavior.
    # -------------------------------------------------------------------------
    "paraphrase": {
        "description": (
            "Paraphrase scenario: Queries using synonyms and indirect wording "
            "against long_doc and needle_in_haystack content. No direct lexical "
            "overlap; validates semantic retrieval and that at least one query "
            "lands LOW (warning) without refusal."
        ),
        "queries": [
            {
                "query": "What should staff wear to the office?",
                "reason": "Paraphrase of dress code (long_doc); no 'dress code' wording",
                "expected_behavior": "May land HIGH or LOW; semantic match to dress code section",
            },
            {
                "query": "Who do I call when systems go down outside business hours?",
                "reason": "Paraphrase of after-hours IT contact (needle); no 'emergency' or '1-800'",
                "expected_behavior": "Should retrieve needle; may land LOW due to indirect wording",
            },
            {
                "query": "How much did the company make in the third quarter?",
                "reason": "Paraphrase of Q3 revenue (needle); no 'Q3' or 'revenue' lexical match",
                "expected_behavior": "Semantic match to revenue figure; may land LOW or HIGH",
            },
        ],
    },
    # -------------------------------------------------------------------------
    # Scenario 8: NEGATIVE CONTROL (not in corpus)
    # -------------------------------------------------------------------------
    # Out-of-corpus queries with minimal semantic overlap. Assertions:
    # 1) At least one negative control must be refused (e.g. Q2 via constraint guardrail).
    # 2) No negative control may have level HIGH (no over-confiding).
    # -------------------------------------------------------------------------
    "negative_control": {
        "description": (
            "Negative control scenario: Queries not answered in the stress corpus. "
            "At least one must be refused; none may be HIGH (evaluator fails otherwise)."
        ),
        "queries": [
            {
                "query": "What was the Q2 revenue total?",
                "reason": "Q2 not in corpus; constraint guardrail refuses",
                "expected_behavior": "Must be refused (INSUFFICIENT)",
            },
            {
                "query": "What is the refund policy for training courses?",
                "reason": "No refund/training policy in stress docs",
                "expected_behavior": "Must not be HIGH; ideally refused",
            },
            {
                "query": "When does the Tokyo office open?",
                "reason": "No Tokyo office in corpus",
                "expected_behavior": "Must not be HIGH; ideally refused",
            },
            {
                "query": "What is the company's policy on cryptocurrency payments?",
                "reason": "Topic not in any stress document",
                "expected_behavior": "Must not be HIGH; ideally refused",
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
    Extract doc_hit_count from the chat response in all supported shapes.

    Tries: confidence.doc_hit_count, router_decision.factors.doc_hit_count,
    confidence.doc_summary hit_count, then derives from sources (count of
    chunks from the best document = first source's document_id).
    """
    def _to_int(v: Any) -> int | None:
        if v is None:
            return None
        if isinstance(v, int):
            return v if v >= 0 else None
        if isinstance(v, (float, str)):
            try:
                n = int(float(v))
                return n if n >= 0 else None
            except (ValueError, TypeError):
                return None
        return None

    # 1) Top-level confidence (or nested under any key)
    confidence = response.get("confidence")
    if isinstance(confidence, dict):
        hit = _to_int(confidence.get("doc_hit_count"))
        if hit is not None:
            return hit
        doc_summary = confidence.get("doc_summary")
        if isinstance(doc_summary, dict):
            max_count = 0
            for doc_info in doc_summary.values():
                if isinstance(doc_info, dict):
                    max_count = max(max_count, doc_info.get("hit_count", 0) or 0)
            if max_count > 0:
                return max_count

    # 2) router_decision.factors (some APIs nest diagnostics there)
    router = response.get("router_decision")
    if isinstance(router, dict):
        factors = router.get("factors") or {}
        if isinstance(factors, dict):
            hit = _to_int(factors.get("doc_hit_count"))
            if hit is not None:
                return hit

    # 3) Derive from sources: count chunks from the best document (first source)
    sources = response.get("sources") or []
    if sources:
        first_doc_id = sources[0].get("document_id") or (sources[0].get("chunk_id") or "")[:8]
        if first_doc_id:
            count = sum(
                1 for s in sources
                if (s.get("document_id") or (s.get("chunk_id") or "")[:8]) == first_doc_id
            )
            return count if count > 0 else None

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
    debug_printed_failing: list[bool],
    debug_printed_passing: list[bool],
    verbose: bool = False,
) -> list[StressTestResult]:
    """
    Run all queries in a scenario and collect results.

    Args:
        workspace: Workspace to query
        scenario_name: Name of the scenario
        scenario: Scenario configuration
        verbose: Whether to print verbose output
        debug_printed_failing: Single-element list to track if we printed one failing case
        debug_printed_passing: Single-element list to track if we printed one passing case

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

        # Verbose only: print exact chat response for one failing and one passing DocHits case
        if verbose and "error" not in response:
            if result.doc_hit_count is None and not debug_printed_failing[0]:
                debug_printed_failing[0] = True
                print(f"\n  [DEBUG DocHits MISSING] scenario={scenario_name!r} query={query[:50]!r}...")
                print("  Response keys:", list(response.keys()))
                print("  confidence:", json.dumps(response.get("confidence"), indent=4))
                if response.get("router_decision"):
                    print("  router_decision.factors:", json.dumps(response.get("router_decision", {}).get("factors"), indent=4))
            elif result.doc_hit_count is not None and not debug_printed_passing[0]:
                debug_printed_passing[0] = True
                print(f"\n  [DEBUG DocHits PRESENT] scenario={scenario_name!r} query={query[:50]!r}... doc_hit_count={result.doc_hit_count}")
                print("  Response keys:", list(response.keys()))
                print("  confidence:", json.dumps(response.get("confidence"), indent=4))
                if response.get("router_decision"):
                    print("  router_decision.factors:", json.dumps(response.get("router_decision", {}).get("factors"), indent=4))

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
            doc_hits = str(result.doc_hit_count) if result.doc_hit_count is not None else "-"
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


def validate_constraint_guardrail(all_results: list[StressTestResult]) -> None:
    """
    Assert constraint-coverage guardrail: Q2 revenue must be refused, Q3 revenue allowed.
    Exits with non-zero if assertions fail.
    """
    q2_query = "What was the Q2 revenue total?"
    q3_query = "What was the Q3 revenue total?"
    q2_result = next((r for r in all_results if r.query.strip() == q2_query), None)
    q3_result = next((r for r in all_results if r.query.strip() == q3_query), None)
    failed = []
    if q2_result is None:
        failed.append(f"Q2 query not found in results: {q2_query!r}")
    elif not q2_result.refused:
        failed.append(f"Q2 must be refused (constraint not in corpus); got refused={q2_result.refused}")
    if q3_result is None:
        failed.append(f"Q3 query not found in results: {q3_query!r}")
    elif q3_result.refused:
        failed.append(f"Q3 must remain allowed (constraint in corpus); got refused={q3_result.refused}")
    if failed:
        print("\nConstraint guardrail validation FAILED:")
        for msg in failed:
            print(f"  - {msg}")
        sys.exit(1)
    print("\nConstraint guardrail: Q2 refused, Q3 allowed (PASS)")


def validate_negative_controls(all_results: list[StressTestResult]) -> None:
    """
    Assert: (1) At least one negative_control query is refused.
            (2) No negative_control query has level HIGH (no over-confiding).
    Exits with non-zero if either assertion fails.
    """
    negative_results = [r for r in all_results if r.scenario == "negative_control"]
    failed = []
    refused_count = sum(1 for r in negative_results if r.refused)
    if refused_count == 0:
        failed.append("At least one negative control must be refused (e.g. Q2 via constraint guardrail)")
    for r in negative_results:
        if r.confidence_level == "high":
            failed.append(
                f"Negative control must not be HIGH: {r.query!r} (got level=high, refused={r.refused})"
            )
    if failed:
        print("\nNegative control validation FAILED:")
        for msg in failed:
            print(f"  - {msg}")
        sys.exit(1)
    if negative_results:
        print(
            f"\nNegative controls: {refused_count}/{len(negative_results)} refused, "
            f"none HIGH (PASS)"
        )


def load_golden_eval(path: str) -> list[dict[str, Any]]:
    """Load golden eval set from JSON file. Returns list of { scenario, query, expected_refused, expected_level?, min_doc_hits? }."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Golden eval file must be a JSON array, got {type(data)}")
    return data


def validate_golden_eval(all_results: list[StressTestResult], golden: list[dict[str, Any]]) -> None:
    """
    Validate results against golden expectations.
    Each golden row: scenario, query, expected_refused, expected_level? (high|low|insufficient), min_doc_hits? (optional).
    Exits with non-zero if any expectation fails.
    """
    result_by_key = {(r.scenario, r.query.strip()): r for r in all_results}
    failed = []
    for i, row in enumerate(golden):
        scenario = row.get("scenario")
        query = row.get("query", "").strip()
        expected_refused = row.get("expected_refused")
        expected_level = row.get("expected_level")
        min_doc_hits = row.get("min_doc_hits")
        if scenario is None or query == "" or expected_refused is None:
            failed.append(f"Golden row {i+1}: missing scenario, query, or expected_refused")
            continue
        key = (scenario, query)
        result = result_by_key.get(key)
        if result is None:
            failed.append(f"Golden row {i+1}: no result for scenario={scenario!r} query={query!r}")
            continue
        if result.refused != expected_refused:
            failed.append(
                f"Golden row {i+1}: expected_refused={expected_refused} but got refused={result.refused} "
                f"(scenario={scenario!r} query={query!r})"
            )
        if expected_level is not None and result.confidence_level != expected_level.lower():
            failed.append(
                f"Golden row {i+1}: expected_level={expected_level!r} but got {result.confidence_level!r} "
                f"(scenario={scenario!r} query={query!r})"
            )
        if min_doc_hits is not None and result.doc_hit_count is not None and result.doc_hit_count < min_doc_hits:
            failed.append(
                f"Golden row {i+1}: min_doc_hits={min_doc_hits} but got doc_hit_count={result.doc_hit_count} "
                f"(scenario={scenario!r} query={query!r})"
            )
    if failed:
        print("\nGolden eval validation FAILED:")
        for msg in failed:
            print(f"  - {msg}")
        sys.exit(1)
    print(f"\nGolden eval: {len(golden)} expectations passed (PASS)")


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
    parser.add_argument(
        "--golden",
        metavar="FILE",
        default=None,
        help="Path to golden eval JSON (e.g. scripts/golden_eval.json); validate results against expectations",
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
    debug_printed_failing = [False]
    debug_printed_passing = [False]

    for scenario_name, scenario in scenarios_to_run:
        results = run_scenario(
            args.workspace,
            scenario_name,
            scenario,
            debug_printed_failing,
            debug_printed_passing,
            args.verbose,
        )
        all_results.extend(results)
    
    # Print summary
    print_summary(all_results)

    # Validate constraint-coverage guardrail (Q2 refused, Q3 allowed)
    validate_constraint_guardrail(all_results)
    # Validate negative controls: all must be refused
    validate_negative_controls(all_results)
    # Optional: validate against golden eval set
    if args.golden:
        golden_path = os.path.abspath(args.golden) if os.path.isabs(args.golden) else os.path.normpath(os.path.join(os.getcwd(), args.golden))
        if not os.path.isfile(golden_path):
            print(f"\nERROR: Golden eval file not found: {golden_path}")
            sys.exit(1)
        golden = load_golden_eval(golden_path)
        validate_golden_eval(all_results, golden)

    print(f"\n{'=' * 80}")
    print("  Evaluation Complete")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
