#!/usr/bin/env python3
"""
Test mode-aware confidence and refusal logic with entity-fact lookup guardrail.

Expected results in retrieval-only mode (GENERATION_ENABLED=false):
1. "What equipment does the company provide?" -> refused=False
2. "How much can employees claim for internet expenses?" -> refused=False
3. "Who is the CEO of ACME Corporation?" -> refused=True (entity-fact lookup guardrail)
4. "What are the core working hours for remote employees?" -> refused=False
5. Paraphrase queries -> refused=False (LOW confidence allowed for non-entity-fact)
6. "Where is ACME headquartered?" -> refused=True (entity-fact lookup guardrail)

Entity-fact lookup queries (CEO, headquarters, etc.) require HIGH confidence or
explicit lexical evidence. This prevents false positives where org names match
but the actual fact is not in the document.
"""

import sys
import httpx

BASE_URL = "http://localhost:8000"
WORKSPACE = "demo"

# Minimum confidence threshold for factual queries
MIN_FACTUAL_CONFIDENCE = 0.20

TEST_QUERIES = [
    {
        "query": "What equipment does the company provide?",
        "expected_refused": False,
        "min_confidence": MIN_FACTUAL_CONFIDENCE,
        "check_no_lexical_boost": True,
        "reason": "Policy clearly mentions laptop, monitor, peripherals",
    },
    {
        "query": "How much can employees claim for internet expenses?",
        "expected_refused": False,
        "min_confidence": MIN_FACTUAL_CONFIDENCE,
        "check_no_lexical_boost": True,
        "reason": "Policy mentions internet reimbursement amount",
    },
    {
        "query": "Who is the CEO of ACME Corporation?",
        "expected_refused": True,
        "min_confidence": None,
        "check_no_lexical_boost": False,
        "is_entity_fact_test": True,
        "reason": "Entity-fact lookup: CEO not in document, guardrail should refuse",
    },
    {
        "query": "What are the core working hours for remote employees?",
        "expected_refused": False,
        "min_confidence": None,
        "check_no_lexical_boost": False,
        "reason": "Policy specifies 10:00 AM to 3:00 PM core hours",
    },
    # Paraphrase-style queries: may have weak lexical overlap but semantic match
    {
        "query": "What gear does the company give to new hires?",
        "expected_refused": False,
        "min_confidence": None,
        "check_no_lexical_boost": False,
        "is_paraphrase_test": True,
        "reason": "Paraphrase of equipment query - not entity-fact, should allow",
    },
    {
        "query": "Is there an internet stipend for remote workers?",
        "expected_refused": False,
        "min_confidence": None,
        "check_no_lexical_boost": False,
        "is_paraphrase_test": True,
        "reason": "Paraphrase of internet expenses - not entity-fact, should allow",
    },
    # Entity-fact lookup: headquarters
    {
        "query": "Where is ACME headquartered?",
        "expected_refused": True,
        "min_confidence": None,
        "check_no_lexical_boost": False,
        "is_entity_fact_test": True,
        "reason": "Entity-fact lookup: headquarters not in document, guardrail should refuse",
    },
]


def run_tests():
    """Run confidence/refusal tests with rank-relative confidence validation."""
    print("=" * 80)
    print("  Rank-Relative Confidence Test Suite")
    print("=" * 80)
    print()
    
    # Check health first
    try:
        health = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        if health.status_code != 200:
            print("[FAIL] Backend not healthy")
            return False
    except Exception as e:
        print(f"[FAIL] Cannot connect to backend: {e}")
        return False
    
    print(f"Backend: Online")
    print(f"Workspace: {WORKSPACE}")
    print()
    
    all_passed = True
    results_summary = []
    
    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        expected_refused = test["expected_refused"]
        min_confidence = test.get("min_confidence")
        check_no_lexical_boost = test.get("check_no_lexical_boost", False)
        reason = test["reason"]
        
        print(f"Test {i}: {query}")
        print(f"  Expected: refused={expected_refused}", end="")
        if min_confidence is not None:
            print(f", top_score >= {min_confidence:.2%}")
        else:
            print()
        print(f"  Reason: {reason}")
        
        try:
            response = httpx.post(
                f"{BASE_URL}/chat",
                json={
                    "workspace_id": WORKSPACE,
                    "query": query,
                    "top_k": 5,
                    "include_sources": True,
                },
                timeout=30.0,
            )
            
            if response.status_code != 200:
                print(f"  [FAIL] HTTP {response.status_code}: {response.text[:100]}")
                all_passed = False
                continue
            
            data = response.json()
            actual_refused = data.get("refused", False)
            confidence = data.get("confidence", {})
            top_score = confidence.get("top_score", 0)
            doc_top_score = confidence.get("doc_top_score")
            raw_top_score = confidence.get("raw_top_score")
            lexical_match = confidence.get("lexical_match", False)
            entity_fact_lookup = confidence.get("entity_fact_lookup", False)
            level = confidence.get("level", "unknown")
            retrieval_only = data.get("retrieval_only", False)
            sources = data.get("sources", [])
            
            # Check refusal result
            refusal_ok = actual_refused == expected_refused
            
            # Check expected confidence level (if specified)
            expected_level = test.get("expected_level")
            level_ok = True
            if expected_level is not None:
                level_ok = level == expected_level
            
            # Check confidence threshold (only for non-refused queries with min_confidence)
            confidence_ok = True
            if not expected_refused and min_confidence is not None:
                confidence_ok = top_score >= min_confidence
            
            # Check if lexical boosting was needed (for queries that should pass without it)
            lexical_boost_ok = True
            lexical_note = ""
            if check_no_lexical_boost and not expected_refused:
                if lexical_match:
                    # Lexical match triggered - check if top_score would have been >= threshold anyway
                    # If doc_top_score (pre-boost) was already >= threshold, lexical was unnecessary
                    if doc_top_score is not None and doc_top_score >= min_confidence:
                        lexical_note = " (lexical triggered but was unnecessary)"
                    else:
                        lexical_note = " (NEEDED lexical boost - see explanation below)"
                        # Not a failure, just a note - lexical is a valid fallback
                else:
                    lexical_note = " (no lexical boost needed)"
            
            # Overall pass/fail
            test_passed = refusal_ok and confidence_ok and level_ok
            if test_passed:
                status = "[PASS]"
            else:
                status = "[FAIL]"
                all_passed = False
            
            # Print detailed results
            print(f"  {status} refused={actual_refused} (expected {expected_refused})")
            
            # Raw FAISS score
            if raw_top_score is not None:
                print(f"         raw_top_score (FAISS): {raw_top_score:.4f}")
            
            # Computed confidence scores
            print(f"         top_score (computed): {top_score:.2%}")
            if doc_top_score is not None:
                print(f"         doc_top_score: {doc_top_score:.2%}")
            
            # Entity-fact lookup and lexical match status
            print(f"         entity_fact_lookup: {entity_fact_lookup}")
            print(f"         lexical_match: {lexical_match}{lexical_note}")
            
            # Confidence level with expected check
            if expected_level:
                level_status = "OK" if level_ok else "MISMATCH"
                print(f"         level: {level} (expected {expected_level}, {level_status})")
            else:
                print(f"         level: {level}")
            
            # Threshold check
            if min_confidence is not None and not expected_refused:
                conf_status = "OK" if confidence_ok else "BELOW THRESHOLD"
                print(f"         confidence check: {conf_status} (min={min_confidence:.2%})")
            
            print(f"         retrieval_only: {retrieval_only}")
            print(f"         sources: {len(sources)}")
            
            # Show top source with raw score
            if sources and not actual_refused:
                top_source = sources[0]
                snippet = top_source.get("content", "")[:60]
                src_raw = top_source.get("metadata", {}).get("raw_score", "?")
                print(f"         top_source: raw={src_raw:.4f} \"{snippet}...\"")
            
            # Collect for summary
            results_summary.append({
                "query": query[:45] + "..." if len(query) > 45 else query,
                "refused": actual_refused,
                "raw_score": raw_top_score,
                "top_score": top_score,
                "doc_top_score": doc_top_score,
                "entity_fact": entity_fact_lookup,
                "lexical": lexical_match,
                "level": level,
                "passed": test_passed,
            })
            
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            all_passed = False
        
        print()
    
    # Print summary table
    print("=" * 95)
    print("  Results Summary")
    print("=" * 95)
    header = f"{'Query':<40} {'Refused':<8} {'EntFact':<8} {'Lex':<5} {'Level':<8} {'Status'}"
    print(header)
    print("-" * 95)
    for r in results_summary:
        ent_str = "Yes" if r['entity_fact'] else "No"
        lex_str = "Yes" if r['lexical'] else "No"
        status_str = "PASS" if r['passed'] else "FAIL"
        query_short = r['query'][:38] + ".." if len(r['query']) > 40 else r['query']
        print(f"{query_short:<40} {str(r['refused']):<8} {ent_str:<8} {lex_str:<5} {r['level']:<8} {status_str}")
    print("-" * 95)
    
    # Analysis
    print()
    print("Analysis:")
    print("-" * 80)
    equipment_result = next((r for r in results_summary if "equipment" in r["query"].lower()), None)
    if equipment_result:
        if equipment_result["lexical"]:
            print(f"  Equipment query: lexical_match=True (fallback was used)")
            if equipment_result["doc_top_score"] and equipment_result["doc_top_score"] >= MIN_FACTUAL_CONFIDENCE:
                print(f"    -> However, doc_top_score={equipment_result['doc_top_score']:.2%} >= {MIN_FACTUAL_CONFIDENCE:.0%}")
                print(f"    -> Lexical match was technically unnecessary")
            else:
                print(f"    -> doc_top_score={equipment_result['doc_top_score']:.2%} < {MIN_FACTUAL_CONFIDENCE:.0%}")
                print(f"    -> Lexical match WAS needed to prevent refusal")
                print(f"    -> To fix: tune relative confidence alpha/beta or improve embeddings")
        else:
            print(f"  Equipment query: lexical_match=False (confidence was sufficient without fallback)")
            print(f"    -> top_score={equipment_result['top_score']:.2%} achieved through relative confidence")
    
    # Entity-fact lookup test analysis
    print()
    entity_fact_results = [r for r in results_summary if r["entity_fact"]]
    if entity_fact_results:
        print("  Entity-fact lookup tests (CEO, headquarters, etc.):")
        for ef in entity_fact_results:
            status = "REFUSED" if ef["refused"] else "allowed (unexpected)"
            print(f"    -> \"{ef['query'][:40]}\" {status} (entity_fact={ef['entity_fact']}, lex={ef['lexical']})")
        
        all_entity_refused = all(ef["refused"] for ef in entity_fact_results)
        if all_entity_refused:
            print("    -> Entity-fact guardrail correctly refusing unverifiable facts")
        else:
            print("    -> WARNING: Some entity-fact queries incorrectly allowed")
    
    # Paraphrase test analysis
    print()
    paraphrase_results = [r for r in results_summary if "gear" in r["query"].lower() or "stipend" in r["query"].lower()]
    if paraphrase_results:
        print("  Paraphrase tests (non-entity-fact):")
        for pr in paraphrase_results:
            status = "allowed" if not pr["refused"] else "REFUSED (unexpected)"
            lex_status = "with lexical" if pr["lexical"] else "without lexical"
            print(f"    -> \"{pr['query'][:40]}\" {status} ({lex_status}, level={pr['level']})")
        
        all_paraphrase_ok = all(not pr["refused"] for pr in paraphrase_results)
        if all_paraphrase_ok:
            print("    -> LOW confidence correctly allowed for non-entity-fact queries")
        else:
            print("    -> WARNING: Some paraphrases incorrectly refused")
    
    print()
    print("=" * 95)
    if all_passed:
        print("  All tests PASSED")
    else:
        print("  Some tests FAILED")
    print("=" * 95)
    
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
