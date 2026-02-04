#!/usr/bin/env python3
"""
Test mode-aware confidence and refusal logic.

Expected results in retrieval-only mode (GENERATION_ENABLED=false):
1. "What equipment does the company provide?" -> refused=False
2. "Who is the CEO of ACME Corporation?" -> refused=True  
3. "What are the core working hours for remote employees?" -> refused=False
"""

import sys
import httpx

BASE_URL = "http://localhost:8000"
WORKSPACE = "demo"

TEST_QUERIES = [
    {
        "query": "What equipment does the company provide?",
        "expected_refused": False,
        "reason": "Policy clearly mentions laptop, monitor, peripherals",
    },
    {
        "query": "Who is the CEO of ACME Corporation?",
        "expected_refused": True,
        "reason": "CEO is not mentioned in the policy document",
    },
    {
        "query": "What are the core working hours for remote employees?",
        "expected_refused": False,
        "reason": "Policy specifies 10:00 AM to 3:00 PM core hours",
    },
]


def run_tests():
    """Run confidence/refusal tests."""
    print("=" * 60)
    print("  Mode-Aware Confidence Test Suite")
    print("=" * 60)
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
    
    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        expected_refused = test["expected_refused"]
        reason = test["reason"]
        
        print(f"Test {i}: {query}")
        print(f"  Expected: refused={expected_refused}")
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
            level = confidence.get("level", "unknown")
            retrieval_only = data.get("retrieval_only", False)
            sources = data.get("sources", [])
            
            # Check result
            if actual_refused == expected_refused:
                status = "[PASS]"
            else:
                status = "[FAIL]"
                all_passed = False
            
            print(f"  {status} refused={actual_refused} (expected {expected_refused})")
            print(f"         confidence: {level} (top_score={top_score:.2%})")
            print(f"         retrieval_only: {retrieval_only}")
            print(f"         sources: {len(sources)}")
            
            # Show top source snippet if available
            if sources and not actual_refused:
                top_source = sources[0]
                snippet = top_source.get("content", "")[:80]
                print(f"         top_snippet: \"{snippet}...\"")
            
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            all_passed = False
        
        print()
    
    print("=" * 60)
    if all_passed:
        print("  All tests PASSED")
    else:
        print("  Some tests FAILED")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
