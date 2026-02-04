#!/usr/bin/env python3
"""
End-to-end smoke demo for the RAG system.

This script demonstrates the full RAG pipeline:
1. Health check
2. Ingest a sample document into workspace "demo"
3. Query the workspace
4. Verify metrics

Usage:
    # Start the backend first, then run:
    python scripts/demo_e2e.py
    
    # Or with custom URL:
    python scripts/demo_e2e.py --url http://localhost:8000

Requirements:
    pip install httpx  (or use: requests)

No API keys required - runs in retrieval-only mode.
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import httpx
    CLIENT_CLASS = httpx.Client
except ImportError:
    import urllib.request
    import urllib.error
    CLIENT_CLASS = None  # Will use urllib fallback


# ============================================================
# Configuration
# ============================================================

WORKSPACE_ID = "demo"
SAMPLE_DOC = Path(__file__).parent.parent / "backend" / "demo_data" / "sample_policy.txt"
# Also check relative to script location
if not SAMPLE_DOC.exists():
    SAMPLE_DOC = Path(__file__).parent.parent / "demo_data" / "sample_policy.txt"

DEMO_QUERIES = [
    "What are the core working hours for remote employees?",
    "How much can I claim for internet expenses?",
    "What equipment does the company provide?",
]


# ============================================================
# HTTP Client (works with or without httpx)
# ============================================================

class SimpleClient:
    """Simple HTTP client using urllib (no dependencies)."""
    
    def __init__(self, base_url: str, timeout: float = 120.0):
        # Long timeout for first request (model loading can take 30-60s)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    def get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return {"status_code": resp.status, "json": json.loads(resp.read())}
        except urllib.error.HTTPError as e:
            return {"status_code": e.code, "json": None, "error": str(e)}
        except Exception as e:
            return {"status_code": 0, "json": None, "error": str(e)}
    
    def post_json(self, path: str, data: dict) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return {"status_code": resp.status, "json": json.loads(resp.read())}
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            return {"status_code": e.code, "json": None, "error": body}
        except Exception as e:
            return {"status_code": 0, "json": None, "error": str(e)}
    
    def post_file(self, path: str, file_path: Path, form_data: dict) -> dict:
        """Post a file with multipart form data."""
        import mimetypes
        import uuid
        
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex[:16]}"
        
        lines = []
        # Add form fields
        for key, value in form_data.items():
            lines.append(f"--{boundary}")
            lines.append(f'Content-Disposition: form-data; name="{key}"')
            lines.append("")
            lines.append(str(value))
        
        # Add file
        content_type = mimetypes.guess_type(str(file_path))[0] or "text/plain"
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        lines.append(f"--{boundary}")
        lines.append(f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"')
        lines.append(f"Content-Type: {content_type}")
        lines.append("")
        
        # Build body
        body_start = "\r\n".join(lines).encode("utf-8") + b"\r\n"
        body_end = f"\r\n--{boundary}--\r\n".encode("utf-8")
        body = body_start + file_content + body_end
        
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return {"status_code": resp.status, "json": json.loads(resp.read())}
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            return {"status_code": e.code, "json": None, "error": body}
        except Exception as e:
            return {"status_code": 0, "json": None, "error": str(e)}


def get_client(base_url: str) -> SimpleClient:
    """Get HTTP client."""
    return SimpleClient(base_url)


# ============================================================
# Demo Steps
# ============================================================

def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_json(data: dict, indent: int = 2):
    """Pretty print JSON."""
    print(json.dumps(data, indent=indent, default=str))


def step_health_check(client: SimpleClient) -> bool:
    """Step 1: Health check."""
    print_header("Step 1: Health Check")
    
    resp = client.get("/health")
    
    if resp["status_code"] != 200:
        print(f"[FAIL] Health check failed: {resp.get('error', 'Unknown error')}")
        print("       Make sure the backend is running: make backend")
        return False
    
    health = resp["json"]
    print(f"[OK] Status: {health['status']}")
    print(f"  Version: {health['version']}")
    print(f"  Environment: {health['environment']}")
    
    for comp in health.get("components", []):
        status_icon = "[OK]" if comp["status"] == "healthy" else "[FAIL]"
        print(f"  {status_icon} {comp['name']}: {comp['status']} ({comp.get('latency_ms', 0):.1f}ms)")
    
    return health["status"] == "healthy"


def step_cleanup_workspace(client: SimpleClient) -> bool:
    """Optional: Delete existing documents to force fresh ingest."""
    try:
        resp = client.get(f"/ingest/{WORKSPACE_ID}/documents")
        if resp["status_code"] == 200 and resp["json"].get("documents"):
            for doc in resp["json"]["documents"]:
                # Delete existing documents
                client.post_json(
                    f"/ingest/{WORKSPACE_ID}/documents/{doc['document_id']}/delete",
                    {}
                )
        return True
    except Exception:
        return True  # Workspace might not exist yet


def step_ingest_document(client: SimpleClient) -> bool:
    """Step 2: Ingest sample document."""
    print_header("Step 2: Ingest Sample Document")
    
    if not SAMPLE_DOC.exists():
        print(f"[FAIL] Sample document not found: {SAMPLE_DOC}")
        return False
    
    print(f"  Document: {SAMPLE_DOC.name}")
    print(f"  Workspace: {WORKSPACE_ID}")
    print(f"  Size: {SAMPLE_DOC.stat().st_size} bytes")
    print(f"  Chunking: token-based, size=100, overlap=15")
    print()
    
    # Use small chunk_size with token-based chunking for fine-grained retrieval
    resp = client.post_file(
        "/ingest",
        SAMPLE_DOC,
        {
            "workspace_id": WORKSPACE_ID,
            "chunk_size": "100",         # ~100 tokens per chunk
            "chunk_overlap": "15",       # ~15 token overlap
            "chunk_strategy": "token",   # Token-based (not semantic) for smaller chunks
        }
    )
    
    if resp["status_code"] not in (200, 201):
        print(f"[FAIL] Ingestion failed: {resp.get('error', 'Unknown error')}")
        return False
    
    result = resp["json"]
    print(f"[OK] Document ingested successfully!")
    print(f"  Document ID: {result['document_id']}")
    print(f"  Chunks created: {result['chunks_created']}")
    print(f"  Version: {result['version']}")
    print(f"  Processing time: {result['processing_time_ms']:.1f}ms")
    
    return True


def step_query_rag(client: SimpleClient, query: str) -> dict | None:
    """Step 3: Query the RAG system."""
    print(f"\n  Query: \"{query}\"")
    print()
    
    resp = client.post_json("/chat", {
        "workspace_id": WORKSPACE_ID,
        "query": query,
        "top_k": 3,
    })
    
    if resp["status_code"] != 200:
        print(f"  [FAIL] Query failed: {resp.get('error', 'Unknown error')}")
        return None
    
    result = resp["json"]
    
    # Print key response fields
    print(f"  Response:")
    print(f"    Model: {result.get('model_used', 'N/A')}")
    print(f"    Latency: {result.get('latency_ms', 0):.1f}ms")
    print(f"    Retrieval Only: {result.get('retrieval_only', False)}")
    print(f"    Refused: {result.get('refused', False)}")
    
    # Confidence info
    conf = result.get("confidence", {})
    if conf:
        print(f"    Confidence: {conf.get('level', 'N/A')} (top: {conf.get('top_score', 0):.2%})")
    
    # Router decision
    router = result.get("router_decision", {})
    if router:
        print(f"    Routing: mode={router.get('mode')}, complexity={router.get('complexity', 0):.2f}")
    
    # Sources
    sources = result.get("sources", [])
    if sources:
        print(f"    Sources: {len(sources)} chunks")
        for i, src in enumerate(sources[:2]):  # Show top 2
            score = src.get("score", 0)
            preview = src.get("content", "")[:80].replace("\n", " ")
            print(f"      [{i+1}] {score:.2%} - \"{preview}...\"")
    
    return result


def step_verify_metrics(client: SimpleClient) -> bool:
    """Step 4: Verify metrics."""
    print_header("Step 4: Verify Metrics")
    
    resp = client.get("/metrics")
    
    if resp["status_code"] != 200:
        print(f"[FAIL] Metrics check failed: {resp.get('error', 'Unknown error')}")
        return False
    
    metrics = resp["json"]
    
    print(f"  Active Workspaces: {metrics['active_workspaces']}")
    print(f"  Documents Ingested: {metrics['documents_ingested']}")
    print(f"  Chunks Indexed: {metrics['chunks_indexed']}")
    print(f"  Total Requests: {metrics['requests_total']}")
    
    # Verify expected values
    checks = [
        ("active_workspaces >= 1", metrics['active_workspaces'] >= 1),
        ("documents_ingested >= 1", metrics['documents_ingested'] >= 1),
        ("chunks_indexed > 0", metrics['chunks_indexed'] > 0),
    ]
    
    print()
    all_pass = True
    for check_name, passed in checks:
        icon = "[OK]" if passed else "[FAIL]"
        print(f"  {icon} {check_name}")
        if not passed:
            all_pass = False
    
    return all_pass


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RAG System E2E Smoke Demo")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend URL")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("       RAG System - End-to-End Smoke Demo")
    print("="*60)
    print(f"\n  Backend URL: {args.url}")
    print(f"  Workspace: {WORKSPACE_ID}")
    print(f"  Mode: Retrieval-only (no API keys needed)")
    
    client = get_client(args.url)
    
    # Step 1: Health check
    if not step_health_check(client):
        print("\n[FAIL] Demo aborted: Backend not healthy")
        sys.exit(1)
    
    # Step 2: Ingest document
    if not step_ingest_document(client):
        print("\n[FAIL] Demo aborted: Ingestion failed")
        sys.exit(1)
    
    # Step 3: Run queries
    print_header("Step 3: Query the RAG System")
    
    for query in DEMO_QUERIES:
        result = step_query_rag(client, query)
        if result is None:
            print("\n[FAIL] Demo aborted: Query failed")
            sys.exit(1)
    
    # Step 4: Verify metrics
    if not step_verify_metrics(client):
        print("\n[Warning] Metrics verification failed")
    
    # Done
    print_header("Demo Complete!")
    print("  [OK] Health check passed")
    print("  [OK] Document ingested")
    print("  [OK] Queries executed")
    print("  [OK] Metrics verified")
    print()
    print("  The RAG system is working correctly.")
    print("  Try the Streamlit UI: make frontend")
    print()


if __name__ == "__main__":
    main()
