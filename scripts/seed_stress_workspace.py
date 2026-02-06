#!/usr/bin/env python3
"""
Seed stress-test workspace with test documents.

This script ingests three test documents into the 'stress-test' workspace:
- sparse_policy.txt: Very short document (1-2 chunks)
- redundant_policy.txt: Document with repeated content (many near-duplicate chunks)
- conflicting_policy.txt: Document with contradictory statements

Usage:
    python scripts/seed_stress_workspace.py [--workspace WORKSPACE] [--base-url URL]

Requirements:
    - Backend server running on localhost:8000 (or --base-url)
    - Test documents in backend/app/data/stress_docs/
"""

import argparse
import sys
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
DEFAULT_WORKSPACE = "stress-test"

# Path to stress test documents (relative to project root)
STRESS_DOCS_DIR = Path(__file__).parent.parent / "backend" / "app" / "data" / "stress_docs"

STRESS_DOCUMENTS = [
    "sparse_policy.txt",
    "redundant_policy.txt",
    "conflicting_policy.txt",
    "long_doc.txt",
    "needle_in_haystack.txt",
]


def check_backend_health(base_url: str) -> bool:
    """Check if the backend server is running and healthy."""
    try:
        response = httpx.get(f"{base_url}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def ingest_document(
    base_url: str,
    workspace_id: str,
    file_path: Path,
) -> dict | None:
    """
    Ingest a document via the ingestion API.
    
    Args:
        base_url: Base URL of the backend API
        workspace_id: Target workspace ID
        file_path: Path to the document file
        
    Returns:
        Response JSON on success, None on failure
    """
    if not file_path.exists():
        print(f"  ERROR: File not found: {file_path}")
        return None
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "text/plain")}
            data = {"workspace_id": workspace_id}
            
            response = httpx.post(
                f"{base_url}/ingest",
                files=files,
                data=data,
                timeout=60.0,
            )
            
            if response.status_code == 201:
                return response.json()
            else:
                print(f"  ERROR: Ingestion failed: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"  ERROR: Exception during ingestion: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Seed stress-test workspace with test documents"
    )
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        help=f"Workspace to seed (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help=f"Backend API base URL (default: {BASE_URL})",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("  Stress Test Workspace Seeding")
    print("=" * 80)
    
    # Check backend health
    if not check_backend_health(args.base_url):
        print("\nERROR: Backend is not running or not healthy")
        print(f"Please start the backend with: make dev")
        print(f"Or specify a different base URL with: --base-url <URL>")
        sys.exit(1)
    
    print(f"\nBackend: Online ({args.base_url})")
    print(f"Workspace: {args.workspace}")
    print(f"Documents directory: {STRESS_DOCS_DIR}")
    
    # Check documents directory exists
    if not STRESS_DOCS_DIR.exists():
        print(f"\nERROR: Documents directory not found: {STRESS_DOCS_DIR}")
        print("Please ensure stress test documents are in the correct location.")
        sys.exit(1)
    
    print(f"\nIngesting {len(STRESS_DOCUMENTS)} documents...\n")
    
    # Ingest each document
    results = []
    for doc_name in STRESS_DOCUMENTS:
        doc_path = STRESS_DOCS_DIR / doc_name
        print(f"  [{doc_name}]", end=" ", flush=True)
        
        result = ingest_document(args.base_url, args.workspace, doc_path)
        if result:
            chunks = result.get("chunks_created", 0)
            doc_id = result.get("document_id", "unknown")[:8]
            print(f"✓ Ingested ({chunks} chunks, doc_id: {doc_id})")
            results.append((doc_name, True, chunks))
        else:
            print("✗ Failed")
            results.append((doc_name, False, 0))
    
    # Summary
    print(f"\n{'=' * 80}")
    print("  Summary")
    print(f"{'=' * 80}\n")
    
    success_count = sum(1 for _, success, _ in results if success)
    total_chunks = sum(chunks for _, _, chunks in results)
    
    for doc_name, success, chunks in results:
        status = "✓" if success else "✗"
        print(f"  {status} {doc_name:<30} {chunks} chunks")
    
    print(f"\n  Total: {success_count}/{len(STRESS_DOCUMENTS)} documents ingested")
    print(f"  Total chunks: {total_chunks}")
    
    if success_count == len(STRESS_DOCUMENTS):
        print(f"\n  ✓ Workspace '{args.workspace}' is ready for stress testing")
        print(f"\n  Next steps:")
        print(f"    1. Run: python scripts/evaluate_retrieval_stress.py --workspace {args.workspace}")
        print(f"    2. Or use the workspace in the Streamlit UI")
    else:
        print(f"\n  ✗ Some documents failed to ingest. Please check errors above.")
        sys.exit(1)
    
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
