#!/usr/bin/env python3
"""
Standalone smoke test for workspace_stats functions.

This script can be run without pytest or the full app dependencies.
It tests the core workspace statistics logic in isolation.

Usage:
    python tests/smoke_test_workspace_stats.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# ============================================================
# Copy of workspace_stats functions for isolated testing
# (avoids import chain that requires all app dependencies)
# ============================================================

def count_lines_in_file(file_path: Path) -> int:
    """Count non-blank lines in a file without loading it into memory."""
    if not file_path.exists():
        return 0
    
    count = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        return 0
    
    return count


def count_documents_in_manifest(manifest_path: Path) -> int:
    """Count documents registered in a manifest.json file."""
    if not manifest_path.exists():
        return 0
    
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return len(manifest.get("documents", {}))
    except Exception:
        return 0


def get_workspace_aggregate_stats(workspaces_dir: Path) -> dict:
    """Compute aggregate statistics across all workspaces."""
    stats = {
        "active_workspaces": 0,
        "documents_ingested": 0,
        "chunks_indexed": 0,
    }
    
    if not workspaces_dir.exists():
        return stats
    
    for ws in workspaces_dir.iterdir():
        if not ws.is_dir() or ws.name.startswith("."):
            continue
        
        stats["active_workspaces"] += 1
        
        # Count documents from manifest.json
        manifest_path = ws / "manifest.json"
        stats["documents_ingested"] += count_documents_in_manifest(manifest_path)
        
        # Count chunks from chunks.jsonl (line count)
        chunks_path = ws / "chunks.jsonl"
        stats["chunks_indexed"] += count_lines_in_file(chunks_path)
    
    return stats


# ============================================================
# Tests
# ============================================================

def test_count_lines_in_file():
    """Test counting non-blank lines."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Test with 3 lines
        chunks_file = tmp_path / "chunks.jsonl"
        chunks_file.write_text(
            '{"chunk_id": "1", "content": "hello"}\n'
            '{"chunk_id": "2", "content": "world"}\n'
            '{"chunk_id": "3", "content": "test"}\n'
        )
        
        result = count_lines_in_file(chunks_file)
        assert result == 3, f"Expected 3, got {result}"
        print("[OK] count_lines_in_file: counts 3 lines correctly")
        
        # Test ignoring blank lines
        chunks_file.write_text(
            '{"chunk_id": "1"}\n'
            '\n'
            '{"chunk_id": "2"}\n'
            '   \n'
            '{"chunk_id": "3"}\n'
        )
        result = count_lines_in_file(chunks_file)
        assert result == 3, f"Expected 3 (ignoring blanks), got {result}"
        print("[OK] count_lines_in_file: ignores blank lines")
        
        # Test missing file
        result = count_lines_in_file(tmp_path / "nonexistent.jsonl")
        assert result == 0, f"Expected 0 for missing file, got {result}"
        print("[OK] count_lines_in_file: returns 0 for missing file")


def test_count_documents_in_manifest():
    """Test counting documents in manifest."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        manifest_path = tmp_path / "manifest.json"
        manifest = {
            "documents": {
                "doc-1": {"filename": "a.pdf"},
                "doc-2": {"filename": "b.txt"},
            }
        }
        manifest_path.write_text(json.dumps(manifest))
        
        result = count_documents_in_manifest(manifest_path)
        assert result == 2, f"Expected 2, got {result}"
        print("[OK] count_documents_in_manifest: counts 2 documents correctly")
        
        # Test missing manifest
        result = count_documents_in_manifest(tmp_path / "missing.json")
        assert result == 0, f"Expected 0 for missing manifest, got {result}"
        print("[OK] count_documents_in_manifest: returns 0 for missing file")


def test_get_workspace_aggregate_stats():
    """Test aggregating stats across workspaces."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Create workspace 1
        ws1 = tmp_path / "workspace1"
        ws1.mkdir()
        (ws1 / "manifest.json").write_text(json.dumps({
            "documents": {"doc-1": {}, "doc-2": {}}
        }))
        (ws1 / "chunks.jsonl").write_text(
            '{"chunk_id": "1"}\n'
            '{"chunk_id": "2"}\n'
            '{"chunk_id": "3"}\n'
        )
        
        # Create workspace 2
        ws2 = tmp_path / "workspace2"
        ws2.mkdir()
        (ws2 / "manifest.json").write_text(json.dumps({
            "documents": {"doc-3": {}}
        }))
        (ws2 / "chunks.jsonl").write_text(
            '{"chunk_id": "4"}\n'
            '{"chunk_id": "5"}\n'
        )
        
        stats = get_workspace_aggregate_stats(tmp_path)
        
        assert stats["active_workspaces"] == 2, f"Expected 2 workspaces, got {stats['active_workspaces']}"
        assert stats["documents_ingested"] == 3, f"Expected 3 documents, got {stats['documents_ingested']}"
        assert stats["chunks_indexed"] == 5, f"Expected 5 chunks, got {stats['chunks_indexed']}"
        print("[OK] get_workspace_aggregate_stats: aggregates correctly across 2 workspaces")


def test_smoke_integration():
    """
    Smoke test: create a fake workspace with manifest.json and chunks.jsonl (3 lines),
    call stats function, assert chunks_indexed == 3.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Setup: Create fake workspace
        workspace = tmp_path / "test_workspace"
        workspace.mkdir()
        
        # Create manifest with 2 documents
        manifest = {
            "schema_version": "1.0",
            "workspace_id": "test_workspace",
            "documents": {
                "doc-uuid-1": {
                    "filename": "test1.pdf",
                    "chunk_ids": ["chunk-1", "chunk-2"],
                },
                "doc-uuid-2": {
                    "filename": "test2.txt",
                    "chunk_ids": ["chunk-3"],
                },
            },
            "stats": {
                "total_documents": 2,
                "total_chunks": 3,
            },
        }
        (workspace / "manifest.json").write_text(json.dumps(manifest, indent=2))
        
        # Create chunks.jsonl with exactly 3 lines
        chunks = [
            '{"chunk_id": "chunk-1", "document_id": "doc-uuid-1", "content": "First chunk content"}',
            '{"chunk_id": "chunk-2", "document_id": "doc-uuid-1", "content": "Second chunk content"}',
            '{"chunk_id": "chunk-3", "document_id": "doc-uuid-2", "content": "Third chunk content"}',
        ]
        (workspace / "chunks.jsonl").write_text("\n".join(chunks) + "\n")
        
        # Act: Call stats function
        stats = get_workspace_aggregate_stats(tmp_path)
        
        # Assert
        assert stats["active_workspaces"] == 1, f"Expected 1 workspace, got {stats['active_workspaces']}"
        assert stats["documents_ingested"] == 2, f"Expected 2 documents, got {stats['documents_ingested']}"
        assert stats["chunks_indexed"] == 3, f"Expected 3 chunks, got {stats['chunks_indexed']}"
        
        print("[OK] SMOKE TEST PASSED: chunks_indexed == 3")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Workspace Stats Smoke Test")
    print("=" * 60)
    print()
    
    try:
        test_count_lines_in_file()
        print()
        test_count_documents_in_manifest()
        print()
        test_get_workspace_aggregate_stats()
        print()
        test_smoke_integration()
        print()
        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
