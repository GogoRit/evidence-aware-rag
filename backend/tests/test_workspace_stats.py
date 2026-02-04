"""Tests for workspace statistics gathering."""

import json
from pathlib import Path

import pytest

from app.modules.observability.workspace_stats import (
    count_lines_in_file,
    count_documents_in_manifest,
    get_workspace_aggregate_stats,
)


class TestCountLinesInFile:
    """Tests for count_lines_in_file function."""

    def test_counts_non_blank_lines(self, tmp_path: Path):
        """Should count only non-blank lines."""
        chunks_file = tmp_path / "chunks.jsonl"
        chunks_file.write_text(
            '{"chunk_id": "1", "content": "hello"}\n'
            '{"chunk_id": "2", "content": "world"}\n'
            '{"chunk_id": "3", "content": "test"}\n'
        )
        
        assert count_lines_in_file(chunks_file) == 3

    def test_ignores_blank_lines(self, tmp_path: Path):
        """Should ignore blank lines."""
        chunks_file = tmp_path / "chunks.jsonl"
        chunks_file.write_text(
            '{"chunk_id": "1"}\n'
            '\n'
            '{"chunk_id": "2"}\n'
            '   \n'
            '{"chunk_id": "3"}\n'
        )
        
        assert count_lines_in_file(chunks_file) == 3

    def test_returns_zero_for_missing_file(self, tmp_path: Path):
        """Should return 0 if file doesn't exist."""
        missing_file = tmp_path / "nonexistent.jsonl"
        
        assert count_lines_in_file(missing_file) == 0

    def test_returns_zero_for_empty_file(self, tmp_path: Path):
        """Should return 0 for empty file."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")
        
        assert count_lines_in_file(empty_file) == 0


class TestCountDocumentsInManifest:
    """Tests for count_documents_in_manifest function."""

    def test_counts_documents(self, tmp_path: Path):
        """Should count documents in manifest."""
        manifest_path = tmp_path / "manifest.json"
        manifest = {
            "documents": {
                "doc-1": {"filename": "a.pdf"},
                "doc-2": {"filename": "b.txt"},
            }
        }
        manifest_path.write_text(json.dumps(manifest))
        
        assert count_documents_in_manifest(manifest_path) == 2

    def test_returns_zero_for_missing_manifest(self, tmp_path: Path):
        """Should return 0 if manifest doesn't exist."""
        missing_manifest = tmp_path / "missing.json"
        
        assert count_documents_in_manifest(missing_manifest) == 0

    def test_returns_zero_for_empty_documents(self, tmp_path: Path):
        """Should return 0 if no documents in manifest."""
        manifest_path = tmp_path / "manifest.json"
        manifest = {"documents": {}}
        manifest_path.write_text(json.dumps(manifest))
        
        assert count_documents_in_manifest(manifest_path) == 0


class TestGetWorkspaceAggregateStats:
    """Tests for get_workspace_aggregate_stats function."""

    def test_aggregates_across_workspaces(self, tmp_path: Path):
        """Should aggregate stats across multiple workspaces."""
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
        
        assert stats["active_workspaces"] == 2
        assert stats["documents_ingested"] == 3
        assert stats["chunks_indexed"] == 5

    def test_ignores_hidden_directories(self, tmp_path: Path):
        """Should ignore directories starting with dot."""
        # Create hidden workspace (should be ignored)
        hidden_ws = tmp_path / ".hidden"
        hidden_ws.mkdir()
        (hidden_ws / "manifest.json").write_text(json.dumps({
            "documents": {"doc-1": {}}
        }))
        (hidden_ws / "chunks.jsonl").write_text('{"chunk_id": "1"}\n')
        
        # Create visible workspace
        visible_ws = tmp_path / "visible"
        visible_ws.mkdir()
        (visible_ws / "manifest.json").write_text(json.dumps({
            "documents": {"doc-2": {}}
        }))
        (visible_ws / "chunks.jsonl").write_text('{"chunk_id": "2"}\n')
        
        stats = get_workspace_aggregate_stats(tmp_path)
        
        assert stats["active_workspaces"] == 1
        assert stats["documents_ingested"] == 1
        assert stats["chunks_indexed"] == 1

    def test_returns_zeros_for_missing_directory(self, tmp_path: Path):
        """Should return zeros if workspaces dir doesn't exist."""
        missing_dir = tmp_path / "nonexistent"
        
        stats = get_workspace_aggregate_stats(missing_dir)
        
        assert stats["active_workspaces"] == 0
        assert stats["documents_ingested"] == 0
        assert stats["chunks_indexed"] == 0

    def test_handles_workspace_without_files(self, tmp_path: Path):
        """Should handle workspaces missing manifest or chunks files."""
        # Workspace with only manifest
        ws1 = tmp_path / "ws1"
        ws1.mkdir()
        (ws1 / "manifest.json").write_text(json.dumps({
            "documents": {"doc-1": {}}
        }))
        
        # Workspace with only chunks
        ws2 = tmp_path / "ws2"
        ws2.mkdir()
        (ws2 / "chunks.jsonl").write_text('{"chunk_id": "1"}\n')
        
        # Empty workspace
        ws3 = tmp_path / "ws3"
        ws3.mkdir()
        
        stats = get_workspace_aggregate_stats(tmp_path)
        
        assert stats["active_workspaces"] == 3
        assert stats["documents_ingested"] == 1
        assert stats["chunks_indexed"] == 1


def test_smoke_workspace_stats_integration(tmp_path: Path):
    """
    Smoke test: create a fake workspace with manifest.json and chunks.jsonl,
    call stats function, assert chunks_indexed == 3.
    """
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
    assert stats["active_workspaces"] == 1
    assert stats["documents_ingested"] == 2
    assert stats["chunks_indexed"] == 3, f"Expected 3 chunks, got {stats['chunks_indexed']}"
