"""Workspace statistics gathering for metrics.

Computes aggregate stats across all workspaces without loading
large files into memory.
"""

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


def count_lines_in_file(file_path: Path) -> int:
    """
    Count non-blank lines in a file without loading it into memory.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Number of non-blank lines
    """
    if not file_path.exists():
        return 0
    
    count = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception as e:
        logger.warning("Failed to count lines in file", path=str(file_path), error=str(e))
        return 0
    
    return count


def count_documents_in_manifest(manifest_path: Path) -> int:
    """
    Count documents registered in a manifest.json file.
    
    Args:
        manifest_path: Path to manifest.json
    
    Returns:
        Number of documents in the manifest
    """
    if not manifest_path.exists():
        return 0
    
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return len(manifest.get("documents", {}))
    except Exception as e:
        logger.warning("Failed to read manifest", path=str(manifest_path), error=str(e))
        return 0


def get_workspace_aggregate_stats(workspaces_dir: Path) -> dict[str, Any]:
    """
    Compute aggregate statistics across all workspaces.
    
    Stats computed:
    - active_workspaces: Number of workspace directories
    - documents_ingested: Total documents in manifest.json across workspaces
    - chunks_indexed: Total lines in chunks.jsonl across workspaces
    
    Args:
        workspaces_dir: Path to the workspaces directory
    
    Returns:
        Dictionary with aggregate stats
    """
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
    
    logger.debug(
        "Computed workspace aggregate stats",
        workspaces=stats["active_workspaces"],
        documents=stats["documents_ingested"],
        chunks=stats["chunks_indexed"],
    )
    
    return stats
