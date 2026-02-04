"""Document ingestion module."""

from app.modules.ingestion.loaders import load_document
from app.modules.ingestion.chunking import chunk_document
from app.modules.ingestion.versioning import create_version, get_document_versions
from app.modules.ingestion.storage import (
    load_manifest,
    save_manifest,
    register_document,
    unregister_document,
    append_chunks,
    load_all_chunks,
    load_chunks_by_ids,
    rebuild_chunks_file,
    load_index_meta,
    save_index_meta,
    update_index_meta,
    get_workspace_stats,
    get_active_chunk_ids,
)

__all__ = [
    "load_document",
    "chunk_document",
    "create_version",
    "get_document_versions",
    # Storage
    "load_manifest",
    "save_manifest",
    "register_document",
    "unregister_document",
    "append_chunks",
    "load_all_chunks",
    "load_chunks_by_ids",
    "rebuild_chunks_file",
    "load_index_meta",
    "save_index_meta",
    "update_index_meta",
    "get_workspace_stats",
    "get_active_chunk_ids",
]
