"""Workspace discovery routes."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings


router = APIRouter()


class WorkspacesResponse(BaseModel):
    """Response model for workspace listing."""
    workspaces: list[str]


@router.get("", response_model=WorkspacesResponse)
async def list_workspaces() -> WorkspacesResponse:
    """
    List all available workspaces.
    
    Returns workspace IDs by listing directories under workspaces_dir
    that contain workspace data (manifest.json, chunks.jsonl, or index.faiss).
    """
    settings = get_settings()
    workspaces_dir = settings.workspaces_dir
    
    workspace_ids: list[str] = []
    
    if workspaces_dir.exists():
        for item in sorted(workspaces_dir.iterdir()):
            if item.is_dir():
                # Check if workspace has any data files
                has_manifest = (item / "manifest.json").exists()
                has_chunks = (item / "chunks.jsonl").exists()
                has_index = (item / "index.faiss").exists()
                
                if has_manifest or has_chunks or has_index:
                    workspace_ids.append(item.name)
    
    return WorkspacesResponse(workspaces=workspace_ids)
