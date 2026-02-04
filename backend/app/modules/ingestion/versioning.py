"""Document versioning for tracking changes."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import structlog

logger = structlog.get_logger()


async def get_version_manifest(workspace_path: Path) -> dict[str, Any]:
    """Load or create version manifest for a workspace."""
    manifest_path = workspace_path / "manifest.json"
    
    if manifest_path.exists():
        async with aiofiles.open(manifest_path, "r") as f:
            content = await f.read()
            return json.loads(content)
    
    return {
        "workspace_id": workspace_path.name,
        "created_at": datetime.utcnow().isoformat(),
        "documents": {},
    }


async def save_version_manifest(workspace_path: Path, manifest: dict[str, Any]) -> None:
    """Save version manifest to workspace."""
    manifest_path = workspace_path / "manifest.json"
    manifest["updated_at"] = datetime.utcnow().isoformat()
    
    async with aiofiles.open(manifest_path, "w") as f:
        await f.write(json.dumps(manifest, indent=2))


async def create_version(
    workspace_path: Path,
    filename: str,
    checksum: str,
) -> int:
    """
    Create a new version for a document.
    
    If a document with the same filename and checksum exists,
    returns the existing version number.
    
    Args:
        workspace_path: Path to workspace directory
        filename: Original filename
        checksum: SHA-256 checksum of file content
    
    Returns:
        Version number for this document
    """
    manifest = await get_version_manifest(workspace_path)
    
    # Check if this exact file already exists
    if filename in manifest["documents"]:
        doc_info = manifest["documents"][filename]
        versions = doc_info.get("versions", [])
        
        # Check if checksum matches any existing version
        for v in versions:
            if v.get("checksum") == checksum:
                logger.info(
                    "Document unchanged, reusing version",
                    filename=filename,
                    version=v["version"],
                )
                return v["version"]
        
        # New version of existing document
        new_version = len(versions) + 1
    else:
        # First version of new document
        manifest["documents"][filename] = {"versions": []}
        new_version = 1
    
    # Record new version
    version_info = {
        "version": new_version,
        "checksum": checksum,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    manifest["documents"][filename]["versions"].append(version_info)
    manifest["documents"][filename]["current_version"] = new_version
    
    await save_version_manifest(workspace_path, manifest)
    
    logger.info(
        "Created document version",
        filename=filename,
        version=new_version,
    )
    
    return new_version


async def get_document_versions(
    workspace_path: Path,
    filename: str,
) -> list[dict[str, Any]]:
    """Get all versions of a document."""
    manifest = await get_version_manifest(workspace_path)
    
    if filename not in manifest["documents"]:
        return []
    
    return manifest["documents"][filename].get("versions", [])


async def get_current_version(
    workspace_path: Path,
    filename: str,
) -> int | None:
    """Get current version number for a document."""
    manifest = await get_version_manifest(workspace_path)
    
    if filename not in manifest["documents"]:
        return None
    
    return manifest["documents"][filename].get("current_version")


async def rollback_version(
    workspace_path: Path,
    filename: str,
    target_version: int,
) -> bool:
    """
    Rollback a document to a previous version.
    
    Note: This only updates the manifest pointer.
    Actual file rollback requires additional implementation.
    """
    manifest = await get_version_manifest(workspace_path)
    
    if filename not in manifest["documents"]:
        logger.error("Document not found for rollback", filename=filename)
        return False
    
    versions = manifest["documents"][filename].get("versions", [])
    version_numbers = [v["version"] for v in versions]
    
    if target_version not in version_numbers:
        logger.error(
            "Target version not found",
            filename=filename,
            target_version=target_version,
            available_versions=version_numbers,
        )
        return False
    
    manifest["documents"][filename]["current_version"] = target_version
    await save_version_manifest(workspace_path, manifest)
    
    logger.info(
        "Rolled back document version",
        filename=filename,
        target_version=target_version,
    )
    
    return True
