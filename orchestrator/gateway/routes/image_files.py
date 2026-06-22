"""Serve locally stored generated images through the gateway."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from orchestrator.image_assets import resolve_image_png_path

router = APIRouter()


@router.get("/v1/images/files/{file_id}")
def serve_generated_image(file_id: str) -> FileResponse:
    """Serve a PNG generated with response_format=url (local-only, no CDN)."""
    path = resolve_image_png_path(file_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Image not found or expired.")
    return FileResponse(path, media_type="image/png", filename=f"{file_id}.png")
