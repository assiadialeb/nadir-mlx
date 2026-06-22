"""Optional API-key authentication for local MLX inference worker processes."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status

from orchestrator.security_utils import extract_bearer_token


def require_inference_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Require NADIR_INFERENCE_API_KEY when configured."""
    expected = os.environ.get("NADIR_INFERENCE_API_KEY", "").strip()
    if not expected:
        return

    token = extract_bearer_token(authorization) or (x_api_key or "").strip()
    if token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
