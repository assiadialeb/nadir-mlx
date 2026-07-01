"""Optional API-key authentication for the Nadir gateway."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from orchestrator.env_utils import env_str
from orchestrator.security_utils import extract_bearer_token


def verify_gateway_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Require NADIR_GATEWAY_API_KEY when configured (OpenAI-compatible clients)."""
    expected = env_str("NADIR_GATEWAY_API_KEY", "")
    if not expected:
        return

    token = extract_bearer_token(authorization) or (x_api_key or "").strip()
    if token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
