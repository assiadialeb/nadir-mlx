"""Optional API-key authentication for the Nadir gateway."""

from __future__ import annotations

from fastapi import Header

from orchestrator.env_utils import env_str
from orchestrator.gateway.router import GatewayRouteError
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
        raise GatewayRouteError(
            status_code=401,
            code="invalid_api_key",
            message="Invalid API key.",
        )
