"""Optional API-key authentication for the Nadir gateway."""

from __future__ import annotations

import hmac

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
    if not _api_keys_match(token, expected):
        raise GatewayRouteError(
            status_code=401,
            code="invalid_api_key",
            message="Invalid API key.",
        )


def _api_keys_match(provided: str, expected: str) -> bool:
    """Constant-time compare of API key strings."""
    provided_bytes = provided.encode("utf-8")
    expected_bytes = expected.encode("utf-8")
    if len(provided_bytes) != len(expected_bytes):
        # Keep a fixed compare against expected so length mismatch is not a no-op path.
        hmac.compare_digest(expected_bytes, expected_bytes)
        return False
    return hmac.compare_digest(provided_bytes, expected_bytes)
