"""Shared client auth headers for calling the Nadir gateway."""

from __future__ import annotations

from orchestrator.env_utils import env_str


def gateway_api_key() -> str:
    """Return ``NADIR_GATEWAY_API_KEY`` when configured (empty when auth is off)."""
    return env_str("NADIR_GATEWAY_API_KEY", "").strip()


def gateway_auth_headers() -> dict[str, str]:
    """Authorization headers for OpenAI-compatible gateway clients."""
    key = gateway_api_key()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}
