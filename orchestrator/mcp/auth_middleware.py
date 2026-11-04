"""ASGI middleware: validate the Nadir MCP API key on HTTP requests."""

from __future__ import annotations

import json
import os
from typing import Any

from orchestrator.security_utils import api_keys_match, is_loopback_bind_host


def configured_mcp_api_key() -> str:
    """Return the expected MCP API key (falls back to the gateway key)."""
    return (
        os.environ.get("NADIR_MCP_API_KEY")
        or os.environ.get("NADIR_GATEWAY_API_KEY")
        or ""
    ).strip()


def extract_api_key_from_headers(headers: list[tuple[bytes, bytes]]) -> str | None:
    """Read an API key from Authorization Bearer or X-API-Key headers."""
    header_map = {name.lower(): value for name, value in headers}
    auth_header = header_map.get(b"authorization", b"").decode("utf-8").strip()
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            return token
    api_key = header_map.get(b"x-api-key", b"").decode("utf-8").strip()
    return api_key or None


def bind_host_from_scope(scope: dict[str, Any]) -> str:
    """Return the ASGI server bind host from the connection scope."""
    server = scope.get("server")
    if isinstance(server, (list, tuple)) and server:
        return str(server[0] or "")
    return ""


def api_key_is_valid(
    provided: str | None,
    expected: str,
    *,
    bind_host: str = "127.0.0.1",
) -> bool:
    """Validate the MCP API key.

    - When a key is configured: constant-time compare.
    - When no key is configured: allow only if the server bind is loopback
      (same rule as ``run_mcp``). Non-loopback binds without a key are rejected.
    """
    if expected:
        if not provided:
            return False
        return api_keys_match(provided, expected)
    return is_loopback_bind_host(bind_host)


class NadirMcpAuthMiddleware:
    """Require a valid API key before handing off to the MCP ASGI app."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        expected = configured_mcp_api_key()
        provided = extract_api_key_from_headers(scope.get("headers") or [])
        bind_host = bind_host_from_scope(scope)
        if not api_key_is_valid(provided, expected, bind_host=bind_host):
            message = (
                "Unauthorized: valid Nadir MCP API key required."
                if expected
                else "Unauthorized: NADIR_MCP_API_KEY required for non-loopback MCP binds."
            )
            body = json.dumps({"error": message}).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)
