"""ASGI middleware: validate the Nadir MCP API key on HTTP requests."""

from __future__ import annotations

import hmac
import json
import os
from typing import Any


def configured_mcp_api_key() -> str:
    """Return the expected MCP API key (empty means auth disabled on loopback only)."""
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


def api_key_is_valid(provided: str | None, expected: str) -> bool:
    """Constant-time compare when a key is configured."""
    if not expected:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


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
        if not api_key_is_valid(provided, expected):
            body = json.dumps(
                {"error": "Unauthorized: valid Nadir MCP API key required."}
            ).encode("utf-8")
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
