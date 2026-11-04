"""Tests for Nadir Streamable HTTP MCP helpers (no DB required)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from mcp.server.transport_security import TransportSecurityMiddleware
from starlette.requests import Request

from orchestrator.mcp.auth_middleware import (
    NadirMcpAuthMiddleware,
    api_key_is_valid,
    configured_mcp_api_key,
    extract_api_key_from_headers,
)
from orchestrator.mcp.server import (
    _mcp_transport_security,
    mcp_allowed_hosts,
    mcp_http_enabled,
    mcp_http_mount_path,
)
from orchestrator.mcp.services import _serialize_instance, gateway_base_url
from orchestrator.models import InferenceInstance


def test_extract_bearer_and_x_api_key() -> None:
    headers = [(b"authorization", b"Bearer secret-token")]
    assert extract_api_key_from_headers(headers) == "secret-token"
    headers = [(b"x-api-key", b"alt-token")]
    assert extract_api_key_from_headers(headers) == "alt-token"


def test_api_key_is_valid_when_expected_empty_requires_loopback() -> None:
    assert api_key_is_valid(None, "", bind_host="127.0.0.1") is True
    assert api_key_is_valid("anything", "", bind_host="localhost") is True
    assert api_key_is_valid(None, "", bind_host="0.0.0.0") is False
    assert api_key_is_valid(None, "", bind_host="192.168.1.10") is False


def test_api_key_is_valid_compares_digest() -> None:
    assert api_key_is_valid("abc", "abc", bind_host="0.0.0.0") is True
    assert api_key_is_valid("abc", "xyz", bind_host="0.0.0.0") is False
    assert api_key_is_valid(None, "xyz", bind_host="0.0.0.0") is False


def test_configured_mcp_api_key_prefers_mcp_env(monkeypatch) -> None:
    monkeypatch.setenv("NADIR_MCP_API_KEY", "mcp-key")
    monkeypatch.setenv("NADIR_GATEWAY_API_KEY", "gw-key")
    assert configured_mcp_api_key() == "mcp-key"


def test_configured_mcp_api_key_falls_back_to_gateway(monkeypatch) -> None:
    monkeypatch.delenv("NADIR_MCP_API_KEY", raising=False)
    monkeypatch.setenv("NADIR_GATEWAY_API_KEY", "gw-key")
    assert configured_mcp_api_key() == "gw-key"


def test_mcp_http_enabled_defaults_false(monkeypatch) -> None:
    monkeypatch.delenv("NADIR_MCP_ENABLED", raising=False)
    assert mcp_http_enabled() is False
    monkeypatch.setenv("NADIR_MCP_ENABLED", "true")
    assert mcp_http_enabled() is True


def test_mcp_http_mount_path_default() -> None:
    assert mcp_http_mount_path().startswith("/")


def test_mcp_allowed_hosts_includes_loopback(monkeypatch) -> None:
    monkeypatch.delenv("NADIR_MCP_ALLOWED_HOSTS", raising=False)
    hosts = mcp_allowed_hosts()
    assert "127.0.0.1:*" in hosts
    assert "localhost:*" in hosts


def test_gateway_base_url_from_env(monkeypatch) -> None:
    monkeypatch.setenv("NADIR_GATEWAY_HOST", "127.0.0.1")
    monkeypatch.setenv("NADIR_GATEWAY_PORT", "11380")
    assert gateway_base_url() == "http://127.0.0.1:11380"


def test_serialize_instance_includes_aliases() -> None:
    instance = InferenceInstance(
        id=1,
        model_name="demo-model",
        port=11401,
        launch_mode="TEXT",
        status="RUNNING",
        health_status="HEALTHY",
        server_config={"model_id": "primary", "gateway_aliases": ["alt"]},
    )
    payload = _serialize_instance(instance)
    assert payload["gateway_alias"] == "primary"
    assert "alt" in payload["gateway_aliases"]
    assert payload["port"] == 11401


def test_auth_middleware_rejects_missing_key() -> None:
    downstream = AsyncMock()
    app = NadirMcpAuthMiddleware(downstream)
    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    async def _run() -> None:
        with patch(
            "orchestrator.mcp.auth_middleware.configured_mcp_api_key",
            return_value="required",
        ):
            await app(
                {"type": "http", "headers": [], "server": ("127.0.0.1", 11381)},
                AsyncMock(),
                send,
            )

    asyncio.run(_run())
    downstream.assert_not_called()
    assert sent[0]["status"] == 401


def test_auth_middleware_rejects_empty_key_on_lan_bind() -> None:
    downstream = AsyncMock()
    app = NadirMcpAuthMiddleware(downstream)
    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    async def _run() -> None:
        with patch(
            "orchestrator.mcp.auth_middleware.configured_mcp_api_key",
            return_value="",
        ):
            await app(
                {"type": "http", "headers": [], "server": ("0.0.0.0", 8000)},
                AsyncMock(),
                send,
            )

    asyncio.run(_run())
    downstream.assert_not_called()
    assert sent[0]["status"] == 401


def test_auth_middleware_accepts_valid_bearer() -> None:
    downstream = AsyncMock()
    app = NadirMcpAuthMiddleware(downstream)

    async def send(_message: dict) -> None:
        return None

    async def _run() -> None:
        with patch(
            "orchestrator.mcp.auth_middleware.configured_mcp_api_key",
            return_value="required",
        ):
            await app(
                {
                    "type": "http",
                    "headers": [(b"authorization", b"Bearer required")],
                    "server": ("0.0.0.0", 8000),
                },
                AsyncMock(),
                send,
            )

    asyncio.run(_run())
    downstream.assert_awaited_once()


def _security_request(*, host: str, origin: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = [(b"host", host.encode("utf-8"))]
    if origin is not None:
        headers.append((b"origin", origin.encode("utf-8")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 11381),
    }
    return Request(scope)


def test_transport_security_rejects_foreign_host(monkeypatch) -> None:
    monkeypatch.delenv("NADIR_MCP_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("NADIR_MCP_CORS_ORIGINS", raising=False)
    middleware = TransportSecurityMiddleware(_mcp_transport_security())

    async def _run():
        return await middleware.validate_request(
            _security_request(host="evil.example"),
            is_post=False,
        )

    response = asyncio.run(_run())
    assert response is not None
    assert response.status_code == 421


def test_transport_security_rejects_foreign_origin(monkeypatch) -> None:
    monkeypatch.delenv("NADIR_MCP_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("NADIR_MCP_CORS_ORIGINS", raising=False)
    middleware = TransportSecurityMiddleware(_mcp_transport_security())

    async def _run():
        return await middleware.validate_request(
            _security_request(host="127.0.0.1:11381", origin="http://evil.example"),
            is_post=False,
        )

    response = asyncio.run(_run())
    assert response is not None
    assert response.status_code == 403


def test_transport_security_allows_loopback_without_origin(monkeypatch) -> None:
    monkeypatch.delenv("NADIR_MCP_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("NADIR_MCP_CORS_ORIGINS", raising=False)
    middleware = TransportSecurityMiddleware(_mcp_transport_security())

    async def _run():
        return await middleware.validate_request(
            _security_request(host="127.0.0.1:11381"),
            is_post=False,
        )

    response = asyncio.run(_run())
    assert response is None
