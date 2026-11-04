"""Tests for Nadir Streamable HTTP MCP helpers (no DB required)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from orchestrator.mcp.auth_middleware import (
    NadirMcpAuthMiddleware,
    api_key_is_valid,
    configured_mcp_api_key,
    extract_api_key_from_headers,
)
from orchestrator.mcp.server import mcp_http_enabled, mcp_http_mount_path
from orchestrator.mcp.services import _serialize_instance, gateway_base_url
from orchestrator.models import InferenceInstance


def test_extract_bearer_and_x_api_key() -> None:
    headers = [(b"authorization", b"Bearer secret-token")]
    assert extract_api_key_from_headers(headers) == "secret-token"
    headers = [(b"x-api-key", b"alt-token")]
    assert extract_api_key_from_headers(headers) == "alt-token"


def test_api_key_is_valid_when_expected_empty() -> None:
    assert api_key_is_valid(None, "") is True
    assert api_key_is_valid("anything", "") is True


def test_api_key_is_valid_compares_digest() -> None:
    assert api_key_is_valid("abc", "abc") is True
    assert api_key_is_valid("abc", "xyz") is False
    assert api_key_is_valid(None, "xyz") is False


def test_configured_mcp_api_key_prefers_mcp_env(monkeypatch) -> None:
    monkeypatch.setenv("NADIR_MCP_API_KEY", "mcp-key")
    monkeypatch.setenv("NADIR_GATEWAY_API_KEY", "gw-key")
    assert configured_mcp_api_key() == "mcp-key"


def test_configured_mcp_api_key_falls_back_to_gateway(monkeypatch) -> None:
    monkeypatch.delenv("NADIR_MCP_API_KEY", raising=False)
    monkeypatch.setenv("NADIR_GATEWAY_API_KEY", "gw-key")
    assert configured_mcp_api_key() == "gw-key"


def test_mcp_http_enabled_respects_env(monkeypatch) -> None:
    monkeypatch.setenv("NADIR_MCP_ENABLED", "true")
    assert mcp_http_enabled() is True
    monkeypatch.setenv("NADIR_MCP_ENABLED", "false")
    assert mcp_http_enabled() is False


def test_mcp_http_mount_path_default() -> None:
    assert mcp_http_mount_path().startswith("/")


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
            await app({"type": "http", "headers": []}, AsyncMock(), send)

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
                },
                AsyncMock(),
                send,
            )

    asyncio.run(_run())
    downstream.assert_awaited_once()
