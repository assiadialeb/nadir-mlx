"""Security hardening tests for the Nadir gateway (S1–S4)."""

from __future__ import annotations

import os
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from django.test import SimpleTestCase
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from orchestrator.gateway.app import create_app
from orchestrator.gateway.config import assert_gateway_bind_safe, load_gateway_config
from orchestrator.gateway.middleware import install_gateway_middleware
from orchestrator.gateway.routes.common import parse_json_body


class GatewaySecurityMiddlewareTests(SimpleTestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def test_trusted_host_rejects_evil_host(self) -> None:
        response = self.client.get("/health", headers={"host": "evil.example"})
        self.assertEqual(response.status_code, 400)

    def test_origin_guard_rejects_unknown_origin(self) -> None:
        response = self.client.get(
            "/health",
            headers={"origin": "https://evil.example"},
        )
        self.assertEqual(response.status_code, 403)

    def test_cli_without_origin_is_allowed(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_text_plain_body_returns_415(self) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            content=b'{"model":"demo","messages":[{"role":"user","content":"hi"}]}',
            headers={"content-type": "text/plain"},
        )
        self.assertEqual(response.status_code, 415)

    def test_allowed_origin_passes_guard(self) -> None:
        app = FastAPI()
        install_gateway_middleware(app, cors_origins=["https://app.example"])

        @app.get("/ping")
        def ping() -> dict[str, str]:
            return {"ok": "1"}

        client = TestClient(app)
        response = client.get(
            "/ping",
            headers={"origin": "https://app.example", "host": "testserver"},
        )
        self.assertEqual(response.status_code, 200)


class ParseJsonBodyContentTypeTests(IsolatedAsyncioTestCase):
    async def _parse(self, content_type: str | None, body: bytes = b'{"a":1}'):
        headers: list[tuple[bytes, bytes]] = []
        if content_type is not None:
            headers.append((b"content-type", content_type.encode()))
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": headers,
        }

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        return await parse_json_body(Request(scope, receive))

    async def test_missing_content_type_ok(self) -> None:
        result = await self._parse(None)
        self.assertEqual(result, {"a": 1})

    async def test_text_plain_rejected(self) -> None:
        result = await self._parse("text/plain")
        self.assertEqual(result.status_code, 415)


class GatewayBindGuardTests(SimpleTestCase):
    def test_loopback_bind_allowed_without_key(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NADIR_GATEWAY_API_KEY", None)
            assert_gateway_bind_safe("127.0.0.1")

    def test_lan_bind_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NADIR_GATEWAY_API_KEY", None)
            with self.assertRaises(SystemExit):
                assert_gateway_bind_safe("192.168.1.10")

    def test_lan_bind_allowed_with_api_key(self) -> None:
        with patch.dict(os.environ, {"NADIR_GATEWAY_API_KEY": "secret"}, clear=False):
            assert_gateway_bind_safe("192.168.1.10")

    @patch.dict(
        os.environ,
        {"NADIR_GATEWAY_HOST": "127.0.0.1", "NADIR_GATEWAY_PORT": "11380"},
        clear=False,
    )
    def test_load_gateway_config_accepts_loopback(self) -> None:
        config = load_gateway_config()
        self.assertEqual(config.host, "127.0.0.1")
