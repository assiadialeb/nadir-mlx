"""Tests for Anthropic Messages gateway façade."""

from __future__ import annotations

from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from django.test import SimpleTestCase
from fastapi.testclient import TestClient

from orchestrator.gateway.app import create_app
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, GatewayTarget
from orchestrator.gateway.services.anthropic_proxy import _translate_chat_sse_to_anthropic

TEXT_TARGET = GatewayTarget(
    alias="demo-chat",
    instance_id=1,
    launch_mode="TEXT",
    host="127.0.0.1",
    port=11400,
    upstream_model="default_model",
    api_path=CHAT_COMPLETIONS_PATH,
)


class GatewayAnthropicProxyTests(SimpleTestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())
        self._touch_patcher = patch(
            "orchestrator.lifecycle_services.touch_instance_last_used_at",
        )
        self._touch_patcher.start()

    def tearDown(self) -> None:
        self._touch_patcher.stop()

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    def test_messages_non_stream_translates(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        upstream = httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            headers={"content-type": "application/json"},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=upstream)
        mock_get_client.return_value = mock_client

        response = self.client.post(
            "/v1/messages",
            json={
                "model": "demo-chat",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["type"], "message")
        self.assertEqual(payload["content"][0]["text"], "Hello")
        self.assertEqual(payload["model"], "demo-chat")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    def test_messages_stream_translates_sse(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        async def chunk_stream():
            yield (
                b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
                b"data: [DONE]\n\n"
            )

        upstream = MagicMock()
        upstream.status_code = 200
        upstream.aiter_bytes = chunk_stream
        upstream.aclose = AsyncMock()

        mock_client = AsyncMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=upstream)
        mock_get_client.return_value = mock_client

        response = self.client.post(
            "/v1/messages",
            json={
                "model": "demo-chat",
                "max_tokens": 8,
                "stream": True,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"event: message_start", response.content)
        self.assertIn(b"content_block_delta", response.content)
        self.assertIn(b"text_delta", response.content)
        self.assertIn(b"event: message_stop", response.content)
        self.assertNotIn(b"not_implemented", response.content)

    def test_messages_invalid_json_returns_400(self) -> None:
        response = self.client.post(
            "/v1/messages",
            content=b"{bad",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_request")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    def test_messages_upstream_error_passthrough(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        upstream = httpx.Response(
            400,
            json={"error": {"message": "bad", "type": "invalid_request_error"}},
            headers={"content-type": "application/json"},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=upstream)
        mock_get_client.return_value = mock_client

        response = self.client.post(
            "/v1/messages",
            json={
                "model": "demo-chat",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(response.status_code, 400)


class AnthropicStreamTranslateAsyncTests(IsolatedAsyncioTestCase):
    async def test_translate_chat_sse_emits_anthropic_lifecycle(self) -> None:
        async def chunks():
            yield b'data: {"choices":[{"delta":{"content":"Z"},"finish_reason":null}]}\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n'

        parts = [
            chunk
            async for chunk in _translate_chat_sse_to_anthropic(
                chunks(),
                client_model_alias="demo",
            )
        ]
        joined = b"".join(parts)
        self.assertIn(b"message_start", joined)
        self.assertIn(b"content_block_start", joined)
        self.assertIn(b"text_delta", joined)
        self.assertIn(b"max_tokens", joined)
        self.assertIn(b"message_stop", joined)
