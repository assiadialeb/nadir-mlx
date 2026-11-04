"""Tests for Anthropic Messages gateway façade."""

from __future__ import annotations

import json
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from django.test import SimpleTestCase
from fastapi.testclient import TestClient

from orchestrator.gateway.app import create_app
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, GatewayTarget
from orchestrator.gateway.services.anthropic_proxy import _translate_chat_sse_to_anthropic


def _anthropic_sse_payloads(raw: bytes) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for block in raw.decode().split("\n\n"):
        data_lines = [
            line[6:] for line in block.split("\n") if line.startswith("data: ")
        ]
        if not data_lines:
            continue
        try:
            parsed = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payloads.append(parsed)
    return payloads

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

    def test_messages_invalid_json_returns_anthropic_error(self) -> None:
        response = self.client.post(
            "/v1/messages",
            content=b"{bad",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertNotIn("code", payload["error"])

    def test_messages_missing_model_returns_anthropic_error(self) -> None:
        response = self.client.post(
            "/v1/messages",
            json={"max_tokens": 8, "messages": [{"role": "user", "content": "Hi"}]},
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error"]["type"], "invalid_request_error")

    def test_count_tokens_returns_estimate(self) -> None:
        response = self.client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "demo-chat",
                "messages": [{"role": "user", "content": "Hello Claude Code probe"}],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model"], "demo-chat")
        self.assertGreater(payload["input_tokens"], 0)

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    def test_messages_tools_round_trip(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        upstream = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "reasoning": "use tool",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
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
                "thinking": {"type": "enabled"},
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Weather",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
                "messages": [{"role": "user", "content": "Paris?"}],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["stop_reason"], "tool_use")
        self.assertEqual(payload["content"][0]["type"], "thinking")
        self.assertEqual(payload["content"][1]["type"], "tool_use")
        self.assertEqual(payload["content"][1]["name"], "get_weather")
        posted = mock_client.post.await_args.kwargs["json"]
        self.assertEqual(posted["tools"][0]["function"]["name"], "get_weather")

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
        payload = response.json()
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error"]["message"], "bad")
        self.assertEqual(payload["error"]["type"], "invalid_request_error")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    def test_messages_rewrites_upstream_server_error(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        upstream = httpx.Response(
            502,
            json={"error": {"message": "down", "type": "server_error"}},
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
        self.assertEqual(response.status_code, 502)
        payload = response.json()
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error"]["message"], "down")


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

    async def test_translate_chat_sse_emits_tool_use_blocks(self) -> None:
        async def chunks():
            yield (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                b'"type":"function","function":{"name":"get_weather","arguments":""}}]}}]}\n'
            )
            yield (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                b'"function":{"arguments":"{\\"city\\":\\"Paris\\"}"}}]}}]}\n'
            )
            yield b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n'
            yield b'data: {"usage":{"prompt_tokens":2,"completion_tokens":5}}\n'

        parts = [
            chunk
            async for chunk in _translate_chat_sse_to_anthropic(
                chunks(),
                client_model_alias="demo",
            )
        ]
        joined = b"".join(parts)
        self.assertIn(b'"type":"tool_use"', joined)
        self.assertIn(b"get_weather", joined)
        self.assertIn(b"input_json_delta", joined)
        self.assertIn(b"Paris", joined)
        self.assertIn(b'"stop_reason":"tool_use"', joined)
        self.assertIn(b'"output_tokens":5', joined)
        self.assertIn(b'"input_tokens":2', joined)
        first_start = joined.find(b"content_block_start")
        first_stop = joined.find(b"content_block_stop")
        self.assertGreater(first_stop, first_start)

    async def test_translate_chat_sse_closes_text_before_tool_block(self) -> None:
        async def chunks():
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n'
            yield (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                b'"type":"function","function":{"name":"lookup","arguments":"{}"}}]}}]}\n'
            )
            yield b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n'

        joined = b"".join(
            [
                chunk
                async for chunk in _translate_chat_sse_to_anthropic(
                    chunks(),
                    client_model_alias="demo",
                )
            ]
        )
        text_start = joined.find(b'"type":"text"')
        text_stop = joined.find(b"content_block_stop")
        tool_start = joined.find(b'"type":"tool_use"')
        self.assertGreater(text_stop, text_start)
        self.assertGreater(tool_start, text_stop)

    async def test_translate_chat_sse_emits_thinking_and_midstream_error(self) -> None:
        async def thinking_chunks():
            yield b'data: {"choices":[{"delta":{"reasoning":"plan"}}]}\n'
            yield b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n'

        joined = b"".join(
            [
                chunk
                async for chunk in _translate_chat_sse_to_anthropic(
                    thinking_chunks(),
                    client_model_alias="demo",
                    thinking_enabled=True,
                )
            ]
        )
        self.assertIn(b"thinking_delta", joined)
        self.assertIn(b"text_delta", joined)

        async def error_chunks():
            yield b'data: {"error":{"message":"boom"}}\n'

        err_joined = b"".join(
            [
                chunk
                async for chunk in _translate_chat_sse_to_anthropic(
                    error_chunks(),
                    client_model_alias="demo",
                )
            ]
        )
        self.assertIn(b"event: error", err_joined)
        self.assertIn(b"boom", err_joined)

    async def test_thinking_ignored_when_not_enabled(self) -> None:
        async def chunks():
            yield b'data: {"choices":[{"delta":{"reasoning":"secret"}}]}\n'
            yield b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n'

        joined = b"".join(
            [
                chunk
                async for chunk in _translate_chat_sse_to_anthropic(
                    chunks(),
                    client_model_alias="demo",
                    thinking_enabled=False,
                )
            ]
        )
        self.assertNotIn(b"thinking_delta", joined)
        self.assertIn(b"text_delta", joined)

    async def test_reopens_text_and_thinking_after_tool_without_closed_deltas(self) -> None:
        """Reasoning → text → tool → text → reasoning must never delta a closed index."""

        async def chunks():
            yield b'data: {"choices":[{"delta":{"reasoning":"plan"}}]}\n'
            yield b'data: {"choices":[{"delta":{"content":"before"}}]}\n'
            yield (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                b'"type":"function","function":{"name":"lookup","arguments":"{}"}}]}}]}\n'
            )
            yield b'data: {"choices":[{"delta":{"content":"after"}}]}\n'
            yield b'data: {"choices":[{"delta":{"reasoning":"more"}}]}\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'

        events = [
            chunk
            async for chunk in _translate_chat_sse_to_anthropic(
                chunks(),
                client_model_alias="demo",
                thinking_enabled=True,
            )
        ]
        open_indexes: set[int] = set()
        closed_indexes: set[int] = set()
        for raw in events:
            for payload in _anthropic_sse_payloads(raw):
                event_type = payload.get("type")
                index = payload.get("index")
                if event_type == "content_block_start" and isinstance(index, int):
                    self.assertNotIn(index, open_indexes)
                    self.assertNotIn(index, closed_indexes)
                    open_indexes.add(index)
                elif event_type == "content_block_stop" and isinstance(index, int):
                    self.assertIn(index, open_indexes)
                    open_indexes.remove(index)
                    closed_indexes.add(index)
                elif event_type == "content_block_delta" and isinstance(index, int):
                    self.assertIn(index, open_indexes)
                    self.assertNotIn(index, closed_indexes)
        self.assertEqual(open_indexes, set())
        self.assertGreaterEqual(len(closed_indexes), 5)
