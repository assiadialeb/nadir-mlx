"""Tests for POST /v1/responses gateway façade."""

from __future__ import annotations

from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from django.test import SimpleTestCase
from fastapi.testclient import TestClient

from orchestrator.gateway.app import create_app
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, GatewayTarget
from orchestrator.gateway.services.responses_proxy import (
    _map_chat_sse_line,
    _sse_event,
    _translate_chat_sse_to_responses,
)

TEXT_TARGET = GatewayTarget(
    alias="demo-chat",
    instance_id=1,
    launch_mode="TEXT",
    host="127.0.0.1",
    port=11400,
    upstream_model="default_model",
    api_path=CHAT_COMPLETIONS_PATH,
)


class GatewayResponsesProxyTests(SimpleTestCase):
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
    def test_responses_non_stream_translates_chat_completion(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        upstream = httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "created": 10,
                "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            headers={"content-type": "application/json"},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=upstream)
        mock_get_client.return_value = mock_client

        response = self.client.post(
            "/v1/responses",
            json={"model": "demo-chat", "input": "Hi"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "response")
        self.assertEqual(payload["model"], "demo-chat")
        self.assertEqual(payload["output"][0]["content"][0]["text"], "Hello")
        forwarded = mock_client.post.await_args.kwargs["json"]
        self.assertEqual(forwarded["model"], "default_model")
        self.assertEqual(forwarded["messages"][0]["role"], "user")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    def test_responses_rejects_previous_response_id(self, _mock_ready: MagicMock) -> None:
        response = self.client.post(
            "/v1/responses",
            json={
                "model": "demo-chat",
                "input": "Hi",
                "previous_response_id": "resp_x",
            },
        )
        self.assertEqual(response.status_code, 400)
        error = response.json()["error"]
        self.assertEqual(error["code"], "invalid_request")
        self.assertEqual(error["param"], "previous_response_id")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    def test_responses_upstream_error_passthrough(
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
            "/v1/responses",
            json={"model": "demo-chat", "input": "Hi"},
        )
        self.assertEqual(response.status_code, 400)

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    def test_responses_stream_translates_sse(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        async def chunk_stream():
            yield (
                b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n'
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
            "/v1/responses",
            json={"model": "demo-chat", "input": "Hi", "stream": True},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"event: response.created", response.content)
        self.assertIn(b"response.output_item.added", response.content)
        self.assertIn(b"response.output_text.delta", response.content)
        self.assertIn(b"response.output_text.done", response.content)
        self.assertIn(b"event: response.completed", response.content)


class ResponsesSseTranslateUnitTests(SimpleTestCase):
    def _stream_meta(self) -> dict:
        return {
            "message_id": "msg_1",
            "output_index": 0,
            "text": "",
            "sequence_number": 0,
            "next_output_index": 0,
            "message_opened": False,
            "reasoning_opened": False,
            "reasoning": "",
            "reasoning_id": "rs_1",
            "tools": __import__(
                "orchestrator.gateway.protocol.chat_sse",
                fromlist=["ToolCallAccumulator"],
            ).ToolCallAccumulator(),
            "tool_item_ids": {},
            "tool_output_indexes": {},
            "tool_call_ids": {},
        }

    def test_map_chat_sse_line_delta(self) -> None:
        meta = self._stream_meta()
        events = _map_chat_sse_line(
            b'data: {"choices":[{"delta":{"content":"ab"}}]}',
            meta,
        )
        self.assertEqual(meta["text"], "ab")
        self.assertTrue(any(b"response.output_text.delta" in event for event in events))
        self.assertTrue(meta["message_opened"])

    def test_map_chat_sse_line_ignores_noise(self) -> None:
        meta = self._stream_meta()
        self.assertEqual(_map_chat_sse_line(b": comment", meta), [])
        self.assertEqual(_map_chat_sse_line(b"data: [DONE]", meta), [])
        self.assertEqual(_map_chat_sse_line(b"data: {bad", meta), [])
        self.assertEqual(_map_chat_sse_line(b"data: []", meta), [])
        self.assertEqual(
            _map_chat_sse_line(b'data: {"choices":[{"delta":{}}]}', meta),
            [],
        )

    def test_map_chat_sse_line_marks_failed_on_error_payload(self) -> None:
        meta = self._stream_meta()
        events = _map_chat_sse_line(
            b'data: {"error":{"message":"boom","type":"server_error"}}',
            meta,
        )
        self.assertEqual(events, [])
        self.assertTrue(meta["failed"])
        self.assertEqual(meta["error_message"], "boom")

    def test_sse_event_encoding(self) -> None:
        encoded = _sse_event("ping", {"type": "ping"})
        self.assertEqual(encoded, b'event: ping\ndata: {"type":"ping"}\n\n')


class ResponsesSseTranslateAsyncTests(IsolatedAsyncioTestCase):
    async def test_translate_chat_sse_async(self) -> None:
        async def chunks():
            yield b'data: {"choices":[{"delta":{"content":"Z"}}]}\n'
            yield b"partial-without-newline"

        parts = [
            chunk
            async for chunk in _translate_chat_sse_to_responses(
                chunks(),
                client_model_alias="demo",
            )
        ]
        joined = b"".join(parts)
        self.assertIn(b"response.created", joined)
        self.assertIn(b"response.output_item.added", joined)
        self.assertIn(b"response.content_part.added", joined)
        self.assertIn(b"response.output_text.done", joined)
        self.assertIn(b"response.output_item.done", joined)
        self.assertIn(b"response.completed", joined)
        # Message item opens only after the first text delta.
        created_at = joined.find(b"response.created")
        message_added_at = joined.find(b'"type":"message"')
        self.assertGreater(message_added_at, created_at)

    async def test_translate_chat_sse_emits_failed(self) -> None:
        async def chunks():
            yield b'data: {"error":{"message":"upstream boom"}}\n'

        parts = [
            chunk
            async for chunk in _translate_chat_sse_to_responses(
                chunks(),
                client_model_alias="demo",
            )
        ]
        joined = b"".join(parts)
        self.assertIn(b"response.failed", joined)
        self.assertNotIn(b"response.completed", joined)

    async def test_translate_chat_sse_emits_function_call_events(self) -> None:
        async def chunks():
            yield (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                b'"type":"function","function":{"name":"lookup","arguments":""}}]}}]}\n'
            )
            yield (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                b'"function":{"arguments":"{\\"q\\":1}"}}]}}]}\n'
            )
            yield b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n'

        parts = [
            chunk
            async for chunk in _translate_chat_sse_to_responses(
                chunks(),
                client_model_alias="demo",
            )
        ]
        joined = b"".join(parts)
        self.assertIn(b"response.output_item.added", joined)
        self.assertIn(b"function_call", joined)
        self.assertIn(b"response.function_call_arguments.delta", joined)
        self.assertIn(b"response.function_call_arguments.done", joined)
        self.assertIn(b"lookup", joined)
        self.assertIn(b'"q\\":1', joined)
        # Tool-only streams must not emit an empty assistant message item.
        self.assertNotIn(b'"type":"message"', joined)
