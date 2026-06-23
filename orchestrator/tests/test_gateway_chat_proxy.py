"""Tests for gateway chat/completion proxy routes."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from django.test import SimpleTestCase
from fastapi.testclient import TestClient

from orchestrator.gateway.app import create_app
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, GatewayRouteError, GatewayTarget
from orchestrator.gateway.services.chat_proxy import prepare_chat_upstream_body

TEXT_TARGET = GatewayTarget(
    alias="llama-chat",
    instance_id=1,
    launch_mode="TEXT",
    host="127.0.0.1",
    port=11400,
    upstream_model="default_model",
    api_path=CHAT_COMPLETIONS_PATH,
)

VLM_TARGET = GatewayTarget(
    alias="vlm-alias",
    instance_id=2,
    launch_mode="MULTIMODAL",
    host="127.0.0.1",
    port=11405,
    upstream_model="vlm-alias",
    api_path=CHAT_COMPLETIONS_PATH,
)

EMBED_TARGET = GatewayTarget(
    alias="local-embed",
    instance_id=6,
    launch_mode="EMBEDDING",
    host="127.0.0.1",
    port=11410,
    upstream_model="local-embed",
    api_path="/v1/embeddings",
)

IMAGE_TARGET = GatewayTarget(
    alias="Flux-1",
    instance_id=7,
    launch_mode="IMAGE",
    host="127.0.0.1",
    port=11400,
    upstream_model="Flux-1",
    api_path="/v1/images/generations",
)

TTS_CHAT_TARGET = GatewayTarget(
    alias="kokoro",
    instance_id=9,
    launch_mode="TTS",
    host="127.0.0.1",
    port=11444,
    upstream_model="kokoro",
    api_path="/v1/audio/speech",
)

STT_CHAT_TARGET = GatewayTarget(
    alias="whispers",
    instance_id=8,
    launch_mode="STT",
    host="127.0.0.1",
    port=11445,
    upstream_model="whispers",
    api_path="/v1/audio/transcriptions",
)


def _mock_buffered_client(mock_client_cls: MagicMock, response: MagicMock) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client
    return mock_client


def _mock_streaming_client(
    mock_client_cls: MagicMock,
    response: MagicMock,
) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.build_request = MagicMock(return_value=MagicMock())
    mock_client.send = AsyncMock(return_value=response)
    mock_client.aclose = AsyncMock()
    mock_client_cls.return_value = mock_client
    return mock_client


class GatewayChatProxyTests(SimpleTestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())
        self._touch_patcher = patch(
            "orchestrator.lifecycle_services.touch_instance_last_used_at",
        )
        self._touch_patcher.start()

    def tearDown(self) -> None:
        self._touch_patcher.stop()

    def test_prepare_upstream_body_rewrites_text_model(self) -> None:
        body = prepare_chat_upstream_body(
            {"model": "llama-chat", "messages": []},
            TEXT_TARGET.upstream_model,
        )
        self.assertEqual(body["model"], "default_model")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    def test_chat_completions_non_streaming_proxies_upstream(
        self,
        mock_client_cls: MagicMock,
        _mock_resolve: MagicMock,
    ) -> None:
        upstream = MagicMock()
        upstream.status_code = 200
        upstream.json.return_value = {"id": "chatcmpl-1", "choices": []}
        upstream.headers = httpx.Headers({"content-type": "application/json"})
        mock_client = _mock_buffered_client(mock_client_cls, upstream)

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-chat",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "chatcmpl-1")
        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.await_args
        self.assertEqual(call_args.args[0], "http://127.0.0.1:11400/v1/chat/completions")
        self.assertEqual(call_args.kwargs["json"]["model"], "default_model")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    def test_chat_completions_streaming_forwards_sse_chunks(
        self,
        mock_client_cls: MagicMock,
        _mock_resolve: MagicMock,
    ) -> None:
        async def chunk_stream():
            yield b"data: {\"choices\":[]}\n\n"

        upstream = MagicMock()
        upstream.status_code = 200
        upstream.aiter_bytes = chunk_stream
        upstream.aclose = AsyncMock()
        _mock_streaming_client(mock_client_cls, upstream)

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-chat",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"data:", response.content)

    @patch(
        "orchestrator.lifecycle_services.ensure_instance_ready",
        side_effect=GatewayRouteError(
            status_code=404,
            code="model_not_found",
            message="No inference instance is registered for alias 'missing-alias'.",
        ),
    )
    def test_chat_completions_unknown_alias_returns_404(self, _mock_resolve: MagicMock) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "missing-alias",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["type"], "model_not_found")

    @patch(
        "orchestrator.lifecycle_services.ensure_instance_ready",
        side_effect=GatewayRouteError(
            status_code=503,
            code="model_unavailable",
            message="Instance is stopped.",
        ),
    )
    def test_chat_completions_stopped_instance_returns_503(self, _mock_resolve: MagicMock) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "offline",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        self.assertEqual(response.status_code, 503)

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    def test_text_completions_uses_completions_path(
        self,
        mock_client_cls: MagicMock,
        _mock_resolve: MagicMock,
    ) -> None:
        upstream = MagicMock()
        upstream.status_code = 200
        upstream.json.return_value = {"id": "cmpl-1", "choices": []}
        upstream.headers = httpx.Headers({"content-type": "application/json"})
        mock_client = _mock_buffered_client(mock_client_cls, upstream)

        response = self.client.post(
            "/v1/completions",
            json={"model": "llama-chat", "prompt": "Hello"},
        )

        self.assertEqual(response.status_code, 200)
        call_args = mock_client.post.await_args
        self.assertEqual(call_args.args[0], "http://127.0.0.1:11400/v1/completions")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=VLM_TARGET)
    def test_text_completions_rejects_multimodal_instance(self, _mock_resolve: MagicMock) -> None:
        response = self.client.post(
            "/v1/completions",
            json={"model": "vlm-alias", "prompt": "Hello"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "unsupported_endpoint")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=EMBED_TARGET)
    def test_chat_completions_rejects_embedding_instance(self, _mock_resolve: MagicMock) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "local-embed",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "unsupported_endpoint")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=IMAGE_TARGET)
    def test_chat_completions_rejects_image_instance(self, _mock_resolve: MagicMock) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "Flux-1",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "unsupported_endpoint")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TTS_CHAT_TARGET)
    def test_chat_completions_rejects_tts_instance(self, _mock_resolve: MagicMock) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "kokoro",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "unsupported_endpoint")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=STT_CHAT_TARGET)
    def test_chat_completions_rejects_stt_instance(self, _mock_resolve: MagicMock) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "whispers",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "unsupported_endpoint")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=VLM_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    def test_chat_completions_multimodal_instance_allowed(
        self,
        mock_client_cls: MagicMock,
        _mock_resolve: MagicMock,
    ) -> None:
        upstream = MagicMock()
        upstream.status_code = 200
        upstream.json.return_value = {"id": "chatcmpl-vlm", "choices": []}
        upstream.headers = httpx.Headers({"content-type": "application/json"})
        mock_client = _mock_buffered_client(mock_client_cls, upstream)

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "vlm-alias",
                "messages": [{"role": "user", "content": "Describe image"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        call_args = mock_client.post.await_args
        self.assertEqual(call_args.kwargs["json"]["model"], "vlm-alias")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    def test_chat_completions_upstream_error_is_passthrough(
        self,
        mock_client_cls: MagicMock,
        _mock_resolve: MagicMock,
    ) -> None:
        upstream = MagicMock()
        upstream.status_code = 422
        upstream.json.return_value = {"error": {"message": "Invalid payload"}}
        upstream.headers = httpx.Headers({"content-type": "application/json"})
        upstream.text = '{"error":{"message":"Invalid payload"}}'
        _mock_buffered_client(mock_client_cls, upstream)

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-chat",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        self.assertEqual(response.status_code, 422)

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    def test_chat_completions_forwards_tools_payload_unchanged(
        self,
        mock_client_cls: MagicMock,
        _mock_resolve: MagicMock,
    ) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Return weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        upstream = MagicMock()
        upstream.status_code = 200
        upstream.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"city\":\"Paris\"}",
                                },
                            }
                        ],
                    }
                }
            ]
        }
        upstream.headers = httpx.Headers({"content-type": "application/json"})
        mock_client = _mock_buffered_client(mock_client_cls, upstream)

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-chat",
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": tools,
                "tool_choice": "auto",
            },
        )

        self.assertEqual(response.status_code, 200)
        forwarded = mock_client.post.await_args.kwargs["json"]
        self.assertEqual(forwarded["tools"], tools)
        self.assertEqual(forwarded["tool_choice"], "auto")
        self.assertEqual(forwarded["model"], "default_model")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    def test_chat_completions_forwards_json_schema_response_format(
        self,
        mock_client_cls: MagicMock,
        _mock_resolve: MagicMock,
    ) -> None:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "city_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
        upstream = MagicMock()
        upstream.status_code = 200
        upstream.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "{\"city\":\"Paris\"}"}}]
        }
        upstream.headers = httpx.Headers({"content-type": "application/json"})
        mock_client = _mock_buffered_client(mock_client_cls, upstream)

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-chat",
                "messages": [{"role": "user", "content": "Return JSON"}],
                "response_format": response_format,
            },
        )

        self.assertEqual(response.status_code, 200)
        forwarded = mock_client.post.await_args.kwargs["json"]
        self.assertEqual(forwarded["response_format"], response_format)

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=VLM_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    def test_chat_completions_multimodal_forwards_vision_messages(
        self,
        mock_client_cls: MagicMock,
        _mock_resolve: MagicMock,
    ) -> None:
        vision_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is dominant?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                        },
                    },
                ],
            }
        ]
        upstream = MagicMock()
        upstream.status_code = 200
        upstream.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Red."}}]
        }
        upstream.headers = httpx.Headers({"content-type": "application/json"})
        mock_client = _mock_buffered_client(mock_client_cls, upstream)

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "vlm-alias",
                "messages": vision_messages,
                "max_tokens": 64,
            },
        )

        self.assertEqual(response.status_code, 200)
        forwarded = mock_client.post.await_args.kwargs["json"]
        self.assertEqual(forwarded["messages"], vision_messages)
        self.assertEqual(forwarded["model"], "vlm-alias")
