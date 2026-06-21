"""Tests for multi-mode gateway proxy routes."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from django.test import SimpleTestCase
from fastapi.testclient import TestClient

from orchestrator.gateway.app import create_app
from orchestrator.gateway.router import (
    EMBEDDINGS_PATH,
    GatewayRouteError,
    GatewayTarget,
    IMAGES_PATH,
    RERANK_PATH,
)

EMBED_TARGET = GatewayTarget(
    alias="local-embed",
    instance_id=1,
    launch_mode="EMBEDDING",
    host="127.0.0.1",
    port=11410,
    upstream_model="local-embed",
    api_path=EMBEDDINGS_PATH,
)

IMAGE_TARGET = GatewayTarget(
    alias="flux-local",
    instance_id=2,
    launch_mode="IMAGE",
    host="127.0.0.1",
    port=11411,
    upstream_model="flux-local",
    api_path=IMAGES_PATH,
)

TEXT_TARGET = GatewayTarget(
    alias="llama-chat",
    instance_id=3,
    launch_mode="TEXT",
    host="127.0.0.1",
    port=11412,
    upstream_model="default_model",
    api_path="/v1/chat/completions",
)


def _mock_buffered_client(mock_client_cls: MagicMock, response: MagicMock) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client
    return mock_client


class GatewayModeProxyTests(SimpleTestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    @patch("orchestrator.gateway.selectors.resolve_gateway_target", return_value=EMBED_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    def test_embeddings_proxies_to_upstream(
        self,
        mock_client_cls: MagicMock,
        _mock_resolve: MagicMock,
    ) -> None:
        upstream = MagicMock()
        upstream.status_code = 200
        upstream.json.return_value = {"object": "list", "data": []}
        upstream.headers = httpx.Headers({"content-type": "application/json"})
        mock_client = _mock_buffered_client(mock_client_cls, upstream)

        response = self.client.post(
            "/v1/embeddings",
            json={"model": "local-embed", "input": "hello"},
        )

        self.assertEqual(response.status_code, 200)
        call_args = mock_client.post.await_args
        self.assertEqual(call_args.args[0], "http://127.0.0.1:11410/v1/embeddings")
        upstream_body = call_args.kwargs["json"]
        self.assertEqual(upstream_body["model"], "local-embed")
        self.assertEqual(upstream_body["input"], "hello")

    @patch("orchestrator.gateway.selectors.resolve_gateway_target", return_value=EMBED_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    def test_embeddings_batch_input_preserved(
        self,
        mock_client_cls: MagicMock,
        _mock_resolve: MagicMock,
    ) -> None:
        upstream = MagicMock()
        upstream.status_code = 200
        upstream.json.return_value = {"object": "list", "data": []}
        upstream.headers = httpx.Headers({"content-type": "application/json"})
        mock_client = _mock_buffered_client(mock_client_cls, upstream)

        batch = ["first", "second"]
        response = self.client.post(
            "/v1/embeddings",
            json={"model": "local-embed", "input": batch},
        )

        self.assertEqual(response.status_code, 200)
        upstream_body = mock_client.post.await_args.kwargs["json"]
        self.assertEqual(upstream_body["input"], batch)

    @patch("orchestrator.gateway.selectors.resolve_gateway_target", return_value=TEXT_TARGET)
    def test_embeddings_rejects_text_instance(self, _mock_resolve: MagicMock) -> None:
        response = self.client.post(
            "/v1/embeddings",
            json={"model": "llama-chat", "input": "hello"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "unsupported_endpoint")

    @patch("orchestrator.gateway.selectors.resolve_gateway_target", return_value=IMAGE_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    def test_image_generations_proxies_to_upstream(
        self,
        mock_client_cls: MagicMock,
        _mock_resolve: MagicMock,
    ) -> None:
        upstream = MagicMock()
        upstream.status_code = 200
        upstream.json.return_value = {"created": 1, "data": []}
        upstream.headers = httpx.Headers({"content-type": "application/json"})
        mock_client = _mock_buffered_client(mock_client_cls, upstream)

        response = self.client.post(
            "/v1/images/generations",
            json={"model": "flux-local", "prompt": "a cat"},
        )
        self.assertEqual(response.status_code, 200)
        upstream_body = mock_client.post.await_args.kwargs["json"]
        self.assertEqual(upstream_body["model"], "flux-local")
        self.assertEqual(upstream_body["prompt"], "a cat")

    @patch("orchestrator.gateway.selectors.resolve_gateway_target", return_value=TEXT_TARGET)
    def test_image_generations_rejects_text_instance(self, _mock_resolve: MagicMock) -> None:
        response = self.client.post(
            "/v1/images/generations",
            json={"model": "llama-chat", "prompt": "a cat"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "unsupported_endpoint")

    @patch("orchestrator.gateway.selectors.resolve_gateway_target", return_value=IMAGE_TARGET)
    def test_rerank_rejects_image_instance(self, _mock_resolve: MagicMock) -> None:
        response = self.client.post(
            "/v1/rerank",
            json={
                "model": "flux-local",
                "query": "test",
                "documents": ["doc"],
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "unsupported_endpoint")

    @patch("orchestrator.gateway.selectors.resolve_gateway_target")
    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    def test_rerank_proxies_to_upstream(
        self,
        mock_client_cls: MagicMock,
        mock_resolve: MagicMock,
    ) -> None:
        mock_resolve.return_value = GatewayTarget(
            alias="rerank-local",
            instance_id=4,
            launch_mode="RERANKER",
            host="127.0.0.1",
            port=11413,
            upstream_model="rerank-local",
            api_path=RERANK_PATH,
        )
        upstream = MagicMock()
        upstream.status_code = 200
        upstream.json.return_value = {"results": []}
        upstream.headers = httpx.Headers({"content-type": "application/json"})
        mock_client = _mock_buffered_client(mock_client_cls, upstream)

        response = self.client.post(
            "/v1/rerank",
            json={
                "model": "rerank-local",
                "query": "python",
                "documents": ["Python is great"],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            mock_client.post.await_args.args[0],
            "http://127.0.0.1:11413/v1/rerank",
        )
        upstream_body = mock_client.post.await_args.kwargs["json"]
        self.assertEqual(upstream_body["model"], "rerank-local")
        self.assertEqual(upstream_body["query"], "python")

    @patch("orchestrator.gateway.selectors.resolve_gateway_target", return_value=TEXT_TARGET)
    def test_rerank_rejects_text_instance(self, _mock_resolve: MagicMock) -> None:
        response = self.client.post(
            "/v1/rerank",
            json={
                "model": "llama-chat",
                "query": "python",
                "documents": ["Python is great"],
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "unsupported_endpoint")

    @patch("orchestrator.gateway.selectors.resolve_gateway_target", return_value=EMBED_TARGET)
    def test_rerank_rejects_embedding_instance(self, _mock_resolve: MagicMock) -> None:
        response = self.client.post(
            "/v1/rerank",
            json={
                "model": "local-embed",
                "query": "python",
                "documents": ["Python is great"],
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "unsupported_endpoint")

    @patch("orchestrator.gateway.selectors.resolve_gateway_target")
    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    def test_audio_speech_returns_binary_payload(
        self,
        mock_client_cls: MagicMock,
        mock_resolve: MagicMock,
    ) -> None:
        mock_resolve.return_value = GatewayTarget(
            alias="kokoro",
            instance_id=5,
            launch_mode="TTS",
            host="127.0.0.1",
            port=11414,
            upstream_model="kokoro",
            api_path="/v1/audio/speech",
        )
        upstream = MagicMock()
        upstream.status_code = 200
        upstream.content = b"RIFFaudio"
        upstream.headers = httpx.Headers({"content-type": "audio/wav"})
        upstream.json.side_effect = ValueError("not json")
        _mock_buffered_client(mock_client_cls, upstream)

        response = self.client.post(
            "/v1/audio/speech",
            json={"model": "kokoro", "input": "Hello"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"RIFFaudio")

    @patch("orchestrator.gateway.routes.models.list_running_gateway_models")
    def test_list_models_endpoint_returns_openai_format(
        self,
        mock_list: MagicMock,
    ) -> None:
        mock_list.return_value = {
            "object": "list",
            "data": [{"id": "llama-chat", "object": "model", "created": 1, "owned_by": "nadir"}],
        }
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], "llama-chat")

    def test_health_includes_docs_and_models_links(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["docs"], "/docs")
        self.assertEqual(payload["models"], "/v1/models")
