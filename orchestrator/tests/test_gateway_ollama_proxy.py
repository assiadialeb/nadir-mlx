"""Tests for Ollama API gateway façade."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from django.test import SimpleTestCase
from fastapi.testclient import TestClient

from orchestrator.gateway.app import create_app
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, EMBEDDINGS_PATH, GatewayTarget

TEXT_TARGET = GatewayTarget(
    alias="demo-chat",
    instance_id=1,
    launch_mode="TEXT",
    host="127.0.0.1",
    port=11400,
    upstream_model="default_model",
    api_path=CHAT_COMPLETIONS_PATH,
)

EMBED_TARGET = GatewayTarget(
    alias="demo-embed",
    instance_id=2,
    launch_mode="EMBEDDING",
    host="127.0.0.1",
    port=11410,
    upstream_model="embed-model",
    api_path=EMBEDDINGS_PATH,
)


class GatewayOllamaProxyTests(SimpleTestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())
        self._touch_patcher = patch(
            "orchestrator.lifecycle_services.touch_instance_last_used_at",
        )
        self._touch_patcher.start()

    def tearDown(self) -> None:
        self._touch_patcher.stop()

    @patch(
        "orchestrator.gateway.services.ollama_proxy.list_running_gateway_models",
        return_value={"object": "list", "data": [{"id": "demo-chat"}]},
    )
    def test_tags_lists_models(self, _mock_models: MagicMock) -> None:
        response = self.client.get("/api/tags")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["models"][0]["name"], "demo-chat")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    def test_chat_non_stream(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        upstream = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Hello"}}]},
            headers={"content-type": "application/json"},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=upstream)
        mock_get_client.return_value = mock_client

        response = self.client.post(
            "/api/chat",
            json={"model": "demo-chat", "messages": [{"role": "user", "content": "Hi"}]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"]["content"], "Hello")
        self.assertTrue(response.json()["done"])
        self.assertEqual(response.json()["done_reason"], "stop")

    def test_bootstrap_probes(self) -> None:
        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        self.assertEqual(root.text, "Ollama is running")

        version = self.client.get("/api/version")
        self.assertEqual(version.status_code, 200)
        self.assertIn("version", version.json())

        show = self.client.post("/api/show", json={"name": "demo-chat"})
        self.assertEqual(show.status_code, 200)
        self.assertEqual(show.json()["name"], "demo-chat")

    @patch(
        "orchestrator.gateway.services.ollama_proxy.list_running_gateway_models",
        return_value={"object": "list", "data": [{"id": "demo-chat"}]},
    )
    def test_ps_lists_running_models(self, _mock_models: MagicMock) -> None:
        response = self.client.get("/api/ps")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["models"][0]["name"], "demo-chat")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    def test_generate_non_stream(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        upstream = httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "World"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            headers={"content-type": "application/json"},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=upstream)
        mock_get_client.return_value = mock_client

        response = self.client.post(
            "/api/generate",
            json={"model": "demo-chat", "prompt": "Hello"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "World")
        self.assertTrue(response.json()["done"])
        self.assertEqual(response.json()["eval_count"], 1)

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    def test_chat_stream_translates_ndjson(
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
            "/api/chat",
            json={
                "model": "demo-chat",
                "stream": True,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/x-ndjson", response.headers.get("content-type", ""))
        lines = [json.loads(line) for line in response.content.splitlines() if line]
        self.assertGreaterEqual(len(lines), 2)
        self.assertEqual(lines[0]["message"]["content"], "Hi")
        self.assertFalse(lines[0]["done"])
        self.assertTrue(lines[-1]["done"])

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=EMBED_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    def test_embed_translates(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        upstream = httpx.Response(
            200,
            json={"data": [{"embedding": [0.1, 0.2]}]},
            headers={"content-type": "application/json"},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=upstream)
        mock_get_client.return_value = mock_client

        response = self.client.post(
            "/api/embed",
            json={"model": "demo-embed", "input": "hello"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["embeddings"], [[0.1, 0.2]])

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=EMBED_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    def test_embeddings_legacy_singular(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        upstream = httpx.Response(
            200,
            json={"data": [{"embedding": [0.5, 0.6]}]},
            headers={"content-type": "application/json"},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=upstream)
        mock_get_client.return_value = mock_client

        response = self.client.post(
            "/api/embeddings",
            json={"model": "demo-embed", "prompt": "hello"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["embedding"], [0.5, 0.6])
        self.assertNotIn("embeddings", response.json())

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    def test_chat_upstream_error(
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
            "/api/chat",
            json={"model": "demo-chat", "messages": [{"role": "user", "content": "Hi"}]},
        )
        self.assertEqual(response.status_code, 502)

    def test_chat_invalid_body(self) -> None:
        response = self.client.post(
            "/api/chat",
            content=b"[1]",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
