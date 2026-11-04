"""SDK façade smoke tests (MLX-121).

Uses official ``openai``, ``anthropic``, and ``ollama`` async clients against
the FastAPI app via ``httpx.ASGITransport`` with mocked upstream MLX.

Install clients (test extras)::

    pip install -r requirements-test.txt

Run::

    pytest -m sdk orchestrator/tests/test_gateway_sdk_smoke.py -q
"""

from __future__ import annotations

from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

pytest.importorskip("openai")
pytest.importorskip("anthropic")
pytest.importorskip("ollama")

from anthropic import AsyncAnthropic  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from ollama import AsyncClient as OllamaAsyncClient  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from orchestrator.gateway.app import create_app
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, GatewayTarget

TEXT_TARGET = GatewayTarget(
    alias="demo-chat",
    instance_id=1,
    launch_mode="TEXT",
    host="127.0.0.1",
    port=11400,
    upstream_model="default_model",
    api_path=CHAT_COMPLETIONS_PATH,
)

CHAT_UPSTREAM = {
    "id": "chatcmpl-sdk",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "sdk-ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _mock_json_upstream(mock_get_client: MagicMock, payload: dict) -> None:
    upstream = httpx.Response(
        200,
        json=payload,
        headers={"content-type": "application/json"},
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=upstream)
    mock_get_client.return_value = mock_client


def _mock_sse_upstream(mock_get_client: MagicMock, lines: bytes) -> None:
    async def chunk_stream():
        yield lines

    upstream = MagicMock()
    upstream.status_code = 200
    upstream.aiter_bytes = chunk_stream
    upstream.aclose = AsyncMock()
    mock_client = AsyncMock()
    mock_client.build_request = MagicMock(return_value=MagicMock())
    mock_client.send = AsyncMock(return_value=upstream)
    mock_get_client.return_value = mock_client


@pytest.mark.sdk
class GatewaySdkSmokeTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.app = create_app()
        self._touch_patcher = patch(
            "orchestrator.lifecycle_services.touch_instance_last_used_at",
        )
        self._touch_patcher.start()
        self._http = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://testserver",
        )
        await self._http.__aenter__()

    async def asyncTearDown(self) -> None:
        await self._http.__aexit__(None, None, None)
        self._touch_patcher.stop()

    async def _streaming_supported(self, path: str, body: dict) -> bool:
        probe = await self._http.post(path, json=body)
        return probe.status_code != 501

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    async def test_openai_sdk_chat_completions(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        _mock_json_upstream(mock_get_client, CHAT_UPSTREAM)
        client = AsyncOpenAI(
            api_key="test-key",
            base_url="http://testserver/v1",
            http_client=self._http,
        )
        completion = await client.chat.completions.create(
            model="demo-chat",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
        )
        self.assertEqual(completion.choices[0].message.content, "sdk-ok")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    async def test_openai_sdk_responses_create(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        _mock_json_upstream(mock_get_client, CHAT_UPSTREAM)
        client = AsyncOpenAI(
            api_key="test-key",
            base_url="http://testserver/v1",
            http_client=self._http,
        )
        response = await client.responses.create(
            model="demo-chat",
            input="ping",
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Lookup",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )
        self.assertTrue(getattr(response, "id", None) or getattr(response, "output", None) is not None)

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    async def test_anthropic_sdk_messages(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        _mock_json_upstream(mock_get_client, CHAT_UPSTREAM)
        client = AsyncAnthropic(
            api_key="test-key",
            base_url="http://testserver",
            http_client=self._http,
        )
        message = await client.messages.create(
            model="demo-chat",
            max_tokens=16,
            messages=[{"role": "user", "content": "ping"}],
        )
        text_blocks = [b for b in message.content if getattr(b, "type", None) == "text"]
        self.assertTrue(text_blocks)
        self.assertEqual(text_blocks[0].text, "sdk-ok")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    async def test_anthropic_sdk_messages_stream(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        supported = await self._streaming_supported(
            "/v1/messages",
            {
                "model": "demo-chat",
                "max_tokens": 8,
                "stream": True,
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        if not supported:
            self.skipTest("Anthropic streaming requires MLX-107 / PR #34")

        _mock_sse_upstream(
            mock_get_client,
            b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
            b"data: [DONE]\n\n",
        )
        client = AsyncAnthropic(
            api_key="test-key",
            base_url="http://testserver",
            http_client=self._http,
        )
        chunks: list[str] = []
        async with client.messages.stream(
            model="demo-chat",
            max_tokens=16,
            messages=[{"role": "user", "content": "ping"}],
        ) as stream:
            async for text in stream.text_stream:
                chunks.append(text)
        self.assertIn("hi", "".join(chunks))

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    async def test_ollama_sdk_chat(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        _mock_json_upstream(mock_get_client, CHAT_UPSTREAM)
        client = OllamaAsyncClient(
            host="http://testserver",
            transport=ASGITransport(app=self.app),
        )
        result = await client.chat(
            model="demo-chat",
            messages=[{"role": "user", "content": "ping"}],
            stream=False,
        )
        content = (
            result.get("message", {}).get("content")
            if isinstance(result, dict)
            else result.message.content
        )
        self.assertEqual(content, "sdk-ok")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
    @patch("orchestrator.gateway.services.http_proxy.get_http_client")
    async def test_ollama_sdk_chat_stream(
        self,
        mock_get_client: MagicMock,
        _mock_ready: MagicMock,
    ) -> None:
        supported = await self._streaming_supported(
            "/api/chat",
            {
                "model": "demo-chat",
                "stream": True,
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        if not supported:
            self.skipTest("Ollama streaming requires MLX-108 / PR #35")

        _mock_sse_upstream(
            mock_get_client,
            b'data: {"choices":[{"delta":{"content":"yo"}}]}\n'
            b"data: [DONE]\n\n",
        )
        client = OllamaAsyncClient(
            host="http://testserver",
            transport=ASGITransport(app=self.app),
        )
        parts: list[str] = []
        stream = await client.chat(
            model="demo-chat",
            messages=[{"role": "user", "content": "ping"}],
            stream=True,
        )
        async for chunk in stream:
            message = chunk.get("message") if isinstance(chunk, dict) else chunk.message
            text = message.get("content") if isinstance(message, dict) else message.content
            if text:
                parts.append(text)
        self.assertIn("yo", "".join(parts))
