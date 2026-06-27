"""OpenAPI contract tests for streaming chat completions (MLX-47)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, GatewayTarget
from orchestrator.tests.contracts.helpers import parse_sse_json_payloads
from orchestrator.tests.contracts.validators import component_schema, validate_against_schema

TEXT_TARGET = GatewayTarget(
    alias="contract-stream",
    instance_id=20,
    launch_mode="TEXT",
    host="127.0.0.1",
    port=11480,
    upstream_model="default_model",
    api_path=CHAT_COMPLETIONS_PATH,
)


def _mock_streaming_client(mock_client_cls: MagicMock, response: MagicMock) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.build_request = MagicMock(return_value=MagicMock())
    mock_client.send = AsyncMock(return_value=response)
    mock_client.aclose = AsyncMock()
    mock_client_cls.return_value = mock_client
    return mock_client


@pytest.mark.contract
@pytest.mark.django_db(transaction=True)
@patch("orchestrator.lifecycle_services.touch_instance_last_used_at")
@patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
@patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
def test_chat_completion_stream_chunks_match_openapi_contract(
    mock_client_cls: MagicMock,
    _mock_ready: MagicMock,
    _mock_touch: MagicMock,
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    chunk_payload = {
        "id": "chatcmpl-stream-1",
        "object": "chat.completion.chunk",
        "created": 1719500010,
        "model": "contract-stream",
        "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
    }

    async def chunk_stream() -> AsyncIterator[bytes]:
        yield f"data: {json.dumps(chunk_payload)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    upstream = MagicMock()
    upstream.status_code = 200
    upstream.aiter_bytes = chunk_stream
    upstream.aclose = AsyncMock()
    _mock_streaming_client(mock_client_cls, upstream)

    response = gateway_client.post(
        "/v1/chat/completions",
        json={
            "model": "contract-stream",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    chunk_schema = component_schema(curated_spec, "ChatCompletionChunk")
    payloads = parse_sse_json_payloads(response.content)
    assert payloads, "Expected at least one SSE JSON payload"
    for payload in payloads:
        validate_against_schema(payload, chunk_schema)
