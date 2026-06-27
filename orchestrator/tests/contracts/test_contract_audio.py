"""OpenAPI contract tests for TTS audio/speech (MLX-47)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from orchestrator.gateway.router import GatewayTarget

TTS_TARGET = GatewayTarget(
    alias="contract-tts",
    instance_id=21,
    launch_mode="TTS",
    host="127.0.0.1",
    port=11481,
    upstream_model="contract-tts",
    api_path="/v1/audio/speech",
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
@patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TTS_TARGET)
@patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
def test_audio_speech_returns_binary_audio_payload(
    mock_client_cls: MagicMock,
    _mock_ready: MagicMock,
    _mock_touch: MagicMock,
    gateway_client: Any,
) -> None:
    upstream = MagicMock()
    upstream.status_code = 200
    upstream.headers = httpx.Headers({"content-type": "audio/wav"})

    async def _aiter_bytes() -> AsyncIterator[bytes]:
        yield b"RIFFaudio"

    upstream.aiter_bytes = _aiter_bytes
    upstream.aread = AsyncMock(return_value=b"")
    upstream.aclose = AsyncMock()
    _mock_streaming_client(mock_client_cls, upstream)

    response = gateway_client.post(
        "/v1/audio/speech",
        json={"model": "contract-tts", "input": "Hello"},
    )
    assert response.status_code == 200
    assert response.content.startswith(b"RIFF")
    assert response.headers.get("content-type", "").startswith("audio/")
