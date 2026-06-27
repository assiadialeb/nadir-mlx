"""OpenAPI contract tests for multi-mode JSON proxy routes (MLX-47)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.gateway.router import (
    CHAT_COMPLETIONS_PATH,
    EMBEDDINGS_PATH,
    IMAGES_PATH,
    RERANK_PATH,
    GatewayTarget,
)
from orchestrator.tests.contracts.helpers import assert_contract_json_response, mock_buffered_upstream

EMBED_TARGET = GatewayTarget(
    alias="contract-embed",
    instance_id=10,
    launch_mode="EMBEDDING",
    host="127.0.0.1",
    port=11470,
    upstream_model="contract-embed",
    api_path=EMBEDDINGS_PATH,
)

RERANK_TARGET = GatewayTarget(
    alias="contract-rerank",
    instance_id=11,
    launch_mode="RERANKER",
    host="127.0.0.1",
    port=11471,
    upstream_model="contract-rerank",
    api_path=RERANK_PATH,
)

IMAGE_TARGET = GatewayTarget(
    alias="contract-image",
    instance_id=12,
    launch_mode="IMAGE",
    host="127.0.0.1",
    port=11472,
    upstream_model="contract-image",
    api_path=IMAGES_PATH,
)

TEXT_TARGET = GatewayTarget(
    alias="contract-text",
    instance_id=13,
    launch_mode="TEXT",
    host="127.0.0.1",
    port=11473,
    upstream_model="default_model",
    api_path=CHAT_COMPLETIONS_PATH,
)

STT_TARGET = GatewayTarget(
    alias="contract-stt",
    instance_id=14,
    launch_mode="STT",
    host="127.0.0.1",
    port=11474,
    upstream_model="contract-stt",
    api_path="/v1/audio/transcriptions",
)


@pytest.mark.contract
@pytest.mark.django_db(transaction=True)
@patch("orchestrator.lifecycle_services.touch_instance_last_used_at")
@patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=EMBED_TARGET)
@patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
def test_embeddings_response_matches_openapi_contract(
    mock_client_cls: MagicMock,
    _mock_ready: MagicMock,
    _mock_touch: MagicMock,
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    mock_buffered_upstream(
        mock_client_cls,
        json_body={
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
            "model": "contract-embed",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        },
    )

    response = gateway_client.post(
        "/v1/embeddings",
        json={"model": "contract-embed", "input": "hello"},
    )
    assert_contract_json_response(
        curated_spec,
        path="/v1/embeddings",
        method="post",
        status_code=200,
        response=response,
    )


@pytest.mark.contract
@pytest.mark.django_db(transaction=True)
@patch("orchestrator.lifecycle_services.touch_instance_last_used_at")
@patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=RERANK_TARGET)
@patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
def test_rerank_response_matches_openapi_contract(
    mock_client_cls: MagicMock,
    _mock_ready: MagicMock,
    _mock_touch: MagicMock,
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    mock_buffered_upstream(
        mock_client_cls,
        json_body={
            "model": "contract-rerank",
            "results": [{"index": 0, "relevance_score": 0.91, "document": "Python is great"}],
        },
    )

    response = gateway_client.post(
        "/v1/rerank",
        json={
            "model": "contract-rerank",
            "query": "python",
            "documents": ["Python is great"],
        },
    )
    assert_contract_json_response(
        curated_spec,
        path="/v1/rerank",
        method="post",
        status_code=200,
        response=response,
    )


@pytest.mark.contract
@pytest.mark.django_db(transaction=True)
@patch("orchestrator.lifecycle_services.touch_instance_last_used_at")
@patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=IMAGE_TARGET)
@patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
def test_image_generations_response_matches_openapi_contract(
    mock_client_cls: MagicMock,
    _mock_ready: MagicMock,
    _mock_touch: MagicMock,
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    mock_buffered_upstream(
        mock_client_cls,
        json_body={
            "created": 1719500001,
            "data": [{"b64_json": "aGVsbG8="}],
        },
    )

    response = gateway_client.post(
        "/v1/images/generations",
        json={"model": "contract-image", "prompt": "a cat"},
    )
    assert_contract_json_response(
        curated_spec,
        path="/v1/images/generations",
        method="post",
        status_code=200,
        response=response,
    )


@pytest.mark.contract
@pytest.mark.django_db(transaction=True)
@patch("orchestrator.lifecycle_services.touch_instance_last_used_at")
@patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
@patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
def test_completions_response_matches_openapi_contract(
    mock_client_cls: MagicMock,
    _mock_ready: MagicMock,
    _mock_touch: MagicMock,
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    mock_buffered_upstream(
        mock_client_cls,
        json_body={
            "id": "cmpl-contract-1",
            "object": "text_completion",
            "created": 1719500002,
            "model": "contract-text",
            "choices": [{"index": 0, "text": "Hi", "finish_reason": "stop"}],
        },
    )

    response = gateway_client.post(
        "/v1/completions",
        json={"model": "contract-text", "prompt": "Say hi"},
    )
    assert_contract_json_response(
        curated_spec,
        path="/v1/completions",
        method="post",
        status_code=200,
        response=response,
    )


@pytest.mark.contract
@pytest.mark.django_db(transaction=True)
@patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=STT_TARGET)
@patch("orchestrator.gateway.services.mode_proxy.httpx.AsyncClient")
def test_audio_transcriptions_response_matches_openapi_contract(
    mock_client_cls: MagicMock,
    _mock_ready: MagicMock,
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    mock_buffered_upstream(
        mock_client_cls,
        content=b'{"text":"hello world"}',
    )

    response = gateway_client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", b"RIFFtestdata", "audio/wav")},
        data={"model": "contract-stt", "response_format": "json"},
    )
    assert_contract_json_response(
        curated_spec,
        path="/v1/audio/transcriptions",
        method="post",
        status_code=200,
        response=response,
    )


@pytest.mark.contract
@pytest.mark.django_db(transaction=True)
@patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=STT_TARGET)
@patch("orchestrator.gateway.services.mode_proxy.httpx.AsyncClient")
def test_audio_translations_response_matches_openapi_contract(
    mock_client_cls: MagicMock,
    _mock_ready: MagicMock,
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    mock_buffered_upstream(
        mock_client_cls,
        content=b'{"text":"hello"}',
    )

    response = gateway_client.post(
        "/v1/audio/translations",
        files={"file": ("sample.wav", b"RIFFtestdata", "audio/wav")},
        data={"model": "contract-stt", "response_format": "json"},
    )
    assert_contract_json_response(
        curated_spec,
        path="/v1/audio/translations",
        method="post",
        status_code=200,
        response=response,
    )
