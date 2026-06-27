"""OpenAPI contract tests for POST /v1/chat/completions (MLX-47)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from jsonschema.exceptions import ValidationError

from orchestrator.gateway.route_cache import clear_gateway_route_cache
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, GatewayTarget
from orchestrator.models import InferenceInstance
from orchestrator.tests.contracts.validators import (
    format_validation_error,
    validate_response_body,
)

TEXT_TARGET = GatewayTarget(
    alias="contract-chat",
    instance_id=1,
    launch_mode="TEXT",
    host="127.0.0.1",
    port=11460,
    upstream_model="default_model",
    api_path=CHAT_COMPLETIONS_PATH,
)

OPENAI_CHAT_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-contract-1",
    "object": "chat.completion",
    "created": 1719500000,
    "model": "contract-chat",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from contract test."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
}


def _mock_buffered_client(mock_client_cls: MagicMock, response: MagicMock) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client
    return mock_client


@pytest.mark.contract
@pytest.mark.django_db(transaction=True)
@patch("orchestrator.lifecycle_services.touch_instance_last_used_at")
@patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
@patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
def test_chat_completion_response_matches_openapi_contract(
    mock_client_cls: MagicMock,
    _mock_ready: MagicMock,
    _mock_touch: MagicMock,
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    clear_gateway_route_cache()
    InferenceInstance.objects.create(
        model_name="contract-chat",
        port=11460,
        launch_mode="TEXT",
        server_config={"model_id": "contract-chat"},
        status="RUNNING",
    )

    upstream = MagicMock()
    upstream.status_code = 200
    upstream.json.return_value = OPENAI_CHAT_RESPONSE
    upstream.headers = httpx.Headers({"content-type": "application/json"})
    _mock_buffered_client(mock_client_cls, upstream)

    request_body = {
        "model": "contract-chat",
        "messages": [{"role": "user", "content": "ping"}],
    }
    response = gateway_client.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 200

    try:
        validate_response_body(
            curated_spec,
            path="/v1/chat/completions",
            method="post",
            status_code=200,
            body=response.json(),
        )
    except ValidationError as exc:
        pytest.fail(format_validation_error(exc))


@pytest.mark.contract
@pytest.mark.django_db(transaction=True)
def test_chat_completion_unavailable_model_returns_openai_error_shape(
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    clear_gateway_route_cache()
    InferenceInstance.objects.create(
        model_name="offline-contract",
        port=11461,
        launch_mode="TEXT",
        server_config={"model_id": "offline-contract"},
        status="STOPPED",
    )

    response = gateway_client.post(
        "/v1/chat/completions",
        json={
            "model": "offline-contract",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 503

    try:
        validate_response_body(
            curated_spec,
            path="/v1/chat/completions",
            method="post",
            status_code=503,
            body=response.json(),
        )
    except ValidationError as exc:
        pytest.fail(format_validation_error(exc))

    error = response.json()["error"]
    assert error["code"] == "model_unavailable"
    assert error["type"] == "model_unavailable"


@pytest.mark.contract
@pytest.mark.django_db(transaction=True)
def test_chat_completion_missing_model_returns_openai_error_shape(
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    clear_gateway_route_cache()

    response = gateway_client.post(
        "/v1/chat/completions",
        json={"model": "missing-alias", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404

    validate_response_body(
        curated_spec,
        path="/v1/chat/completions",
        method="post",
        status_code=404,
        body=response.json(),
    )
