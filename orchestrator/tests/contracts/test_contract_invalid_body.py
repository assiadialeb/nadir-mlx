"""OpenAPI contract tests for 400 invalid request body (MLX-62)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from orchestrator.gateway.route_cache import clear_gateway_route_cache
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, GatewayTarget
from orchestrator.models import InferenceInstance
from orchestrator.tests.contracts.helpers import assert_contract_json_response

TEXT_TARGET = GatewayTarget(
    alias="contract-invalid-body",
    instance_id=30,
    launch_mode="TEXT",
    host="127.0.0.1",
    port=11490,
    upstream_model="default_model",
    api_path=CHAT_COMPLETIONS_PATH,
)

JSON_HEADERS = {"Content-Type": "application/json"}


@pytest.mark.contract
def test_chat_completion_malformed_json_returns_invalid_request_contract(
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    response = gateway_client.post(
        "/v1/chat/completions",
        content=b"{not-json",
        headers=JSON_HEADERS,
    )
    body = assert_contract_json_response(
        curated_spec,
        path="/v1/chat/completions",
        method="post",
        status_code=400,
        response=response,
    )
    assert body["error"]["type"] == "invalid_request"


@pytest.mark.contract
def test_chat_completion_non_object_json_returns_invalid_request_contract(
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    response = gateway_client.post(
        "/v1/chat/completions",
        content=b"[1, 2, 3]",
        headers=JSON_HEADERS,
    )
    body = assert_contract_json_response(
        curated_spec,
        path="/v1/chat/completions",
        method="post",
        status_code=400,
        response=response,
    )
    assert body["error"]["type"] == "invalid_request"


@pytest.mark.contract
def test_chat_completion_missing_model_returns_invalid_model_contract(
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    response = gateway_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    body = assert_contract_json_response(
        curated_spec,
        path="/v1/chat/completions",
        method="post",
        status_code=400,
        response=response,
    )
    error = body["error"]
    assert error["type"] == "invalid_model"
    assert error["code"] == "invalid_model"


@pytest.mark.contract
def test_chat_completion_empty_model_returns_invalid_model_contract(
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    response = gateway_client.post(
        "/v1/chat/completions",
        json={"model": "   ", "messages": [{"role": "user", "content": "hi"}]},
    )
    body = assert_contract_json_response(
        curated_spec,
        path="/v1/chat/completions",
        method="post",
        status_code=400,
        response=response,
    )
    error = body["error"]
    assert error["type"] == "invalid_model"
    assert error["code"] == "invalid_model"


@pytest.mark.contract
@pytest.mark.django_db(transaction=True)
@patch("orchestrator.lifecycle_services.touch_instance_last_used_at")
@patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
@patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
def test_chat_completion_missing_messages_upstream_400_matches_contract(
    mock_client_cls: MagicMock,
    _mock_ready: MagicMock,
    _mock_touch: MagicMock,
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    clear_gateway_route_cache()
    InferenceInstance.objects.create(
        model_name="contract-invalid-body",
        port=11490,
        launch_mode="TEXT",
        server_config={"model_id": "contract-invalid-body"},
        status="RUNNING",
    )

    upstream = MagicMock()
    upstream.status_code = 400
    upstream.json.return_value = {
        "error": {
            "message": "messages is required",
            "type": "invalid_request_error",
            "code": "missing_messages",
        }
    }
    upstream.headers = httpx.Headers({"content-type": "application/json"})
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=upstream)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    response = gateway_client.post(
        "/v1/chat/completions",
        json={"model": "contract-invalid-body"},
    )
    body = assert_contract_json_response(
        curated_spec,
        path="/v1/chat/completions",
        method="post",
        status_code=400,
        response=response,
    )
    assert body["error"]["message"]
