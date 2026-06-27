"""OpenAPI contract tests for 400 unsupported_endpoint routing (MLX-61)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, GatewayTarget
from orchestrator.tests.contracts.helpers import assert_contract_json_response

TEXT_TARGET = GatewayTarget(
    alias="contract-text-routing",
    instance_id=20,
    launch_mode="TEXT",
    host="127.0.0.1",
    port=11480,
    upstream_model="default_model",
    api_path=CHAT_COMPLETIONS_PATH,
)

IMAGE_TARGET = GatewayTarget(
    alias="contract-image-routing",
    instance_id=21,
    launch_mode="IMAGE",
    host="127.0.0.1",
    port=11481,
    upstream_model="contract-image-routing",
    api_path="/v1/images/generations",
)


@pytest.mark.contract
@pytest.mark.django_db(transaction=True)
@patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=TEXT_TARGET)
def test_embeddings_on_text_alias_returns_unsupported_endpoint_contract(
    _mock_ready: MagicMock,
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    response = gateway_client.post(
        "/v1/embeddings",
        json={"model": "contract-text-routing", "input": "hello"},
    )
    body = assert_contract_json_response(
        curated_spec,
        path="/v1/embeddings",
        method="post",
        status_code=400,
        response=response,
    )
    error = body["error"]
    assert error["type"] == "unsupported_endpoint"
    assert error["code"] == "unsupported_endpoint"


@pytest.mark.contract
@pytest.mark.django_db(transaction=True)
@patch("orchestrator.lifecycle_services.ensure_instance_ready", return_value=IMAGE_TARGET)
def test_rerank_on_image_alias_returns_unsupported_endpoint_contract(
    _mock_ready: MagicMock,
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    response = gateway_client.post(
        "/v1/rerank",
        json={
            "model": "contract-image-routing",
            "query": "test",
            "documents": ["doc one"],
        },
    )
    body = assert_contract_json_response(
        curated_spec,
        path="/v1/rerank",
        method="post",
        status_code=400,
        response=response,
    )
    error = body["error"]
    assert error["type"] == "unsupported_endpoint"
    assert error["code"] == "unsupported_endpoint"
