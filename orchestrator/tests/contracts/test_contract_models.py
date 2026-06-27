"""OpenAPI contract tests for GET /v1/models (MLX-47)."""

from __future__ import annotations

from typing import Any

import pytest
from jsonschema.exceptions import ValidationError

from orchestrator.gateway.route_cache import clear_gateway_route_cache
from orchestrator.models import InferenceInstance
from orchestrator.tests.contracts.validators import format_validation_error, validate_response_body


@pytest.mark.contract
@pytest.mark.django_db(transaction=True)
def test_list_models_response_matches_openapi_contract(
    gateway_client: Any,
    curated_spec: dict[str, Any],
) -> None:
    clear_gateway_route_cache()
    InferenceInstance.objects.create(
        model_name="contract-llama",
        port=11450,
        launch_mode="TEXT",
        server_config={"model_id": "contract-llama"},
        status="RUNNING",
    )
    InferenceInstance.objects.create(
        model_name="contract-stopped",
        port=11451,
        launch_mode="TEXT",
        server_config={
            "model_id": "contract-stopped",
            "ops": {"lifecycle_mode": "on_demand"},
        },
        status="STOPPED",
    )

    response = gateway_client.get("/v1/models")
    assert response.status_code == 200

    try:
        validate_response_body(
            curated_spec,
            path="/v1/models",
            method="get",
            status_code=200,
            body=response.json(),
        )
    except ValidationError as exc:
        pytest.fail(format_validation_error(exc))

    payload = response.json()
    assert payload["object"] == "list"
    model_ids = {entry["id"] for entry in payload["data"]}
    assert model_ids == {"contract-llama", "contract-stopped"}

    stopped = next(item for item in payload["data"] if item["id"] == "contract-stopped")
    assert stopped["metadata"]["nadir"]["lifecycle_mode"] == "on_demand"
