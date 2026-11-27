"""The contract gateway client closes the connections its requests open (MLX-137)."""

from __future__ import annotations

from typing import Any

import pytest

from orchestrator.models import InferenceInstance
from orchestrator.tests.contracts.conftest import gateway_test_client


@pytest.mark.contract
@pytest.mark.ticket("MLX-137")
@pytest.mark.ac("MLX-137:AC1")
@pytest.mark.django_db(transaction=True)
def test_gateway_client_closes_executor_thread_connections_ac1() -> None:
    """AC1: connections opened by gateway executor threads are closed on exit."""
    InferenceInstance.objects.create(
        model_name="teardown-llama",
        port=11470,
        launch_mode="TEXT",
        server_config={"model_id": "teardown-llama"},
        status="RUNNING",
    )
    opened: list[Any] = []

    with gateway_test_client(opened) as client:
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert opened, "the request should read the database from an executor thread"
        assert any(wrapper.connection is not None for wrapper in opened)

    assert all(wrapper.connection is None for wrapper in opened)
