"""Shared fixtures for OpenAPI contract tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from orchestrator.gateway.app import create_app
from orchestrator.gateway.route_cache import clear_gateway_route_cache
from orchestrator.gateway_telemetry_services import flush_gateway_timing_writes
from orchestrator.tests.contracts.validators import load_curated_spec


@pytest.fixture
def curated_spec() -> dict[str, Any]:
    return load_curated_spec()


@pytest.fixture
def gateway_client() -> Iterator[TestClient]:
    """Gateway test client that waits for its telemetry writes (MLX-137).

    Each gateway request queues a ``GatewayRequestMetric`` row for the
    ``nadir-gateway-telemetry`` writer thread. If the test ends while that
    thread is still inserting, the ``django_db(transaction=True)`` teardown
    flush hits ``database table is locked`` on the shared in-memory SQLite DB.
    Draining the queue first makes the teardown deterministic.

    The client is not entered as a context manager on purpose: the lifespan
    would build the shared httpx client before the tests patch
    ``httpx.AsyncClient``.
    """
    clear_gateway_route_cache()
    try:
        yield TestClient(create_app())
    finally:
        flush_gateway_timing_writes()
