"""Shared fixtures for OpenAPI contract tests."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from orchestrator.gateway.app import create_app
from orchestrator.gateway.route_cache import clear_gateway_route_cache
from orchestrator.tests.contracts.validators import load_curated_spec


@pytest.fixture
def curated_spec() -> dict[str, Any]:
    return load_curated_spec()


@pytest.fixture
def gateway_client() -> TestClient:
    clear_gateway_route_cache()
    return TestClient(create_app())
