"""Tests for the gateway FastAPI application."""

import os
from unittest.mock import patch

from django.test import TestCase
from fastapi.testclient import TestClient

from orchestrator.gateway.app import create_app


class GatewayAppTests(TestCase):
    def test_health_endpoint_returns_ok(self) -> None:
        client = TestClient(create_app())
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "service": "nadir-gateway",
                "docs": "/docs",
                "models": "/v1/models",
            },
        )

    @patch.dict(os.environ, {"NADIR_GATEWAY_API_KEY": "secret-key"}, clear=False)
    def test_protected_route_requires_gateway_api_key(self) -> None:
        client = TestClient(create_app())
        unauthorized = client.get("/v1/models")
        self.assertEqual(unauthorized.status_code, 401)

        authorized = client.get("/v1/models", headers={"X-API-Key": "secret-key"})
        self.assertEqual(authorized.status_code, 200)
