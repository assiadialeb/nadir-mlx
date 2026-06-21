"""Tests for the gateway FastAPI application."""

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
            {"status": "ok", "service": "nadir-gateway"},
        )
