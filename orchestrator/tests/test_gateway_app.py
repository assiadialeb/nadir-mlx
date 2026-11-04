"""Tests for the gateway FastAPI application."""

import os
from unittest.mock import patch

from django.test import SimpleTestCase
from fastapi.testclient import TestClient

from orchestrator.gateway.app import create_app
from orchestrator.gateway.router import GatewayRouteError


class GatewayAppTests(SimpleTestCase):
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
        self.assertTrue(response.headers.get("x-request-id"))

    def test_request_id_propagates_incoming_header(self) -> None:
        client = TestClient(create_app())
        response = client.get("/health", headers={"X-Request-Id": "req-fixed-1"})
        self.assertEqual(response.headers.get("x-request-id"), "req-fixed-1")

    @patch.dict(os.environ, {"NADIR_GATEWAY_API_KEY": "secret-key"}, clear=False)
    @patch(
        "orchestrator.gateway.routes.models.list_running_gateway_models",
        return_value={"object": "list", "data": []},
    )
    def test_protected_route_requires_gateway_api_key(self, _mock_models) -> None:
        client = TestClient(create_app())
        unauthorized = client.get("/v1/models")
        self.assertEqual(unauthorized.status_code, 401)
        error = unauthorized.json()["error"]
        self.assertEqual(error["type"], "authentication_error")
        self.assertEqual(error["code"], "invalid_api_key")
        self.assertNotIn("detail", unauthorized.json())

        authorized = client.get("/v1/models", headers={"X-API-Key": "secret-key"})
        self.assertEqual(authorized.status_code, 200)

    def test_unknown_route_returns_openai_error_envelope(self) -> None:
        client = TestClient(create_app())
        response = client.get("/v1/does-not-exist")
        self.assertEqual(response.status_code, 404)
        payload = response.json()
        self.assertNotIn("detail", payload)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["code"], "invalid_request")

    @patch(
        "orchestrator.gateway.routes.models.get_gateway_model",
        return_value={
            "id": "demo-chat",
            "object": "model",
            "created": 1,
            "owned_by": "nadir",
        },
    )
    def test_retrieve_model_returns_single_object(self, _mock_get) -> None:
        client = TestClient(create_app())
        response = client.get("/v1/models/demo-chat")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "demo-chat")

    @patch(
        "orchestrator.gateway.routes.models.get_gateway_model",
        side_effect=GatewayRouteError(
            status_code=404,
            code="model_not_found",
            message="No inference instance is registered for alias 'missing'.",
        ),
    )
    def test_retrieve_unknown_model_returns_openai_error(self, _mock_get) -> None:
        client = TestClient(create_app())
        response = client.get("/v1/models/missing")
        self.assertEqual(response.status_code, 404)
        error = response.json()["error"]
        self.assertEqual(error["type"], "invalid_request_error")
        self.assertEqual(error["code"], "model_not_found")

