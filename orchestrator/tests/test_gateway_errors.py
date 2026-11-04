"""Tests for OpenAI-compatible gateway error envelopes."""

from django.test import SimpleTestCase

from orchestrator.gateway.errors import openai_error_envelope
from orchestrator.gateway.router import GatewayRouteError


class GatewayErrorEnvelopeTests(SimpleTestCase):
    def test_model_not_found_uses_invalid_request_type_and_param(self) -> None:
        payload = openai_error_envelope("missing", "model_not_found")
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["code"], "model_not_found")
        self.assertEqual(payload["error"]["param"], "model")

    def test_route_error_to_openai_error_matches_envelope(self) -> None:
        exc = GatewayRouteError(
            status_code=404,
            code="model_not_found",
            message="No alias",
        )
        self.assertEqual(exc.to_openai_error()["error"]["type"], "invalid_request_error")
