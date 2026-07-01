"""Tests for shared FastAPI OpenAPI helpers."""

from django.test import SimpleTestCase
from fastapi import HTTPException

from orchestrator.fastapi_openapi import (
    InferenceApiError,
    open_api_responses,
    to_http_exception,
)


class FastApiOpenApiTests(SimpleTestCase):
    def test_open_api_responses_builds_description_map(self) -> None:
        responses = open_api_responses(400, 503)
        self.assertEqual(responses[400]["description"], "Invalid request")
        self.assertEqual(responses[503]["description"], "Service unavailable")

    def test_to_http_exception_maps_inference_api_error(self) -> None:
        error = InferenceApiError(400, "bad input")
        http_error = to_http_exception(error)
        self.assertIsInstance(http_error, HTTPException)
        self.assertEqual(http_error.status_code, 400)
        self.assertEqual(http_error.detail, "bad input")
