"""Tests for optional gateway API-key authentication."""

from __future__ import annotations

import os
from unittest.mock import patch

from django.test import SimpleTestCase

from orchestrator.gateway.auth import verify_gateway_api_key
from orchestrator.gateway.router import GatewayRouteError


class GatewayAuthTests(SimpleTestCase):
    def test_verify_gateway_api_key_allows_when_unconfigured(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NADIR_GATEWAY_API_KEY", None)
            verify_gateway_api_key(authorization=None, x_api_key=None)

    def test_verify_gateway_api_key_accepts_bearer_token(self) -> None:
        with patch.dict(os.environ, {"NADIR_GATEWAY_API_KEY": "secret"}, clear=False):
            verify_gateway_api_key(authorization="Bearer secret", x_api_key=None)

    def test_verify_gateway_api_key_accepts_x_api_key_header(self) -> None:
        with patch.dict(os.environ, {"NADIR_GATEWAY_API_KEY": "secret"}, clear=False):
            verify_gateway_api_key(authorization=None, x_api_key="secret")

    def test_verify_gateway_api_key_rejects_invalid_token(self) -> None:
        with patch.dict(os.environ, {"NADIR_GATEWAY_API_KEY": "secret"}, clear=False):
            with self.assertRaises(GatewayRouteError) as ctx:
                verify_gateway_api_key(authorization="Bearer wrong", x_api_key=None)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.code, "invalid_api_key")

    def test_verify_gateway_api_key_rejects_length_mismatch(self) -> None:
        with patch.dict(os.environ, {"NADIR_GATEWAY_API_KEY": "secret"}, clear=False):
            with self.assertRaises(GatewayRouteError) as ctx:
                verify_gateway_api_key(authorization="Bearer se", x_api_key=None)
        self.assertEqual(ctx.exception.code, "invalid_api_key")

    def test_verify_gateway_api_key_treats_blank_env_as_unconfigured(self) -> None:
        with patch.dict(os.environ, {"NADIR_GATEWAY_API_KEY": "  "}, clear=False):
            verify_gateway_api_key(authorization=None, x_api_key=None)
