"""Tests for shared gateway client auth headers."""

from __future__ import annotations

import os
from unittest.mock import patch

from django.test import SimpleTestCase

from orchestrator.gateway_client_auth import gateway_api_key, gateway_auth_headers


class GatewayClientAuthTests(SimpleTestCase):
    def test_empty_when_unset(self) -> None:
        with patch.dict(os.environ, {"NADIR_GATEWAY_API_KEY": ""}, clear=False):
            self.assertEqual(gateway_api_key(), "")
            self.assertEqual(gateway_auth_headers(), {})

    def test_bearer_header_when_configured(self) -> None:
        with patch.dict(os.environ, {"NADIR_GATEWAY_API_KEY": "secret"}, clear=False):
            self.assertEqual(gateway_api_key(), "secret")
            self.assertEqual(
                gateway_auth_headers(),
                {"Authorization": "Bearer secret"},
            )
