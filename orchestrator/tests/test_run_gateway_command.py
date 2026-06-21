"""Tests for run_gateway management command."""

from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase


class RunGatewayCommandTests(TestCase):
    @patch("orchestrator.gateway.__main__.main")
    def test_run_gateway_invokes_gateway_main(self, mock_main) -> None:
        out = StringIO()
        call_command("run_gateway", stdout=out)
        mock_main.assert_called_once()
        self.assertIn("Starting Nadir Gateway", out.getvalue())
