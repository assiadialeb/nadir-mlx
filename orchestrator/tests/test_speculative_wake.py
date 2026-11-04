"""Tests for speculative discovery wake (MLX-126)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from django.test import TestCase

from orchestrator.gateway.route_cache import clear_gateway_route_cache
from orchestrator.lifecycle_selectors import LIFECYCLE_MODE_ON_DEMAND
from orchestrator.lifecycle_services import maybe_speculative_wake
from orchestrator.models import InferenceInstance


class SpeculativeWakeTests(TestCase):
    def setUp(self) -> None:
        clear_gateway_route_cache()

    @patch("orchestrator.lifecycle_services.ensure_instance_ready")
    def test_speculative_wake_starts_background_thread(
        self,
        mock_ensure: MagicMock,
    ) -> None:
        InferenceInstance.objects.create(
            model_name="preview-model",
            port=11450,
            launch_mode="TEXT",
            server_config={
                "model_id": "preview-chat",
                "host": "127.0.0.1",
                "ops": {
                    "lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND,
                    "speculative_wake": True,
                },
            },
            status="STOPPED",
        )
        maybe_speculative_wake("preview-chat")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and mock_ensure.call_count == 0:
            time.sleep(0.05)
        mock_ensure.assert_called_once_with("preview-chat")

    @patch("orchestrator.lifecycle_services.ensure_instance_ready")
    def test_speculative_wake_noop_when_disabled(self, mock_ensure: MagicMock) -> None:
        InferenceInstance.objects.create(
            model_name="quiet-model",
            port=11451,
            launch_mode="TEXT",
            server_config={
                "model_id": "quiet-chat",
                "ops": {
                    "lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND,
                    "speculative_wake": False,
                },
            },
            status="STOPPED",
        )
        maybe_speculative_wake("quiet-chat")
        time.sleep(0.2)
        mock_ensure.assert_not_called()
