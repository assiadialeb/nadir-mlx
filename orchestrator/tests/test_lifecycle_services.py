"""Tests for lifecycle wake services (MLX-40)."""

import os
from unittest.mock import MagicMock, patch

from django.test import TestCase

from orchestrator.gateway.router import GatewayRouteError
from orchestrator.gateway.route_cache import clear_gateway_route_cache
from orchestrator.lifecycle_selectors import LIFECYCLE_MODE_ON_DEMAND
from orchestrator.lifecycle_services import (
    ensure_instance_ready,
    gateway_wake_poll_interval_seconds,
    gateway_wake_timeout_seconds,
    is_wake_in_progress,
)
from orchestrator.models import InferenceInstance


class LifecycleServicesSettingsTests(TestCase):
    def test_gateway_wake_timeout_reads_env(self) -> None:
        with patch.dict(os.environ, {"NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS": "120"}, clear=False):
            self.assertEqual(gateway_wake_timeout_seconds(), 120.0)

    def test_gateway_wake_poll_interval_reads_env(self) -> None:
        with patch.dict(os.environ, {"NADIR_GATEWAY_WAKE_POLL_INTERVAL_SECONDS": "0.5"}, clear=False):
            self.assertEqual(gateway_wake_poll_interval_seconds(), 0.5)


class EnsureInstanceReadyTests(TestCase):
    def setUp(self) -> None:
        clear_gateway_route_cache()

    def test_unknown_alias_returns_404(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            ensure_instance_ready("missing-alias")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.code, "model_not_found")

    @patch("orchestrator.lifecycle_services.probe_http_health", return_value=True)
    def test_running_healthy_returns_target_immediately(
        self,
        _mock_probe: MagicMock,
    ) -> None:
        instance = InferenceInstance.objects.create(
            model_name="ready-model",
            port=11410,
            launch_mode="TEXT",
            server_config={"model_id": "ready-chat", "host": "127.0.0.1"},
            status="RUNNING",
        )
        target = ensure_instance_ready("ready-chat")
        self.assertEqual(target.instance_id, instance.pk)
        self.assertEqual(target.alias, "ready-chat")

    @patch("orchestrator.lifecycle_services.start_instance")
    @patch("orchestrator.lifecycle_services.probe_http_health", return_value=True)
    def test_on_demand_stopped_wakes_instance(
        self,
        _mock_probe: MagicMock,
        mock_start: MagicMock,
    ) -> None:
        instance = InferenceInstance.objects.create(
            model_name="sleeping-model",
            port=11411,
            launch_mode="TEXT",
            server_config={
                "model_id": "sleeping-chat",
                "host": "127.0.0.1",
                "ops": {"lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND},
            },
            status="STOPPED",
        )

        def _mark_running(*_args: object, **_kwargs: object) -> InferenceInstance:
            instance.status = "RUNNING"
            instance.pid = 4242
            instance.save(update_fields=["status", "pid"])
            return instance

        mock_start.side_effect = _mark_running

        target = ensure_instance_ready("sleeping-chat")
        self.assertEqual(target.instance_id, instance.pk)
        mock_start.assert_called_once_with(
            "sleeping-model",
            11411,
            "TEXT",
            instance.server_config,
        )

    @patch("orchestrator.lifecycle_services.start_instance")
    def test_always_on_stopped_does_not_wake(self, mock_start: MagicMock) -> None:
        InferenceInstance.objects.create(
            model_name="offline-model",
            port=11412,
            launch_mode="TEXT",
            server_config={"model_id": "offline-chat", "host": "127.0.0.1"},
            status="STOPPED",
        )
        with self.assertRaises(GatewayRouteError) as ctx:
            ensure_instance_ready("offline-chat")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.code, "model_unavailable")
        mock_start.assert_not_called()

    @patch("orchestrator.lifecycle_services.time.sleep")
    @patch("orchestrator.lifecycle_services.probe_http_health", return_value=False)
    def test_loading_timeout_marks_instance_stopped(
        self,
        _mock_probe: MagicMock,
        _mock_sleep: MagicMock,
    ) -> None:
        instance = InferenceInstance.objects.create(
            model_name="slow-model",
            port=11413,
            launch_mode="TEXT",
            server_config={
                "model_id": "slow-chat",
                "host": "127.0.0.1",
                "ops": {"lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND},
            },
            status="LOADING",
        )
        env = {
            "NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS": "0.2",
            "NADIR_GATEWAY_WAKE_POLL_INTERVAL_SECONDS": "0.05",
        }
        with patch.dict(os.environ, env, clear=False):
            with self.assertRaises(GatewayRouteError) as ctx:
                ensure_instance_ready("slow-chat")
        self.assertEqual(ctx.exception.code, "model_waking_timeout")
        instance.refresh_from_db()
        self.assertEqual(instance.status, "STOPPED")

    @patch("orchestrator.lifecycle_services.start_instance")
    def test_wake_lock_blocks_concurrent_wake(self, mock_start: MagicMock) -> None:
        InferenceInstance.objects.create(
            model_name="locked-model",
            port=11414,
            launch_mode="TEXT",
            server_config={
                "model_id": "locked-chat",
                "ops": {"lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND},
            },
            status="STOPPED",
        )
        lock = __import__(
            "orchestrator.lifecycle_services",
            fromlist=["_wake_lock_for_alias"],
        )._wake_lock_for_alias("locked-chat")
        lock.acquire()
        try:
            self.assertTrue(is_wake_in_progress("locked-chat"))
        finally:
            lock.release()

        mock_start.assert_not_called()
