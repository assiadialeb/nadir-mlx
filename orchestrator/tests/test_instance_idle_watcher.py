"""Tests for idle offload watcher (MLX-43)."""

import os
from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.utils import timezone

from orchestrator.instance_idle_watcher import (
    _attempt_idle_offload,
    idle_check_interval_seconds,
    idle_offload_enabled,
    run_idle_offload_cycle,
)
from orchestrator.lifecycle_selectors import LIFECYCLE_MODE_ON_DEMAND
from orchestrator.models import InferenceInstance


class IdleWatcherSettingsTests(TestCase):
    def test_idle_offload_enabled_reads_env(self) -> None:
        with patch.dict(os.environ, {"NADIR_IDLE_OFFLOAD_ENABLED": "false"}, clear=False):
            self.assertFalse(idle_offload_enabled())

    def test_idle_check_interval_reads_env(self) -> None:
        with patch.dict(os.environ, {"NADIR_IDLE_CHECK_INTERVAL_SECONDS": "30"}, clear=False):
            self.assertEqual(idle_check_interval_seconds(), 30.0)


class IdleOffloadAttemptTests(TestCase):
    @patch("orchestrator.instance_idle_watcher.stop_instance")
    def test_stops_idle_on_demand_instance(self, mock_stop: MagicMock) -> None:
        instance = InferenceInstance.objects.create(
            model_name="idle-model",
            port=11430,
            launch_mode="TEXT",
            server_config={
                "model_id": "idle-chat",
                "host": "127.0.0.1",
                "ops": {
                    "lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND,
                    "idle_minutes": 5,
                },
            },
            status="RUNNING",
            last_used_at=timezone.now() - timedelta(minutes=10),
        )
        _attempt_idle_offload(instance)
        mock_stop.assert_called_once()
        stopped = mock_stop.call_args[0][0]
        self.assertEqual(stopped.pk, instance.pk)

    @patch("orchestrator.instance_idle_watcher.stop_instance")
    def test_skips_recent_activity(self, mock_stop: MagicMock) -> None:
        InferenceInstance.objects.create(
            model_name="active-model",
            port=11431,
            launch_mode="TEXT",
            server_config={
                "model_id": "active-chat",
                "ops": {
                    "lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND,
                    "idle_minutes": 30,
                },
            },
            status="RUNNING",
            last_used_at=timezone.now(),
        )
        _attempt_idle_offload(
            InferenceInstance.objects.get(model_name="active-model"),
        )
        mock_stop.assert_not_called()

    @patch("orchestrator.instance_idle_watcher.stop_instance")
    def test_skips_always_on_instance(self, mock_stop: MagicMock) -> None:
        InferenceInstance.objects.create(
            model_name="always-on-model",
            port=11432,
            launch_mode="TEXT",
            server_config={"model_id": "always-chat"},
            status="RUNNING",
            last_used_at=timezone.now() - timedelta(hours=2),
        )
        _attempt_idle_offload(
            InferenceInstance.objects.get(model_name="always-on-model"),
        )
        mock_stop.assert_not_called()

    @patch("orchestrator.instance_idle_watcher.is_wake_in_progress", return_value=True)
    @patch("orchestrator.instance_idle_watcher.stop_instance")
    def test_skips_when_wake_in_progress(
        self,
        mock_stop: MagicMock,
        _mock_wake: MagicMock,
    ) -> None:
        instance = InferenceInstance.objects.create(
            model_name="waking-model",
            port=11433,
            launch_mode="TEXT",
            server_config={
                "model_id": "waking-chat",
                "ops": {"lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND, "idle_minutes": 5},
            },
            status="RUNNING",
            last_used_at=timezone.now() - timedelta(minutes=10),
        )
        _attempt_idle_offload(instance)
        mock_stop.assert_not_called()

    @patch("orchestrator.instance_idle_watcher.stop_instance")
    def test_skips_manual_stop_in_progress(self, mock_stop: MagicMock) -> None:
        instance = InferenceInstance.objects.create(
            model_name="manual-stop-model",
            port=11434,
            launch_mode="TEXT",
            server_config={
                "model_id": "manual-chat",
                "ops": {
                    "lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND,
                    "idle_minutes": 5,
                    "manual_stop_in_progress": True,
                },
            },
            status="RUNNING",
            last_used_at=timezone.now() - timedelta(minutes=10),
        )
        _attempt_idle_offload(instance)
        mock_stop.assert_not_called()

    @patch("orchestrator.instance_idle_watcher.stop_instance")
    def test_rechecks_activity_before_stop(self, mock_stop: MagicMock) -> None:
        instance = InferenceInstance.objects.create(
            model_name="race-model",
            port=11435,
            launch_mode="TEXT",
            server_config={
                "model_id": "race-chat",
                "ops": {"lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND, "idle_minutes": 5},
            },
            status="RUNNING",
            last_used_at=timezone.now() - timedelta(minutes=10),
        )

        def _refresh_and_touch() -> None:
            instance.last_used_at = timezone.now()
            instance.save(update_fields=["last_used_at"])

        with patch.object(InferenceInstance, "refresh_from_db", side_effect=_refresh_and_touch):
            _attempt_idle_offload(instance)

        mock_stop.assert_not_called()


class RunIdleOffloadCycleTests(TestCase):
    @patch("orchestrator.instance_idle_watcher.idle_offload_enabled", return_value=False)
    @patch("orchestrator.instance_idle_watcher._attempt_idle_offload")
    def test_cycle_noop_when_disabled(
        self,
        mock_attempt: MagicMock,
        _mock_enabled: MagicMock,
    ) -> None:
        run_idle_offload_cycle()
        mock_attempt.assert_not_called()

    @patch("orchestrator.instance_idle_watcher._attempt_idle_offload")
    def test_cycle_evaluates_running_instances(self, mock_attempt: MagicMock) -> None:
        InferenceInstance.objects.create(
            model_name="cycle-model",
            port=11436,
            launch_mode="TEXT",
            server_config={"model_id": "cycle-chat"},
            status="RUNNING",
        )
        InferenceInstance.objects.create(
            model_name="cycle-stopped",
            port=11437,
            launch_mode="TEXT",
            server_config={"model_id": "cycle-stopped-chat"},
            status="STOPPED",
        )
        run_idle_offload_cycle()
        self.assertEqual(mock_attempt.call_count, 1)
