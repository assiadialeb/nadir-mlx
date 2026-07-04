"""Tests for background health polling and auto-restart (instance_watchdog)."""

from __future__ import annotations

import os
from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings
from django.utils import timezone

from orchestrator.instance_watchdog import (
    _attempt_auto_restart,
    _auto_restart_enabled,
    _is_restart_frozen,
    _max_restart_retries,
    _restart_backoff_seconds,
    run_watchdog_cycle,
    start_watchdog_if_needed,
)
from orchestrator.models import InferenceInstance


class InstanceWatchdogHelperTests(TestCase):
    def test_auto_restart_enabled_reads_ops_and_legacy_flag(self) -> None:
        instance = InferenceInstance(
            server_config={"ops": {"auto_restart": True}},
        )
        self.assertTrue(_auto_restart_enabled(instance))

        legacy = InferenceInstance(server_config={"auto_restart": True, "ops": {}})
        self.assertTrue(_auto_restart_enabled(legacy))

        disabled = InferenceInstance(server_config={"ops": {"auto_restart": False}})
        self.assertFalse(_auto_restart_enabled(disabled))

    def test_max_restart_retries_defaults_on_invalid_value(self) -> None:
        instance = InferenceInstance(
            server_config={"ops": {"auto_restart_max_retries": "bad"}},
        )
        self.assertEqual(_max_restart_retries(instance), 3)

    @override_settings(INSTANCE_AUTO_RESTART_BACKOFF_SECONDS=30)
    def test_restart_backoff_seconds_caps_at_ten_minutes(self) -> None:
        self.assertEqual(_restart_backoff_seconds(1), 30)
        self.assertEqual(_restart_backoff_seconds(5), 480)
        self.assertEqual(_restart_backoff_seconds(10), 600)

    def test_is_restart_frozen_honors_restart_frozen_until(self) -> None:
        instance = InferenceInstance(
            server_config={
                "ops": {
                    "restart_frozen_until": (
                        timezone.now() + timedelta(minutes=5)
                    ).isoformat(),
                },
            },
        )
        self.assertTrue(_is_restart_frozen(instance))

    def test_is_restart_frozen_ignores_invalid_timestamp(self) -> None:
        instance = InferenceInstance(
            server_config={"ops": {"restart_frozen_until": "not-a-date"}},
        )
        self.assertFalse(_is_restart_frozen(instance))


class InstanceWatchdogAutoRestartTests(TestCase):
    def setUp(self) -> None:
        self.instance = InferenceInstance.objects.create(
            model_name="watchdog-model",
            port=11455,
            launch_mode="TEXT",
            server_config={
                "model_id": "watchdog-chat",
                "ops": {"auto_restart": True, "auto_restart_max_retries": 2},
            },
            status="FAILED",
        )

    @patch("orchestrator.instance_watchdog.restart_instance")
    def test_attempt_auto_restart_restarts_failed_instance(
        self,
        mock_restart: MagicMock,
    ) -> None:
        _attempt_auto_restart(self.instance)
        mock_restart.assert_called_once()
        self.instance.refresh_from_db()
        self.assertEqual(self.instance.server_config["ops"]["restart_attempts"], 1)

    @patch("orchestrator.instance_watchdog.restart_instance", side_effect=RuntimeError("boom"))
    def test_attempt_auto_restart_records_failure_and_freeze(
        self,
        _mock_restart: MagicMock,
    ) -> None:
        _attempt_auto_restart(self.instance)
        self.instance.refresh_from_db()
        ops = self.instance.server_config["ops"]
        self.assertEqual(ops["restart_attempts"], 1)
        self.assertIn("last_restart_error", ops)
        self.assertIn("restart_frozen_until", ops)

    @patch("orchestrator.instance_watchdog.restart_instance")
    def test_attempt_auto_restart_freezes_after_max_retries(
        self,
        mock_restart: MagicMock,
    ) -> None:
        config = dict(self.instance.server_config)
        config["ops"] = {
            **config["ops"],
            "restart_attempts": 2,
        }
        self.instance.server_config = config
        self.instance.save(update_fields=["server_config"])

        _attempt_auto_restart(self.instance)
        mock_restart.assert_not_called()
        self.instance.refresh_from_db()
        self.assertIn("restart_frozen_until", self.instance.server_config["ops"])

    @patch("orchestrator.instance_watchdog.is_manual_stop_in_progress", return_value=True)
    @patch("orchestrator.instance_watchdog.restart_instance")
    def test_attempt_auto_restart_skips_manual_stop(
        self,
        mock_restart: MagicMock,
        _mock_manual: MagicMock,
    ) -> None:
        _attempt_auto_restart(self.instance)
        mock_restart.assert_not_called()


class InstanceWatchdogCycleTests(TestCase):
    @patch("orchestrator.instance_watchdog._attempt_auto_restart")
    @patch("orchestrator.instance_watchdog.refresh_all_instance_health")
    def test_run_watchdog_cycle_refreshes_health_and_restarts(
        self,
        mock_refresh: MagicMock,
        mock_restart: MagicMock,
    ) -> None:
        InferenceInstance.objects.create(
            model_name="stopped-model",
            port=11456,
            launch_mode="TEXT",
            server_config={"model_id": "stopped"},
            status="STOPPED",
        )
        run_watchdog_cycle()
        mock_refresh.assert_called_once()
        mock_restart.assert_called_once()


class InstanceWatchdogStartupTests(TestCase):
    def setUp(self) -> None:
        import orchestrator.instance_watchdog as module

        module._watchdog_started = False

    @override_settings(INSTANCE_WATCHDOG_ENABLED=False)
    def test_start_watchdog_if_needed_respects_disabled_setting(self) -> None:
        with patch("threading.Thread") as mock_thread:
            start_watchdog_if_needed()
        mock_thread.assert_not_called()

    @patch("orchestrator.instance_watchdog.should_skip_watchdog", return_value=True)
    def test_start_watchdog_if_needed_skips_when_watchdog_blocked(
        self,
        _mock_skip: MagicMock,
    ) -> None:
        with patch("threading.Thread") as mock_thread:
            start_watchdog_if_needed()
        mock_thread.assert_not_called()

    @patch("orchestrator.instance_watchdog.should_skip_watchdog", return_value=False)
    @patch.dict(os.environ, {"RUN_MAIN": "true"}, clear=False)
    @patch("sys.argv", ["manage.py", "runserver"])
    def test_start_watchdog_if_needed_starts_daemon_thread_once(
        self,
        _mock_skip: MagicMock,
    ) -> None:
        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            start_watchdog_if_needed()
            start_watchdog_if_needed()
        mock_thread.assert_called_once()
        kwargs = mock_thread.call_args.kwargs
        self.assertTrue(kwargs["daemon"])
        self.assertEqual(kwargs["name"], "mlx-instance-watchdog")
