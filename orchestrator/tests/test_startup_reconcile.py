"""Tests for always_on restore after process/host restart."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from orchestrator.models import InferenceInstance
from orchestrator.startup_reconcile import (
    _should_restore_on_startup,
    reconcile_instances_on_startup,
    start_startup_reconcile_if_needed,
)


class StartupReconcilePolicyTests(TestCase):
    def test_should_restore_always_on_stopped(self) -> None:
        instance = InferenceInstance(
            status="STOPPED",
            server_config={"ops": {"lifecycle_mode": "always_on"}},
        )
        self.assertTrue(_should_restore_on_startup(instance))

    def test_should_not_restore_on_demand(self) -> None:
        instance = InferenceInstance(
            status="STOPPED",
            server_config={"ops": {"lifecycle_mode": "on_demand"}},
        )
        self.assertFalse(_should_restore_on_startup(instance))

    def test_should_not_restore_running(self) -> None:
        instance = InferenceInstance(
            status="RUNNING",
            server_config={"ops": {"lifecycle_mode": "always_on"}},
        )
        self.assertFalse(_should_restore_on_startup(instance))

    @patch(
        "orchestrator.startup_reconcile.is_manual_stop_in_progress",
        return_value=True,
    )
    def test_should_not_restore_during_manual_stop(
        self,
        _mock_manual: MagicMock,
    ) -> None:
        instance = InferenceInstance(
            status="STOPPED",
            server_config={"ops": {"lifecycle_mode": "always_on"}},
        )
        self.assertFalse(_should_restore_on_startup(instance))

    def test_default_lifecycle_is_always_on(self) -> None:
        instance = InferenceInstance(status="FAILED", server_config={})
        self.assertTrue(_should_restore_on_startup(instance))


class StartupReconcileCycleTests(TestCase):
    def setUp(self) -> None:
        self.always_on = InferenceInstance.objects.create(
            model_name="always-on-model",
            port=11460,
            launch_mode="TEXT",
            server_config={
                "model_id": "always-chat",
                "ops": {"lifecycle_mode": "always_on"},
            },
            status="STOPPED",
        )
        self.on_demand = InferenceInstance.objects.create(
            model_name="on-demand-model",
            port=11461,
            launch_mode="TEXT",
            server_config={
                "model_id": "demand-chat",
                "ops": {"lifecycle_mode": "on_demand"},
            },
            status="STOPPED",
        )

    @patch("orchestrator.startup_reconcile.restart_instance")
    @patch("orchestrator.startup_reconcile.refresh_all_instance_health")
    def test_reconcile_restarts_always_on_only(
        self,
        mock_refresh: MagicMock,
        mock_restart: MagicMock,
    ) -> None:
        result = reconcile_instances_on_startup()
        mock_refresh.assert_called_once()
        mock_restart.assert_called_once()
        self.assertEqual(mock_restart.call_args.args[0].id, self.always_on.id)
        self.assertEqual(result["restored"], 1)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["failed"], 0)

    @patch(
        "orchestrator.startup_reconcile.restart_instance",
        side_effect=RuntimeError("boom"),
    )
    @patch("orchestrator.startup_reconcile.refresh_all_instance_health")
    def test_reconcile_counts_restart_failures(
        self,
        _mock_refresh: MagicMock,
        _mock_restart: MagicMock,
    ) -> None:
        result = reconcile_instances_on_startup()
        self.assertEqual(result["restored"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["skipped"], 1)


class StartupReconcileThreadTests(TestCase):
    def setUp(self) -> None:
        import orchestrator.startup_reconcile as module

        module._reconcile_started = False

    @override_settings(INSTANCE_STARTUP_RECONCILE_ENABLED=False)
    def test_start_respects_disabled_setting(self) -> None:
        with patch("threading.Thread") as mock_thread:
            start_startup_reconcile_if_needed()
        mock_thread.assert_not_called()

    @patch("orchestrator.startup_reconcile.should_skip_watchdog", return_value=True)
    def test_start_skips_when_watchdog_blocked(
        self,
        _mock_skip: MagicMock,
    ) -> None:
        with patch("threading.Thread") as mock_thread:
            start_startup_reconcile_if_needed()
        mock_thread.assert_not_called()

    @patch("orchestrator.startup_reconcile.should_skip_watchdog", return_value=False)
    @patch.dict(os.environ, {"RUN_MAIN": "true"}, clear=False)
    @patch("sys.argv", ["manage.py", "runserver"])
    def test_start_schedules_daemon_thread_once(
        self,
        _mock_skip: MagicMock,
    ) -> None:
        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            start_startup_reconcile_if_needed()
            start_startup_reconcile_if_needed()
        mock_thread.assert_called_once()
        kwargs = mock_thread.call_args.kwargs
        self.assertTrue(kwargs["daemon"])
        self.assertEqual(kwargs["name"], "mlx-startup-reconcile")
