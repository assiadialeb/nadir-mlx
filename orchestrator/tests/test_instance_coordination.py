"""Tests for cross-process slot locks, background transitions and shared runtime state."""

from __future__ import annotations

import importlib
import os
import threading
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, TestCase

from orchestrator.instance_locks import (
    InstanceSlotBusy,
    SlotLockHandoff,
    instance_slot_lock,
    is_slot_lock_held_elsewhere,
)
from orchestrator.models import BenchmarkRun, InferenceInstance


def _held_in_other_thread(port: int) -> bool:
    result: dict[str, bool] = {}

    def _probe() -> None:
        result["held"] = is_slot_lock_held_elsewhere(port)

    worker = threading.Thread(target=_probe)
    worker.start()
    worker.join()
    return result["held"]


class InstanceSlotLockTests(SimpleTestCase):
    def test_instance_slot_lock_is_reentrant_in_same_thread(self) -> None:
        with instance_slot_lock(12001):
            with instance_slot_lock(12001, blocking=False):
                self.assertFalse(is_slot_lock_held_elsewhere(12001))
            self.assertTrue(_held_in_other_thread(12001))
        self.assertFalse(_held_in_other_thread(12001))

    def test_instance_slot_lock_non_blocking_raises_when_busy(self) -> None:
        handoff = SlotLockHandoff(12002)
        try:
            with self.assertRaises(InstanceSlotBusy):
                with instance_slot_lock(12002, blocking=False):
                    pass
        finally:
            handoff.release()
        with instance_slot_lock(12002, blocking=False):
            pass

    def test_instance_slot_lock_blocking_times_out(self) -> None:
        handoff = SlotLockHandoff(12003)
        try:
            with self.assertRaises(InstanceSlotBusy):
                with instance_slot_lock(12003, timeout_seconds=0.2):
                    pass
        finally:
            handoff.release()

    def test_slot_lock_handoff_refuses_busy_slot(self) -> None:
        with instance_slot_lock(12004):
            with self.assertRaises(InstanceSlotBusy):
                SlotLockHandoff(12004)

    def test_slot_lock_handoff_makes_nested_locks_reentrant(self) -> None:
        handoff = SlotLockHandoff(12005)
        try:
            with handoff.held_by_current_thread():
                with instance_slot_lock(12005, blocking=False):
                    pass
        finally:
            handoff.release()
        handoff.release()
        self.assertFalse(is_slot_lock_held_elsewhere(12005))


class InstanceBackgroundTests(SimpleTestCase):
    @patch("orchestrator.instance_background.threading.Thread")
    @patch("orchestrator.instance_background.prepare_instance_launch")
    def test_start_instance_in_background_reserves_slot_then_spawns_worker(
        self,
        mock_prepare: MagicMock,
        mock_thread: MagicMock,
    ) -> None:
        from orchestrator.instance_background import start_instance_in_background
        from orchestrator.server_manager import InstanceLaunchPlan

        mock_prepare.return_value = InstanceLaunchPlan(
            instance_id=1,
            port=12010,
            model_path="/tmp/model",
            launch_mode="TEXT",
            requested_model_name="demo",
        )
        plan = start_instance_in_background("demo", 12010, "TEXT", {})
        self.assertEqual(plan.port, 12010)
        mock_thread.return_value.start.assert_called_once()
        self.assertTrue(is_slot_lock_held_elsewhere(12010))

        worker = mock_thread.call_args.kwargs["target"]
        with patch("orchestrator.instance_background.launch_prepared_instance") as mock_launch:
            with patch("django.db.close_old_connections"):
                worker()
        mock_launch.assert_called_once_with(plan)
        self.assertFalse(is_slot_lock_held_elsewhere(12010))

    @patch("orchestrator.instance_background.threading.Thread")
    @patch(
        "orchestrator.instance_background.prepare_instance_launch",
        side_effect=ValueError("bad model"),
    )
    def test_start_instance_in_background_releases_slot_on_validation_error(
        self,
        _mock_prepare: MagicMock,
        mock_thread: MagicMock,
    ) -> None:
        from orchestrator.instance_background import start_instance_in_background

        with self.assertRaises(ValueError):
            start_instance_in_background("demo", 12011, "TEXT", {})
        mock_thread.assert_not_called()
        self.assertFalse(is_slot_lock_held_elsewhere(12011))

    @patch("orchestrator.instance_background.threading.Thread")
    @patch("orchestrator.instance_background.prepare_instance_launch")
    def test_start_instance_in_background_noop_when_already_active(
        self,
        mock_prepare: MagicMock,
        mock_thread: MagicMock,
    ) -> None:
        from orchestrator.instance_background import start_instance_in_background
        from orchestrator.server_manager import InstanceLaunchPlan

        mock_prepare.return_value = InstanceLaunchPlan(
            instance_id=1,
            port=12012,
            model_path="/tmp/model",
            launch_mode="TEXT",
            requested_model_name="demo",
            already_active=True,
        )
        self.assertTrue(start_instance_in_background("demo", 12012, "TEXT", {}).already_active)
        mock_thread.assert_not_called()
        self.assertFalse(is_slot_lock_held_elsewhere(12012))

    def test_restart_instance_in_background_refuses_busy_slot(self) -> None:
        from orchestrator.instance_background import restart_instance_in_background

        with instance_slot_lock(12013):
            with self.assertRaises(InstanceSlotBusy):
                restart_instance_in_background(MagicMock(port=12013))


class RouteCacheGenerationTests(SimpleTestCase):
    @patch("orchestrator.gateway.selectors.build_route_snapshot_from_db")
    def test_get_route_snapshot_rebuilds_after_other_process_clears(
        self,
        mock_build: MagicMock,
    ) -> None:
        from orchestrator.gateway import route_cache

        mock_build.side_effect = lambda: object()
        first = route_cache.get_route_snapshot(force_refresh=True)
        self.assertIs(route_cache.get_route_snapshot(), first)

        # Another process only touches the shared generation file.
        route_cache._bump_route_generation()
        self.assertIsNot(route_cache.get_route_snapshot(), first)


class LanExposureSelectorTests(SimpleTestCase):
    def test_is_exposed_beyond_loopback(self) -> None:
        from orchestrator.lifecycle_selectors import is_exposed_beyond_loopback

        self.assertFalse(is_exposed_beyond_loopback({"host": "127.0.0.1"}))
        self.assertFalse(is_exposed_beyond_loopback({}))
        self.assertTrue(is_exposed_beyond_loopback({"host": "0.0.0.0"}))
        self.assertTrue(is_exposed_beyond_loopback({"host": "192.168.1.10"}))


class OrphanedBenchmarkTests(TestCase):
    def _run(self, status: str, params: dict) -> BenchmarkRun:
        return BenchmarkRun.objects.create(
            target_type="ENDPOINT",
            endpoint_url="http://127.0.0.1:12030/v1",
            status=status,
            params=params,
        )

    def test_fail_orphaned_benchmark_runs_marks_dead_runners_failed(self) -> None:
        from orchestrator.benchmark_service import fail_orphaned_benchmark_runs

        alive = self._run("RUNNING", {"runner_pid": os.getpid()})
        dead = self._run("RUNNING", {"runner_pid": 999_999_999})
        legacy = self._run("PENDING", {})
        done = self._run("COMPLETED", {"runner_pid": 999_999_999})

        self.assertEqual(fail_orphaned_benchmark_runs(), 2)
        for run, expected in ((alive, "RUNNING"), (dead, "FAILED"), (legacy, "FAILED"), (done, "COMPLETED")):
            run.refresh_from_db()
            self.assertEqual(run.status, expected)
        self.assertIn("Interrupted", dead.error_message)


class DesiredStateMigrationTests(TestCase):
    def test_seed_desired_state_and_clean_config(self) -> None:
        from django.apps import apps

        migration = importlib.import_module(
            "orchestrator.migrations.0015_inferenceinstance_desired_state"
        )
        running = InferenceInstance.objects.create(
            model_name="running-model",
            port=12020,
            status="RUNNING",
            server_config={"host": "0.0.0.0", "ops": {"manual_stop_in_progress": True}},
        )
        stopped = InferenceInstance.objects.create(
            model_name="stopped-model",
            port=12021,
            status="STOPPED",
            server_config={"host": "127.0.0.1"},
        )
        migration._seed_desired_state_and_clean_config(apps, None)
        running.refresh_from_db()
        stopped.refresh_from_db()
        self.assertEqual(running.desired_state, InferenceInstance.DESIRED_STATE_RUNNING)
        self.assertEqual(running.server_config["host"], "127.0.0.1")
        self.assertNotIn("manual_stop_in_progress", running.server_config["ops"])
        self.assertEqual(stopped.desired_state, InferenceInstance.DESIRED_STATE_STOPPED)
