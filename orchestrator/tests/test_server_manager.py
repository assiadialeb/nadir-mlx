"""Unit tests for inference instance lifecycle helpers."""

import signal
from unittest import TestCase
from unittest.mock import MagicMock, patch

from django.test import TestCase as DjangoTestCase

from orchestrator.server_manager import (
    _collect_stop_targets,
    _ensure_port_released,
    _find_reusable_instance,
    _force_stop_pids,
    _is_process_alive,
    _resolve_server_config,
    is_manual_stop_in_progress,
    stop_instance,
)
from orchestrator.models import InferenceInstance


class ServerManagerStopTests(TestCase):
    def test_is_process_alive_returns_false_for_invalid_pid(self) -> None:
        self.assertFalse(_is_process_alive(0))
        self.assertFalse(_is_process_alive(-1))

    @patch("orchestrator.server_manager.os.kill", side_effect=ProcessLookupError)
    def test_is_process_alive_returns_false_when_missing(self, _mock_kill: MagicMock) -> None:
        self.assertFalse(_is_process_alive(4242))

    @patch("orchestrator.server_manager._find_listener_pids", return_value=[200])
    @patch("orchestrator.server_manager._find_orchestrator_launcher_pids", return_value=[100])
    def test_collect_stop_targets_includes_port_and_pid_tree(
        self,
        _mock_launchers: MagicMock,
        _mock_listeners: MagicMock,
    ) -> None:
        instance = MagicMock(pid=50, port=11400)
        with patch(
            "orchestrator.server_manager._collect_descendant_pids",
            return_value={50, 51},
        ):
            targets = _collect_stop_targets(instance)
        self.assertEqual(targets, {50, 51, 100, 200})

    @patch("orchestrator.server_manager._is_process_alive", return_value=False)
    @patch("orchestrator.server_manager._terminate_pid")
    def test_force_stop_pids_uses_sigterm_then_stops(
        self,
        mock_terminate: MagicMock,
        _mock_alive: MagicMock,
    ) -> None:
        survivors = _force_stop_pids({123}, grace_seconds=0.0)
        self.assertEqual(survivors, set())
        mock_terminate.assert_called_once_with(123, signal.SIGTERM)

    @patch("orchestrator.server_manager._ensure_port_released", return_value=[])
    @patch("orchestrator.server_manager._force_stop_pids", return_value=set())
    @patch("orchestrator.server_manager._collect_stop_targets", return_value={999})
    @patch("orchestrator.server_manager._set_manual_stop_in_progress")
    def test_stop_instance_marks_instance_stopped_when_cleanup_succeeds(
        self,
        mock_stop_flag: MagicMock,
        _mock_targets: MagicMock,
        _mock_force_stop: MagicMock,
        _mock_ensure_port: MagicMock,
    ) -> None:
        instance = MagicMock(pid=999, port=11400, status="RUNNING", server_config={"ops": {}})
        stop_instance(instance)
        self.assertIsNone(instance.pid)
        self.assertEqual(instance.status, "STOPPED")
        instance.save.assert_called_once()
        mock_stop_flag.assert_any_call(instance, active=True)
        self.assertFalse(is_manual_stop_in_progress(instance))

    @patch("orchestrator.server_manager.is_port_free", side_effect=[False, False, True])
    @patch("orchestrator.server_manager._port_blocker_pids", return_value=[])
    @patch("orchestrator.server_manager.time.sleep")
    @patch("orchestrator.server_manager.time.time")
    def test_ensure_port_released_waits_for_kernel_delay(
        self,
        mock_time: MagicMock,
        _mock_sleep: MagicMock,
        _mock_blockers: MagicMock,
        _mock_port_free: MagicMock,
    ) -> None:
        mock_time.side_effect = [0.0, 0.3, 0.6, 1.0]
        remaining = _ensure_port_released(11446, timeout_seconds=2.0)
        self.assertEqual(remaining, [])

    @patch("orchestrator.server_manager._ensure_port_released", return_value=[4242])
    @patch("orchestrator.server_manager._force_stop_pids", return_value=set())
    @patch("orchestrator.server_manager._collect_stop_targets", return_value=set())
    @patch("orchestrator.server_manager._set_manual_stop_in_progress")
    def test_stop_instance_clears_manual_stop_flag_on_failure(
        self,
        mock_stop_flag: MagicMock,
        _mock_targets: MagicMock,
        _mock_force_stop: MagicMock,
        _mock_ensure_port: MagicMock,
    ) -> None:
        instance = MagicMock(pid=999, port=11446, status="RUNNING", server_config={"ops": {}})
        with self.assertRaises(RuntimeError):
            stop_instance(instance)
        mock_stop_flag.assert_any_call(instance, active=False)


class ServerManagerReuseTests(DjangoTestCase):
    def test_find_reusable_instance_includes_failed_status(self) -> None:
        instance = InferenceInstance.objects.create(
            model_name="Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
            port=11428,
            launch_mode="TEXT",
            server_config={"model_id": "Qwen3.5-27B"},
            status="FAILED",
        )
        found = _find_reusable_instance(instance.model_name, instance.port)
        self.assertEqual(found.pk, instance.pk)

    def test_resolve_server_config_allows_same_alias_on_failed_slot(self) -> None:
        instance = InferenceInstance.objects.create(
            model_name="Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
            port=11428,
            launch_mode="TEXT",
            server_config={"model_id": "Qwen3.5-27B"},
            status="FAILED",
        )
        config = _resolve_server_config(
            "TEXT",
            {"model_id": "Qwen3.5-27B"},
            instance.model_name,
            exclude_instance_id=instance.pk,
        )
        self.assertEqual(config["model_id"], "Qwen3.5-27B")
