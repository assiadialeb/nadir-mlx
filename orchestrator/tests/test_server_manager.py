"""Unit tests for inference instance lifecycle helpers."""

import signal
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from django.test import TestCase as DjangoTestCase

from orchestrator.server_manager import (
    _collect_stop_targets,
    _detect_log_failure,
    _ensure_port_released,
    _find_reusable_instance,
    _force_stop_pids,
    _is_process_alive,
    _resolve_server_config,
    is_instance_process,
    is_stop_in_progress,
    stop_instance,
)
from orchestrator.models import InferenceInstance


def _descendants(root: int) -> set[int]:
    return {root, root + 1}


class ServerManagerStopTests(TestCase):
    def test_is_process_alive_returns_false_for_invalid_pid(self) -> None:
        self.assertFalse(_is_process_alive(0))
        self.assertFalse(_is_process_alive(-1))

    @patch("orchestrator.server_manager.os.kill", side_effect=ProcessLookupError)
    def test_is_process_alive_returns_false_when_missing(self, _mock_kill: MagicMock) -> None:
        self.assertFalse(_is_process_alive(4242))

    @patch("orchestrator.server_manager.is_instance_process", return_value=True)
    @patch("orchestrator.server_manager._find_listener_pids", return_value=[51])
    @patch("orchestrator.server_manager._find_orchestrator_launcher_pids", return_value=[100])
    def test_collect_stop_targets_includes_launcher_and_pid_trees(
        self,
        _mock_launchers: MagicMock,
        _mock_listeners: MagicMock,
        _mock_identity: MagicMock,
    ) -> None:
        instance = MagicMock(pid=50, port=11400)
        with patch(
            "orchestrator.server_manager._collect_descendant_pids",
            side_effect=_descendants,
        ):
            targets = _collect_stop_targets(instance)
        self.assertEqual(targets, {50, 51, 100, 101})

    @patch("orchestrator.server_manager.is_instance_process", return_value=False)
    @patch("orchestrator.server_manager._find_listener_pids", return_value=[777])
    @patch("orchestrator.server_manager._find_orchestrator_launcher_pids", return_value=[])
    def test_collect_stop_targets_ignores_recycled_pid_and_foreign_listener(
        self,
        _mock_launchers: MagicMock,
        _mock_listeners: MagicMock,
        _mock_identity: MagicMock,
    ) -> None:
        instance = MagicMock(pid=50, port=11400)
        self.assertEqual(_collect_stop_targets(instance), set())

    @patch("orchestrator.server_manager._is_process_alive", return_value=True)
    @patch(
        "orchestrator.server_manager._process_command",
        return_value="python -m orchestrator.mlx_launcher --model /m --port 11400 --model-id x",
    )
    def test_is_instance_process_matches_launcher_on_exact_port(
        self,
        _mock_command: MagicMock,
        _mock_alive: MagicMock,
    ) -> None:
        self.assertTrue(is_instance_process(42, 11400))
        self.assertFalse(is_instance_process(42, 1140))

    @patch("orchestrator.server_manager._is_process_alive", return_value=True)
    @patch("orchestrator.server_manager._process_command", return_value="/usr/sbin/sshd -D")
    def test_is_instance_process_rejects_unrelated_program(
        self,
        _mock_command: MagicMock,
        _mock_alive: MagicMock,
    ) -> None:
        self.assertFalse(is_instance_process(42, 11400))

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

    @patch("orchestrator.server_manager.clear_gateway_route_cache")
    @patch("orchestrator.server_manager._ensure_port_released", return_value=[])
    @patch("orchestrator.server_manager._force_stop_pids", return_value=set())
    @patch("orchestrator.server_manager._collect_stop_targets", return_value={999})
    def test_stop_instance_marks_instance_stopped_when_cleanup_succeeds(
        self,
        _mock_targets: MagicMock,
        _mock_force_stop: MagicMock,
        _mock_ensure_port: MagicMock,
        _mock_cache: MagicMock,
    ) -> None:
        instance = MagicMock(pid=999, port=11400, status="RUNNING", server_config={"ops": {}})
        stop_instance(instance, desired_state=InferenceInstance.DESIRED_STATE_STOPPED)
        self.assertIsNone(instance.pid)
        self.assertEqual(instance.status, "STOPPED")
        self.assertEqual(instance.desired_state, InferenceInstance.DESIRED_STATE_STOPPED)
        self.assertIsNone(instance.stopping_since)
        self.assertFalse(is_stop_in_progress(instance))

    @patch("orchestrator.server_manager.is_port_free", side_effect=[False, False, True])
    @patch("orchestrator.server_manager._owned_port_pids", return_value=set())
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

    @patch("orchestrator.server_manager.clear_gateway_route_cache")
    @patch("orchestrator.server_manager._ensure_port_released", return_value=[4242])
    @patch("orchestrator.server_manager._force_stop_pids", return_value=set())
    @patch("orchestrator.server_manager._collect_stop_targets", return_value=set())
    def test_stop_instance_clears_stopping_marker_on_failure(
        self,
        _mock_targets: MagicMock,
        _mock_force_stop: MagicMock,
        _mock_ensure_port: MagicMock,
        _mock_cache: MagicMock,
    ) -> None:
        instance = MagicMock(pid=999, port=11446, status="RUNNING", server_config={"ops": {}})
        with self.assertRaises(RuntimeError):
            stop_instance(instance)
        self.assertEqual(instance.status, "FAILED")
        self.assertIsNone(instance.pid)
        self.assertIsNone(instance.stopping_since)

    def test_is_stop_in_progress_expires_after_timeout(self) -> None:
        from datetime import timedelta

        from django.utils import timezone

        fresh = MagicMock(stopping_since=timezone.now())
        stale = MagicMock(stopping_since=timezone.now() - timedelta(minutes=5))
        self.assertTrue(is_stop_in_progress(fresh))
        self.assertFalse(is_stop_in_progress(stale))
        self.assertFalse(is_stop_in_progress(MagicMock(stopping_since=None)))


class ServerManagerLogFailureTests(TestCase):
    def test_detect_log_failure_matches_mtp_masked_embedder_reshape(self) -> None:
        log_body = (
            "ValueError: [reshape] Cannot reshape array of size 131072 "
            "into shape (1,1,4096,256)."
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(log_body)
            log_path = handle.name
        try:
            failure = _detect_log_failure(log_path)
        finally:
            Path(log_path).unlink(missing_ok=True)
        self.assertIn("[reshape]", failure or "")


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
