"""Unit tests for inference instance lifecycle helpers."""

import signal
from unittest import TestCase
from unittest.mock import MagicMock, patch

from orchestrator.server_manager import (
    _collect_stop_targets,
    _force_stop_pids,
    _is_process_alive,
    stop_instance,
)


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

    @patch("orchestrator.server_manager.is_port_free", return_value=True)
    @patch("orchestrator.server_manager._find_listener_pids", return_value=[])
    @patch("orchestrator.server_manager._find_orchestrator_launcher_pids", return_value=[])
    @patch("orchestrator.server_manager._terminate_launchers_on_port")
    @patch("orchestrator.server_manager._force_stop_pids", return_value=set())
    @patch("orchestrator.server_manager._collect_stop_targets", return_value={999})
    def test_stop_instance_marks_instance_stopped_when_cleanup_succeeds(
        self,
        _mock_targets: MagicMock,
        _mock_force_stop: MagicMock,
        _mock_terminate_port: MagicMock,
        _mock_launchers: MagicMock,
        _mock_listeners: MagicMock,
        _mock_port_free: MagicMock,
    ) -> None:
        instance = MagicMock(pid=999, port=11400, status="RUNNING")
        stop_instance(instance)
        self.assertIsNone(instance.pid)
        self.assertEqual(instance.status, "STOPPED")
        instance.save.assert_called_once()
