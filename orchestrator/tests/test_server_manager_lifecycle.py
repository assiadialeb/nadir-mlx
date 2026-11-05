"""Tests for server_manager start/status helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from django.test import TestCase as DjangoTestCase, override_settings

from orchestrator.models import InferenceInstance
from orchestrator.server_manager import (
    _get_launch_env,
    check_instance_status,
    default_server_host,
    delete_instance,
    get_downloaded_models,
    get_instance_logs,
    is_port_free,
    parse_launch_mode,
    restart_instance,
    start_instance,
)


class ParseLaunchModeTests(TestCase):
    def test_parse_launch_mode_defaults_to_text(self) -> None:
        self.assertEqual(parse_launch_mode(None), "TEXT")

    def test_parse_launch_mode_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            parse_launch_mode("INVALID")


class DefaultServerHostTests(TestCase):
    @override_settings(MLX_DEFAULT_SERVER_HOST="0.0.0.0")
    def test_default_server_host_reads_settings(self) -> None:
        self.assertEqual(default_server_host(), "0.0.0.0")


class GetLaunchEnvTests(TestCase):
    @override_settings(
        IMAGE_OUTPUT_DIR="/tmp/images",
        NADIR_GATEWAY_PUBLIC_BASE_URL="http://localhost:11380",
    )
    def test_get_launch_env_sets_image_defaults(self) -> None:
        env = _get_launch_env(
            "IMAGE",
            {"default_quality": "high", "model_id": "flux"},
            model_name="flux-model",
        )
        self.assertEqual(env["TQDM_DISABLE"], "1")
        self.assertEqual(env["IMAGE_DEFAULT_QUALITY"], "high")
        self.assertEqual(env["IMAGE_OUTPUT_DIR"], "/tmp/images")

    def test_get_launch_env_sets_gateway_alias_for_text(self) -> None:
        env = _get_launch_env("TEXT", {"model_id": "qwen"}, model_name="qwen-folder")
        self.assertEqual(env["NADIR_GATEWAY_ALIAS"], "qwen")

    def test_get_launch_env_strips_serve_supervisor_vars(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "NADIR_SERVE_PARENT_PID": "12345",
                "NADIR_BACKGROUND_OWNER": "gateway",
            },
            clear=False,
        ):
            env = _get_launch_env("TEXT", {"model_id": "qwen"}, model_name="qwen-folder")
        self.assertNotIn("NADIR_SERVE_PARENT_PID", env)
        self.assertNotIn("NADIR_BACKGROUND_OWNER", env)


class GetDownloadedModelsTests(DjangoTestCase):
    def test_get_downloaded_models_returns_complete_folders(self) -> None:
        with tempfile.TemporaryDirectory() as models_root:
            complete = Path(models_root) / "ready-model"
            complete.mkdir()
            (complete / "config.json").write_text("{}", encoding="utf-8")
            incomplete = Path(models_root) / "partial-model"
            incomplete.mkdir()
            with override_settings(MODELS_DIR=str(models_root)):
                with patch(
                    "orchestrator.model_utils.sync_model_download_status",
                ):
                    with patch(
                        "orchestrator.server_manager.is_model_complete",
                        side_effect=lambda path: Path(path).name == "ready-model",
                    ):
                        models = get_downloaded_models()
            self.assertEqual(models, ["ready-model"])


def _create_instance(**overrides: object) -> InferenceInstance:
    fields: dict[str, object] = {
        "model_name": "demo-model",
        "port": 11450,
        "launch_mode": "TEXT",
        "status": "RUNNING",
        "pid": 4242,
    }
    fields.update(overrides)
    return InferenceInstance.objects.create(**fields)


@patch("orchestrator.server_manager.clear_gateway_route_cache")
class CheckInstanceStatusTests(DjangoTestCase):
    def test_check_instance_status_stops_when_pid_missing(self, _mock_cache: MagicMock) -> None:
        instance = _create_instance(pid=None, server_config={"ops": {"lifecycle_mode": "on_demand"}})
        status = check_instance_status(instance)
        instance.refresh_from_db()
        self.assertEqual(status, "STOPPED")
        self.assertEqual(instance.desired_state, InferenceInstance.DESIRED_STATE_STOPPED)

    @patch("orchestrator.server_manager.is_instance_process", return_value=False)
    @patch("orchestrator.server_manager._detect_log_failure", return_value=None)
    def test_check_instance_status_fails_always_on_exit_so_watchdog_restarts(
        self,
        _mock_failure: MagicMock,
        _mock_identity: MagicMock,
        _mock_cache: MagicMock,
    ) -> None:
        instance = _create_instance(port=11451)
        status = check_instance_status(instance)
        instance.refresh_from_db()
        self.assertEqual(status, "FAILED")
        self.assertIsNone(instance.pid)
        self.assertEqual(instance.desired_state, InferenceInstance.DESIRED_STATE_RUNNING)

    @patch("orchestrator.server_manager.is_instance_process", return_value=False)
    @patch("orchestrator.server_manager._detect_log_failure", return_value=None)
    def test_check_instance_status_stops_when_recycled_pid_is_unrelated(
        self,
        _mock_failure: MagicMock,
        _mock_identity: MagicMock,
        _mock_cache: MagicMock,
    ) -> None:
        instance = _create_instance(port=11451, server_config={"ops": {"lifecycle_mode": "manual"}})
        self.assertEqual(check_instance_status(instance), "STOPPED")

    @patch("orchestrator.server_manager.is_instance_process", return_value=True)
    @patch("orchestrator.server_manager._detect_log_failure", return_value="CUDA OOM")
    def test_check_instance_status_fails_orphan_loading_on_log_failure(
        self,
        mock_failure: MagicMock,
        _mock_identity: MagicMock,
        _mock_cache: MagicMock,
    ) -> None:
        instance = _create_instance(port=11452, status="LOADING", pid=4243)
        with patch("orchestrator.server_manager.stop_instance") as mock_stop:
            status = check_instance_status(instance)
            mock_stop.assert_called_once()
        self.assertEqual(status, "FAILED")
        mock_failure.assert_called_once()

    @patch("orchestrator.instance_health.probe_http_health", return_value=True)
    @patch("orchestrator.server_manager.is_instance_process", return_value=True)
    @patch("orchestrator.server_manager._detect_log_failure", return_value=None)
    def test_check_instance_status_promotes_orphan_loading_once_healthy(
        self,
        _mock_failure: MagicMock,
        _mock_identity: MagicMock,
        _mock_health: MagicMock,
        _mock_cache: MagicMock,
    ) -> None:
        instance = _create_instance(port=11453, status="LOADING", pid=4244)
        status = check_instance_status(instance)
        instance.refresh_from_db()
        self.assertEqual(status, "RUNNING")
        self.assertIsNotNone(instance.last_used_at)

    @patch("orchestrator.instance_health.probe_http_health", return_value=False)
    @patch("orchestrator.server_manager.is_instance_process", return_value=True)
    @patch("orchestrator.server_manager._detect_log_failure", return_value=None)
    def test_check_instance_status_keeps_loading_until_health_answers(
        self,
        _mock_failure: MagicMock,
        _mock_identity: MagicMock,
        _mock_health: MagicMock,
        _mock_cache: MagicMock,
    ) -> None:
        instance = _create_instance(port=11453, status="LOADING", pid=4244)
        self.assertEqual(check_instance_status(instance), "LOADING")

    @patch("orchestrator.server_manager.is_slot_lock_held_elsewhere", return_value=True)
    @patch("orchestrator.server_manager.is_instance_process", return_value=False)
    def test_check_instance_status_ignores_slot_mid_transition(
        self,
        mock_identity: MagicMock,
        _mock_lock: MagicMock,
        _mock_cache: MagicMock,
    ) -> None:
        instance = _create_instance(port=11454, status="LOADING")
        self.assertEqual(check_instance_status(instance), "LOADING")
        mock_identity.assert_not_called()


_LAUNCH_PATCHES = (
    patch("orchestrator.server_manager.clear_gateway_route_cache"),
    patch("orchestrator.instance_warmup.maybe_run_load_warmup", return_value=None),
    patch("orchestrator.instance_health.probe_http_health", return_value=True),
    patch("orchestrator.instance_eviction.ensure_headroom_for_load"),
    patch("orchestrator.server_manager.time.sleep"),
    patch("orchestrator.server_manager._detect_log_failure", return_value=None),
    patch("orchestrator.server_manager.is_port_free", return_value=True),
    patch("orchestrator.server_manager._terminate_launchers_on_port"),
    patch("orchestrator.server_manager._build_launch_command", return_value=["python", "-m", "mlx"]),
    patch("orchestrator.server_manager._prepare_model_for_launch"),
    patch("orchestrator.server_manager.is_model_complete", return_value=True),
)


class LaunchLifecycleTests(DjangoTestCase):
    """Two-phase start, intent (desired_state) and idempotency."""

    def setUp(self) -> None:
        self._model_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._model_dir.cleanup)
        for patcher in _LAUNCH_PATCHES:
            patcher.start()
            self.addCleanup(patcher.stop)
        resolve = patch(
            "orchestrator.server_manager.resolve_model_dir",
            return_value=Path(self._model_dir.name),
        )
        resolve.start()
        self.addCleanup(resolve.stop)
        popen = patch("orchestrator.server_manager.subprocess.Popen")
        self.mock_popen = popen.start()
        self.addCleanup(popen.stop)
        self.mock_popen.return_value = MagicMock(pid=9001, poll=MagicMock(return_value=None))

    def test_launch_prepared_instance_cancels_when_stop_requested(self) -> None:
        from orchestrator.server_manager import (
            InstanceStartCancelled,
            launch_prepared_instance,
            prepare_instance_launch,
        )

        plan = prepare_instance_launch("demo-model", port=11470)
        InferenceInstance.objects.filter(pk=plan.instance_id).update(
            desired_state=InferenceInstance.DESIRED_STATE_STOPPED,
        )
        with self.assertRaises(InstanceStartCancelled):
            launch_prepared_instance(plan)
        row = InferenceInstance.objects.get(pk=plan.instance_id)
        self.assertEqual(row.status, "STOPPED")
        self.assertIsNone(row.pid)

    @patch("orchestrator.server_manager.is_instance_process", return_value=True)
    def test_prepare_instance_launch_is_idempotent_when_already_active(
        self,
        _mock_identity: MagicMock,
    ) -> None:
        from orchestrator.server_manager import prepare_instance_launch

        existing = _create_instance(port=11471, pid=77)
        plan = prepare_instance_launch("demo-model", port=11471)
        self.assertTrue(plan.already_active)
        self.assertEqual(plan.instance_id, existing.pk)

    def test_prepare_instance_launch_marks_desired_running(self) -> None:
        from orchestrator.server_manager import prepare_instance_launch

        stopped = _create_instance(
            port=11472,
            status="STOPPED",
            pid=None,
            desired_state=InferenceInstance.DESIRED_STATE_STOPPED,
        )
        prepare_instance_launch("demo-model", port=11472)
        stopped.refresh_from_db()
        self.assertEqual(stopped.status, "LOADING")
        self.assertEqual(stopped.desired_state, InferenceInstance.DESIRED_STATE_RUNNING)

    def test_restart_instance_automatic_respects_user_stop(self) -> None:
        instance = _create_instance(
            port=11473,
            status="FAILED",
            pid=None,
            desired_state=InferenceInstance.DESIRED_STATE_STOPPED,
        )
        result = restart_instance(instance, automatic=True)
        self.assertEqual(result.status, "FAILED")
        self.mock_popen.assert_not_called()

    def test_request_instance_stop_queues_when_slot_busy(self) -> None:
        from orchestrator.instance_locks import SlotLockHandoff
        from orchestrator.server_manager import request_instance_stop

        instance = _create_instance(port=11474, status="LOADING")
        handoff = SlotLockHandoff(11474)
        try:
            with patch("orchestrator.server_manager.USER_ACTION_SLOT_WAIT_SECONDS", 0.05):
                self.assertFalse(request_instance_stop(instance))
        finally:
            handoff.release()
        instance.refresh_from_db()
        self.assertEqual(instance.desired_state, InferenceInstance.DESIRED_STATE_STOPPED)
        self.assertEqual(instance.status, "LOADING")


@patch("orchestrator.server_manager.clear_gateway_route_cache")
@patch("orchestrator.server_manager._collect_stop_targets", return_value=set())
@patch("orchestrator.server_manager._ensure_port_released", return_value=[])
class QueuedStopConvergenceTests(DjangoTestCase):
    def test_request_instance_stop_waits_for_brief_probe(self, *_mocks: object) -> None:
        import threading

        from orchestrator.instance_locks import SlotLockHandoff
        from orchestrator.server_manager import request_instance_stop

        instance = _create_instance(port=11475, status="RUNNING")
        handoff = SlotLockHandoff(11475)
        threading.Timer(0.1, handoff.release).start()

        self.assertTrue(request_instance_stop(instance))
        instance.refresh_from_db()
        self.assertEqual(instance.status, "STOPPED")

    def test_watchdog_applies_stop_queued_behind_busy_slot(self, *_mocks: object) -> None:
        import threading

        from orchestrator.instance_locks import SlotLockHandoff
        from orchestrator.instance_watchdog import run_watchdog_cycle
        from orchestrator.server_manager import request_instance_stop

        instance = _create_instance(port=11476, status="RUNNING")
        holder_ready = threading.Event()
        release_holder = threading.Event()

        def _hold_slot() -> None:
            handoff = SlotLockHandoff(11476)
            holder_ready.set()
            release_holder.wait(timeout=5)
            handoff.release()

        holder = threading.Thread(target=_hold_slot)
        holder.start()
        holder_ready.wait(timeout=5)
        with patch("orchestrator.server_manager.USER_ACTION_SLOT_WAIT_SECONDS", 0.05):
            self.assertFalse(request_instance_stop(instance))
        release_holder.set()
        holder.join(timeout=5)

        with patch("orchestrator.instance_watchdog.refresh_all_instance_health"):
            run_watchdog_cycle()
        instance.refresh_from_db()
        self.assertEqual(instance.status, "STOPPED")
        self.assertEqual(instance.desired_state, InferenceInstance.DESIRED_STATE_STOPPED)

    def test_watchdog_leaves_running_intent_alone(self, *_mocks: object) -> None:
        from orchestrator.instance_watchdog import run_watchdog_cycle

        instance = _create_instance(port=11477, status="RUNNING")
        with patch("orchestrator.instance_watchdog.refresh_all_instance_health"):
            run_watchdog_cycle()
        instance.refresh_from_db()
        self.assertEqual(instance.status, "RUNNING")


class StartInstanceTests(DjangoTestCase):
    @patch("orchestrator.server_manager.clear_gateway_route_cache")
    @patch("orchestrator.instance_warmup.maybe_run_load_warmup", return_value=None)
    @patch("orchestrator.instance_health.probe_http_health", return_value=True)
    @patch("orchestrator.instance_eviction.ensure_headroom_for_load")
    @patch("orchestrator.server_manager.time.sleep")
    @patch("orchestrator.server_manager.subprocess.Popen")
    @patch("orchestrator.server_manager._detect_log_failure", return_value=None)
    @patch("orchestrator.server_manager.is_port_free", return_value=True)
    @patch("orchestrator.server_manager._terminate_launchers_on_port")
    @patch("orchestrator.server_manager._build_launch_command", return_value=["python", "-m", "mlx"])
    @patch("orchestrator.server_manager._prepare_model_for_launch")
    @patch("orchestrator.server_manager.is_model_complete", return_value=True)
    @patch("orchestrator.server_manager.resolve_model_dir")
    def test_start_instance_launches_process_and_marks_running(
        self,
        mock_resolve: MagicMock,
        _mock_complete: MagicMock,
        _mock_prepare: MagicMock,
        _mock_cmd: MagicMock,
        _mock_terminate: MagicMock,
        _mock_port_free: MagicMock,
        _mock_failure: MagicMock,
        mock_popen: MagicMock,
        _mock_sleep: MagicMock,
        _mock_headroom: MagicMock,
        mock_health: MagicMock,
        mock_warmup: MagicMock,
        _mock_cache: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as model_dir:
            mock_resolve.return_value = Path(model_dir)
            mock_process = MagicMock()
            mock_process.pid = 9001
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process
            instance = start_instance("demo-model", port=11454, launch_mode="TEXT")
        self.assertEqual(instance.status, "RUNNING")
        self.assertEqual(instance.pid, 9001)
        self.assertEqual(instance.port, 11454)
        self.assertEqual(instance.desired_state, InferenceInstance.DESIRED_STATE_RUNNING)
        mock_health.assert_called()
        mock_warmup.assert_called_once()

    @patch("orchestrator.server_manager.instance_ready_timeout_seconds", return_value=0.0)
    @patch("orchestrator.server_manager.clear_gateway_route_cache")
    @patch("orchestrator.server_manager._collect_stop_targets", return_value=set())
    @patch("orchestrator.server_manager._ensure_port_released", return_value=[])
    @patch("orchestrator.instance_health.probe_http_health", return_value=False)
    @patch("orchestrator.instance_eviction.ensure_headroom_for_load")
    @patch("orchestrator.server_manager.subprocess.Popen")
    @patch("orchestrator.server_manager._detect_log_failure", return_value=None)
    @patch("orchestrator.server_manager.is_port_free", return_value=True)
    @patch("orchestrator.server_manager._terminate_launchers_on_port")
    @patch("orchestrator.server_manager._build_launch_command", return_value=["python", "-m", "mlx"])
    @patch("orchestrator.server_manager._prepare_model_for_launch")
    @patch("orchestrator.server_manager.is_model_complete", return_value=True)
    @patch("orchestrator.server_manager.resolve_model_dir")
    def test_start_instance_fails_when_health_never_answers(
        self,
        mock_resolve: MagicMock,
        *mocks: MagicMock,
    ) -> None:
        mock_popen = mocks[6]
        mock_popen.return_value = MagicMock(pid=9002, poll=MagicMock(return_value=None))
        with tempfile.TemporaryDirectory() as model_dir:
            mock_resolve.return_value = Path(model_dir)
            with self.assertRaises(RuntimeError):
                start_instance("demo-model", port=11458, launch_mode="TEXT")
        row = InferenceInstance.objects.get(model_name="demo-model", port=11458)
        self.assertEqual(row.status, "FAILED")
        self.assertIsNone(row.pid)

    @patch("orchestrator.server_manager.resolve_model_dir")
    def test_start_instance_rejects_missing_model_folder(self, mock_resolve: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as empty_root:
            missing = Path(empty_root) / "ghost-model"
            mock_resolve.return_value = missing
            with self.assertRaises(ValueError):
                start_instance("ghost-model", port=11455)

    @patch(
        "orchestrator.instance_eviction.ensure_headroom_for_load",
        side_effect=RuntimeError("no headroom"),
    )
    @patch("orchestrator.server_manager.is_port_free", return_value=True)
    @patch("orchestrator.server_manager._terminate_launchers_on_port")
    @patch("orchestrator.server_manager._prepare_model_for_launch")
    @patch("orchestrator.server_manager.is_model_complete", return_value=True)
    @patch("orchestrator.server_manager.resolve_model_dir")
    def test_start_instance_marks_failed_when_pre_popen_raises(
        self,
        mock_resolve: MagicMock,
        _mock_complete: MagicMock,
        _mock_prepare: MagicMock,
        _mock_terminate: MagicMock,
        _mock_port_free: MagicMock,
        _mock_headroom: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as model_dir:
            mock_resolve.return_value = Path(model_dir)
            with self.assertRaises(RuntimeError):
                start_instance("demo-model", port=11457, launch_mode="TEXT")
        row = InferenceInstance.objects.get(model_name="demo-model", port=11457)
        self.assertEqual(row.status, "FAILED")
        self.assertIsNone(row.pid)


class DeleteAndRestartTests(DjangoTestCase):
    @patch("orchestrator.server_manager.stop_instance")
    def test_delete_instance_stops_running_row(self, mock_stop: MagicMock) -> None:
        instance = InferenceInstance.objects.create(
            model_name="demo-model",
            port=11456,
            launch_mode="TEXT",
            status="RUNNING",
            pid=100,
        )
        delete_instance(instance)
        mock_stop.assert_called_once()
        self.assertFalse(InferenceInstance.objects.filter(pk=instance.pk).exists())

    @patch("orchestrator.server_manager.launch_prepared_instance")
    @patch("orchestrator.server_manager.prepare_instance_launch")
    @patch("orchestrator.server_manager._stop_instance_locked")
    def test_restart_instance_stops_then_starts(
        self,
        mock_stop: MagicMock,
        mock_prepare: MagicMock,
        mock_launch: MagicMock,
    ) -> None:
        instance = InferenceInstance.objects.create(
            model_name="demo-model",
            port=11457,
            launch_mode="TEXT",
            status="RUNNING",
            pid=101,
        )
        mock_launch.return_value = instance
        result = restart_instance(instance)
        mock_stop.assert_called_once_with(instance, desired_state=None)
        self.assertTrue(mock_prepare.call_args.kwargs["mark_desired_running"])
        mock_launch.assert_called_once_with(mock_prepare.return_value)
        self.assertEqual(result.pk, instance.pk)


class PortAndLogTests(TestCase):
    def test_is_port_free_detects_bound_port(self) -> None:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            self.assertFalse(is_port_free(port))

    def test_get_instance_logs_returns_tail_of_file(self) -> None:
        with tempfile.TemporaryDirectory() as logs_root:
            log_path = Path(logs_root) / "demo-model_11460.log"
            log_path.write_text("\n".join(f"line-{index}" for index in range(10)), encoding="utf-8")
            with patch(
                "orchestrator.server_manager.resolve_log_file_path",
                return_value=log_path,
            ):
                body = get_instance_logs("demo-model", 11460)
        self.assertIn("line-9", body)
