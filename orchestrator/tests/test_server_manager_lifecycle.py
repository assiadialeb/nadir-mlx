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


class CheckInstanceStatusTests(DjangoTestCase):
    def test_check_instance_status_stops_when_pid_missing(self) -> None:
        instance = InferenceInstance.objects.create(
            model_name="demo-model",
            port=11450,
            launch_mode="TEXT",
            status="RUNNING",
            pid=None,
        )
        status = check_instance_status(instance)
        instance.refresh_from_db()
        self.assertEqual(status, "STOPPED")
        self.assertEqual(instance.status, "STOPPED")

    @patch("orchestrator.server_manager.os.kill", side_effect=ProcessLookupError)
    @patch("orchestrator.server_manager._detect_log_failure", return_value=None)
    def test_check_instance_status_marks_stopped_when_process_gone(
        self,
        _mock_failure: MagicMock,
        _mock_kill: MagicMock,
    ) -> None:
        instance = InferenceInstance.objects.create(
            model_name="demo-model",
            port=11451,
            launch_mode="TEXT",
            status="RUNNING",
            pid=4242,
        )
        status = check_instance_status(instance)
        instance.refresh_from_db()
        self.assertEqual(status, "STOPPED")
        self.assertIsNone(instance.pid)

    @patch("orchestrator.server_manager.os.kill")
    @patch("orchestrator.server_manager._detect_log_failure", return_value="CUDA OOM")
    def test_check_instance_status_promotes_loading_to_running(
        self,
        mock_failure: MagicMock,
        _mock_kill: MagicMock,
    ) -> None:
        instance = InferenceInstance.objects.create(
            model_name="demo-model",
            port=11452,
            launch_mode="TEXT",
            status="LOADING",
            pid=4243,
        )
        with patch("orchestrator.server_manager.stop_instance") as mock_stop:
            status = check_instance_status(instance)
            mock_stop.assert_called_once()
        instance.refresh_from_db()
        self.assertEqual(status, "FAILED")
        mock_failure.assert_called_once()

    @patch("orchestrator.server_manager.os.kill")
    @patch("orchestrator.server_manager._detect_log_failure", return_value=None)
    def test_check_instance_status_promotes_loading_without_log_failure(
        self,
        _mock_failure: MagicMock,
        _mock_kill: MagicMock,
    ) -> None:
        instance = InferenceInstance.objects.create(
            model_name="demo-model",
            port=11453,
            launch_mode="TEXT",
            status="LOADING",
            pid=4244,
        )
        status = check_instance_status(instance)
        instance.refresh_from_db()
        self.assertEqual(status, "RUNNING")
        self.assertEqual(instance.status, "RUNNING")


class StartInstanceTests(DjangoTestCase):
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

    @patch("orchestrator.server_manager.resolve_model_dir")
    def test_start_instance_rejects_missing_model_folder(self, mock_resolve: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as empty_root:
            missing = Path(empty_root) / "ghost-model"
            mock_resolve.return_value = missing
            with self.assertRaises(ValueError):
                start_instance("ghost-model", port=11455)


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

    @patch("orchestrator.server_manager.start_instance")
    @patch("orchestrator.server_manager.stop_instance")
    def test_restart_instance_stops_then_starts(
        self,
        mock_stop: MagicMock,
        mock_start: MagicMock,
    ) -> None:
        instance = InferenceInstance.objects.create(
            model_name="demo-model",
            port=11457,
            launch_mode="TEXT",
            status="RUNNING",
            pid=101,
        )
        relaunched = InferenceInstance.objects.create(
            model_name="demo-model",
            port=11457,
            launch_mode="TEXT",
            status="RUNNING",
            pid=202,
        )
        mock_start.return_value = relaunched
        result = restart_instance(instance)
        mock_stop.assert_called_once_with(instance)
        mock_start.assert_called_once()
        self.assertEqual(result.pk, relaunched.pk)


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
