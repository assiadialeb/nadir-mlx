"""Tests for instance health probing and watchdog helpers."""

from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from orchestrator.instance_health import (
    evaluate_instance_health,
    probe_http_health,
    refresh_instance_health,
    should_skip_watchdog,
)
from orchestrator.instance_watchdog import (
    _auto_restart_enabled,
    _max_restart_retries,
    run_watchdog_cycle,
)
from orchestrator.models import InferenceInstance


class InstanceHealthTests(TestCase):
    def test_probe_http_health_returns_true_on_success(self) -> None:
        instance = MagicMock(port=11400, server_config={"host": "127.0.0.1"})
        with patch("orchestrator.instance_health.httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            self.assertTrue(probe_http_health(instance))

    def test_probe_http_health_returns_false_on_error(self) -> None:
        import httpx

        instance = MagicMock(port=11400, server_config={"host": "127.0.0.1"})
        with patch("orchestrator.instance_health.httpx.get", side_effect=httpx.HTTPError("down")):
            self.assertFalse(probe_http_health(instance))

    @patch("orchestrator.instance_health.probe_http_health", return_value=True)
    @patch("orchestrator.instance_health._find_listener_pids", return_value=[123])
    @patch("orchestrator.instance_health._is_process_alive", return_value=True)
    def test_evaluate_instance_health_healthy(
        self,
        _mock_alive: MagicMock,
        _mock_listeners: MagicMock,
        _mock_probe: MagicMock,
    ) -> None:
        instance = MagicMock(status="RUNNING", pid=123, port=11400, server_config={})
        self.assertEqual(evaluate_instance_health(instance), "HEALTHY")

    @patch("orchestrator.instance_watchdog.refresh_all_instance_health")
    @patch("orchestrator.instance_watchdog._attempt_auto_restart")
    @patch("orchestrator.models.InferenceInstance.objects")
    def test_run_watchdog_cycle_refreshes_and_restarts(
        self,
        mock_objects: MagicMock,
        mock_restart: MagicMock,
        mock_refresh: MagicMock,
    ) -> None:
        mock_objects.filter.return_value = []
        run_watchdog_cycle()
        mock_refresh.assert_called_once()
        mock_restart.assert_not_called()

    def test_should_skip_watchdog_when_env_set(self) -> None:
        with patch.dict("os.environ", {"MLX_DISABLE_INSTANCE_WATCHDOG": "1"}):
            self.assertTrue(should_skip_watchdog())


class InstanceWatchdogConfigTests(TestCase):
    def test_auto_restart_enabled_reads_ops_block(self) -> None:
        instance = MagicMock(server_config={"ops": {"auto_restart": True}})
        self.assertTrue(_auto_restart_enabled(instance))

    def test_max_restart_retries_defaults_to_three(self) -> None:
        instance = MagicMock(server_config={"ops": {}})
        self.assertEqual(_max_restart_retries(instance), 3)


class UpdateStoppedInstanceTests(TestCase):
    @patch("orchestrator.server_manager.is_port_free", return_value=True)
    @patch("orchestrator.models.InferenceInstance.objects")
    def test_update_stopped_instance_rejects_running(
        self,
        _mock_objects: MagicMock,
        _mock_port: MagicMock,
    ) -> None:
        from orchestrator.server_manager import update_stopped_instance

        instance = MagicMock(status="RUNNING", port=11400, launch_mode="TEXT", model_name="demo")
        with self.assertRaises(ValueError):
            update_stopped_instance(instance, port=11401)

    @patch("orchestrator.server_manager.is_port_free", return_value=True)
    @patch("orchestrator.server_manager._resolve_server_config", return_value={"host": "127.0.0.1", "ops": {}})
    @patch("orchestrator.models.InferenceInstance.objects")
    def test_update_stopped_instance_updates_port(
        self,
        mock_objects: MagicMock,
        _mock_resolve: MagicMock,
        _mock_port: MagicMock,
    ) -> None:
        from orchestrator.server_manager import update_stopped_instance

        mock_objects.filter.return_value.exclude.return_value.exists.return_value = False
        instance = MagicMock(status="STOPPED", port=11400, launch_mode="TEXT", model_name="demo")
        update_stopped_instance(instance, port=11401)
        self.assertEqual(instance.port, 11401)
        instance.save.assert_called_once()


class ServerConfigOpsTests(TestCase):
    @override_settings(MLX_DEFAULT_SERVER_HOST="127.0.0.1")
    def test_build_default_server_config_includes_ops(self) -> None:
        from orchestrator.server_config_schema import build_default_server_config

        config = build_default_server_config("TEXT")
        self.assertEqual(config["host"], "127.0.0.1")
        self.assertFalse(config["ops"]["auto_restart"])
        self.assertEqual(config["ops"]["auto_restart_max_retries"], 3)

    @override_settings(MLX_DEFAULT_SERVER_HOST="127.0.0.1")
    def test_validate_normalizes_ops_fields(self) -> None:
        from orchestrator.server_config_schema import validate_and_normalize_server_config

        config = validate_and_normalize_server_config(
            "TEXT",
            {"auto_restart": True, "auto_restart_max_retries": 5},
            "demo-model",
        )
        self.assertTrue(config["ops"]["auto_restart"])
        self.assertEqual(config["ops"]["auto_restart_max_retries"], 5)


class RefreshInstanceHealthIntegrationTests(TestCase):
    def test_refresh_marks_stopped_as_unknown(self) -> None:
        instance = InferenceInstance.objects.create(
            model_name="demo-model",
            port=11499,
            launch_mode="TEXT",
            status="STOPPED",
            server_config={"host": "127.0.0.1", "ops": {}},
        )
        status = refresh_instance_health(instance)
        instance.refresh_from_db()
        self.assertEqual(status, "UNKNOWN")
        self.assertEqual(instance.health_status, "UNKNOWN")
        self.assertIsNotNone(instance.last_health_check_at)
