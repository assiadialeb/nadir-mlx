"""Reinforced lifecycle E2E tests with mocked subprocess and upstream (MLX-73)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from django.test import TransactionTestCase
from fastapi.testclient import TestClient

from orchestrator.gateway.app import create_app
from orchestrator.gateway.route_cache import clear_gateway_route_cache
from orchestrator.lifecycle_selectors import LIFECYCLE_MODE_ON_DEMAND
from orchestrator.lifecycle_services import ensure_instance_ready
from orchestrator.models import InferenceInstance
from orchestrator.server_manager import delete_instance, start_instance, stop_instance


def _mock_upstream_chat_response() -> MagicMock:
    upstream = MagicMock()
    upstream.status_code = 200
    upstream.headers = httpx.Headers({"content-type": "application/json"})
    upstream.json = MagicMock(
        return_value={
            "id": "chatcmpl-lifecycle-e2e",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        },
    )
    return upstream


def _wire_async_http_client(mock_client_cls: MagicMock, upstream: MagicMock) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=upstream)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client
    return mock_client


class LifecycleE2ETests(TransactionTestCase):
    def setUp(self) -> None:
        clear_gateway_route_cache()
        self.gateway = TestClient(create_app())

    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    @patch("orchestrator.lifecycle_services.probe_http_health", return_value=True)
    @patch("orchestrator.lifecycle_services.start_instance")
    def test_on_demand_register_wake_infer_stop_delete(
        self,
        mock_start: MagicMock,
        _mock_probe: MagicMock,
        mock_client_cls: MagicMock,
    ) -> None:
        instance = InferenceInstance.objects.create(
            model_name="lifecycle-demo",
            port=11470,
            launch_mode="TEXT",
            server_config={
                "model_id": "lifecycle-demo",
                "host": "127.0.0.1",
                "ops": {"lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND},
            },
            status="STOPPED",
        )
        instance_id = instance.pk

        def _mark_running(*_args: object, **_kwargs: object) -> InferenceInstance:
            instance.status = "RUNNING"
            instance.pid = 7001
            instance.save(update_fields=["status", "pid"])
            return instance

        mock_start.side_effect = _mark_running
        _wire_async_http_client(mock_client_cls, _mock_upstream_chat_response())

        response = self.gateway.post(
            "/v1/chat/completions",
            json={
                "model": "lifecycle-demo",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 8,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "chat.completion")
        mock_start.assert_called_once()

        with patch("orchestrator.server_manager._ensure_port_released", return_value=[]):
            with patch("orchestrator.server_manager._force_stop_pids", return_value=set()):
                with patch("orchestrator.server_manager._collect_stop_targets", return_value={7001}):
                    stop_instance(instance)

        instance.refresh_from_db()
        self.assertEqual(instance.status, "STOPPED")
        self.assertIsNone(instance.pid)

        delete_instance(instance)
        self.assertFalse(InferenceInstance.objects.filter(pk=instance_id).exists())

    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    @patch("orchestrator.lifecycle_services.probe_http_health", return_value=True)
    @patch("orchestrator.server_manager._detect_log_failure", return_value=None)
    @patch("orchestrator.server_manager.time.sleep")
    @patch("orchestrator.server_manager.subprocess.Popen")
    @patch("orchestrator.server_manager.is_port_free", return_value=True)
    @patch("orchestrator.server_manager._terminate_launchers_on_port")
    @patch("orchestrator.server_manager._prepare_model_for_launch")
    @patch("orchestrator.server_manager.is_model_complete", return_value=True)
    @patch("orchestrator.server_manager.os.path.isdir", return_value=True)
    @patch("orchestrator.server_manager.resolve_model_dir")
    def test_always_on_start_infer_stop_delete(
        self,
        mock_resolve_dir: MagicMock,
        _mock_isdir: MagicMock,
        _mock_complete: MagicMock,
        _mock_prepare: MagicMock,
        _mock_terminate: MagicMock,
        _mock_port_free: MagicMock,
        mock_popen: MagicMock,
        _mock_sleep: MagicMock,
        _mock_log_failure: MagicMock,
        _mock_probe: MagicMock,
        mock_client_cls: MagicMock,
    ) -> None:
        mock_resolve_dir.return_value = "/tmp/lifecycle-model"
        process = MagicMock()
        process.pid = 7100
        process.poll.return_value = None
        mock_popen.return_value = process

        instance = start_instance(
            "lifecycle-model",
            port=11471,
            launch_mode="TEXT",
            server_config={"model_id": "lifecycle-always-on", "host": "127.0.0.1"},
        )
        self.assertEqual(instance.status, "RUNNING")
        clear_gateway_route_cache()

        _wire_async_http_client(mock_client_cls, _mock_upstream_chat_response())
        response = self.gateway.post(
            "/v1/chat/completions",
            json={
                "model": "lifecycle-always-on",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        self.assertEqual(response.status_code, 200)

        with patch("orchestrator.server_manager._ensure_port_released", return_value=[]):
            with patch("orchestrator.server_manager._force_stop_pids", return_value=set()):
                with patch(
                    "orchestrator.server_manager._collect_stop_targets",
                    return_value={7100},
                ):
                    delete_instance(instance)

        self.assertFalse(InferenceInstance.objects.filter(pk=instance.pk).exists())

    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    @patch("orchestrator.lifecycle_services.probe_http_health", return_value=True)
    @patch("orchestrator.lifecycle_services.start_instance")
    def test_on_demand_wake_stop_and_wake_again(
        self,
        mock_start: MagicMock,
        _mock_probe: MagicMock,
        mock_client_cls: MagicMock,
    ) -> None:
        instance = InferenceInstance.objects.create(
            model_name="lifecycle-rewake",
            port=11472,
            launch_mode="TEXT",
            server_config={
                "model_id": "lifecycle-rewake",
                "host": "127.0.0.1",
                "ops": {"lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND},
            },
            status="STOPPED",
        )

        def _mark_running(*_args: object, **_kwargs: object) -> InferenceInstance:
            instance.status = "RUNNING"
            instance.pid = 7200
            instance.save(update_fields=["status", "pid"])
            return instance

        mock_start.side_effect = _mark_running
        _wire_async_http_client(mock_client_cls, _mock_upstream_chat_response())

        for _ in range(2):
            clear_gateway_route_cache()
            instance.status = "STOPPED"
            instance.pid = None
            instance.save(update_fields=["status", "pid"])

            ready = ensure_instance_ready("lifecycle-rewake")
            self.assertEqual(ready.alias, "lifecycle-rewake")

            response = self.gateway.post(
                "/v1/chat/completions",
                json={
                    "model": "lifecycle-rewake",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
            self.assertEqual(response.status_code, 200)

            with patch("orchestrator.server_manager._ensure_port_released", return_value=[]):
                with patch("orchestrator.server_manager._force_stop_pids", return_value=set()):
                    with patch(
                        "orchestrator.server_manager._collect_stop_targets",
                        return_value={7200},
                    ):
                        stop_instance(instance)

        self.assertEqual(mock_start.call_count, 2)
