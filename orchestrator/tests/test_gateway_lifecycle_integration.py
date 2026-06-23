"""Gateway integration tests for lifecycle wake path (MLX-46)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from django.test import TransactionTestCase
from fastapi.testclient import TestClient

from orchestrator.gateway.app import create_app
from orchestrator.gateway.route_cache import clear_gateway_route_cache
from orchestrator.lifecycle_selectors import LIFECYCLE_MODE_ON_DEMAND
from orchestrator.models import InferenceInstance


class GatewayLifecycleIntegrationTests(TransactionTestCase):
    def setUp(self) -> None:
        clear_gateway_route_cache()
        self.client = TestClient(create_app())

    @patch("orchestrator.lifecycle_services.start_instance")
    @patch("orchestrator.lifecycle_services.probe_http_health", return_value=True)
    @patch("orchestrator.gateway.services.http_proxy.httpx.AsyncClient")
    def test_chat_completions_wakes_stopped_on_demand_instance(
        self,
        mock_client_cls: MagicMock,
        _mock_probe: MagicMock,
        mock_start: MagicMock,
    ) -> None:
        instance = InferenceInstance.objects.create(
            model_name="sleeping-model",
            port=11420,
            launch_mode="TEXT",
            server_config={
                "model_id": "sleeping-chat",
                "host": "127.0.0.1",
                "ops": {"lifecycle_mode": LIFECYCLE_MODE_ON_DEMAND},
            },
            status="STOPPED",
        )

        def _mark_running(*_args: object, **_kwargs: object) -> InferenceInstance:
            instance.status = "RUNNING"
            instance.pid = 9999
            instance.save(update_fields=["status", "pid"])
            return instance

        mock_start.side_effect = _mark_running

        upstream = MagicMock()
        upstream.status_code = 200
        upstream.headers = httpx.Headers({"content-type": "application/json"})
        upstream.json = MagicMock(return_value={"id": "cmpl-1", "choices": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=upstream)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "sleeping-chat",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        mock_start.assert_called_once()
        instance.refresh_from_db()
        self.assertEqual(instance.status, "RUNNING")

    @patch("orchestrator.lifecycle_services.start_instance")
    def test_chat_completions_always_on_stopped_returns_503(
        self,
        mock_start: MagicMock,
    ) -> None:
        InferenceInstance.objects.create(
            model_name="offline-model",
            port=11421,
            launch_mode="TEXT",
            server_config={"model_id": "offline-chat", "host": "127.0.0.1"},
            status="STOPPED",
        )

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "offline-chat",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "model_unavailable")
        mock_start.assert_not_called()
