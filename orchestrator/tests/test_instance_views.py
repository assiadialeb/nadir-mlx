"""Tests for server start / stop / restart views (non-blocking transitions)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.contrib.messages import get_messages
from django.test import Client, TestCase
from django.urls import reverse

from orchestrator.instance_locks import InstanceSlotBusy
from orchestrator.models import InferenceInstance
from orchestrator.server_manager import InstanceLaunchPlan


class InstanceViewsTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        User.objects.create_user(username="ops", password="ops-pass")
        self.client.login(username="ops", password="ops-pass")
        self.instance = InferenceInstance.objects.create(
            model_name="demo-model",
            port=11490,
            launch_mode="TEXT",
            status="RUNNING",
            pid=4242,
        )

    def _messages(self, response) -> list[tuple[str, str]]:
        return [(message.tags, str(message)) for message in get_messages(response.wsgi_request)]

    @patch("orchestrator.views.resolve_server_config_from_request", return_value={})
    @patch("orchestrator.views.start_instance_in_background")
    def test_start_instance_view_returns_immediately_with_info_toast(
        self,
        mock_start: MagicMock,
        _mock_config: MagicMock,
    ) -> None:
        mock_start.return_value = InstanceLaunchPlan(
            instance_id=1,
            port=11491,
            model_path="/tmp/demo",
            launch_mode="TEXT",
            requested_model_name="demo-model",
        )
        response = self.client.post(
            reverse("start_instance"),
            {"model_name": "demo-model", "port": "11491", "launch_mode": "TEXT"},
        )
        self.assertRedirects(response, reverse("servers"), fetch_redirect_response=False)
        tags, text = self._messages(response)[0]
        self.assertEqual(tags, "info")
        self.assertIn("loading on port 11491", text)

    @patch("orchestrator.views.resolve_server_config_from_request", return_value={})
    @patch(
        "orchestrator.views.start_instance_in_background",
        side_effect=InstanceSlotBusy(11491),
    )
    def test_start_instance_view_reports_busy_slot(
        self,
        _mock_start: MagicMock,
        _mock_config: MagicMock,
    ) -> None:
        response = self.client.post(
            reverse("start_instance"),
            {"model_name": "demo-model", "port": "11491", "launch_mode": "TEXT"},
        )
        tags, text = self._messages(response)[0]
        self.assertEqual(tags, "error")
        self.assertIn("already in progress", text)

    @patch("orchestrator.views.request_instance_stop", return_value=True)
    def test_stop_instance_view_reports_stopped(self, _mock_stop: MagicMock) -> None:
        response = self.client.post(reverse("stop_instance", args=[self.instance.pk]))
        self.assertEqual(self._messages(response)[0][0], "success")

    @patch("orchestrator.views.request_instance_stop", return_value=False)
    def test_stop_instance_view_reports_queued_stop(self, _mock_stop: MagicMock) -> None:
        response = self.client.post(reverse("stop_instance", args=[self.instance.pk]))
        tags, text = self._messages(response)[0]
        self.assertEqual(tags, "info")
        self.assertIn("Stop queued", text)

    @patch("orchestrator.views.restart_instance_in_background")
    def test_restart_instance_view_runs_in_background(self, mock_restart: MagicMock) -> None:
        response = self.client.post(reverse("restart_instance", args=[self.instance.pk]))
        mock_restart.assert_called_once()
        self.assertEqual(self._messages(response)[0][0], "info")

    def test_servers_view_marks_loading_rows_for_auto_refresh(self) -> None:
        self.instance.status = "LOADING"
        self.instance.server_config = {"host": "0.0.0.0"}
        self.instance.save(update_fields=["status", "server_config"])
        response = self.client.get(reverse("servers"))
        body = response.content.decode("utf-8")
        self.assertIn("data-instance-transition", body)
        self.assertIn("LAN-exposed", body)
        self.assertNotIn("confirm(", body)
