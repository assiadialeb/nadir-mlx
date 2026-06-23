"""Tests for gateway activity tracking (MLX-42)."""

from datetime import timedelta

from django.test import TestCase
from django.utils import timezone

from orchestrator.lifecycle_services import instance_activity_at, touch_instance_last_used_at
from orchestrator.models import InferenceInstance


class TouchInstanceLastUsedAtTests(TestCase):
    def test_touch_updates_last_used_at(self) -> None:
        instance = InferenceInstance.objects.create(
            model_name="touch-model",
            port=11420,
            launch_mode="TEXT",
            server_config={"model_id": "touch-chat", "host": "127.0.0.1"},
            status="RUNNING",
        )
        self.assertIsNone(instance.last_used_at)

        touch_instance_last_used_at(instance.pk)
        instance.refresh_from_db()
        self.assertIsNotNone(instance.last_used_at)

    def test_instance_activity_at_prefers_last_used_at(self) -> None:
        now = timezone.now()
        instance = InferenceInstance.objects.create(
            model_name="activity-model",
            port=11421,
            launch_mode="TEXT",
            server_config={"host": "127.0.0.1"},
            status="RUNNING",
            last_used_at=now,
            stopped_at=now - timedelta(hours=1),
        )
        self.assertEqual(instance_activity_at(instance), now)

    def test_instance_activity_at_falls_back_to_stopped_at(self) -> None:
        stopped_at = timezone.now() - timedelta(minutes=10)
        instance = InferenceInstance.objects.create(
            model_name="stopped-model",
            port=11422,
            launch_mode="TEXT",
            server_config={"host": "127.0.0.1"},
            status="STOPPED",
            stopped_at=stopped_at,
        )
        self.assertEqual(instance_activity_at(instance), stopped_at)

    def test_instance_activity_at_falls_back_to_created_at(self) -> None:
        instance = InferenceInstance.objects.create(
            model_name="created-model",
            port=11423,
            launch_mode="TEXT",
            server_config={"host": "127.0.0.1"},
            status="RUNNING",
        )
        self.assertEqual(instance_activity_at(instance), instance.created_at)
