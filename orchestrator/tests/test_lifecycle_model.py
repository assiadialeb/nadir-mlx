"""Tests for InferenceInstance.last_used_at field (MLX-39)."""

from django.test import TestCase

from orchestrator.models import InferenceInstance


class InferenceInstanceLastUsedAtTests(TestCase):
    def test_last_used_at_defaults_to_null(self) -> None:
        instance = InferenceInstance.objects.create(
            model_name="test-model",
            port=11400,
            launch_mode="TEXT",
            status="STOPPED",
        )
        self.assertIsNone(instance.last_used_at)
