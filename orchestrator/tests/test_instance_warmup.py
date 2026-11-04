"""Tests for post-load warmup (MLX-126)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from orchestrator.instance_warmup import (
    clear_warmup_state_for_tests,
    mark_instance_needs_warmup,
    maybe_run_load_warmup,
)


def _instance(*, warmup: bool = True, status: str = "RUNNING") -> MagicMock:
    row = MagicMock()
    row.pk = 42
    row.model_name = "demo"
    row.port = 11442
    row.status = status
    row.launch_mode = "TEXT"
    row.server_config = {"ops": {"warmup_on_load": warmup}}
    return row


class InstanceWarmupTests(SimpleTestCase):
    def setUp(self) -> None:
        clear_warmup_state_for_tests()

    @patch("orchestrator.instance_warmup.probe_generation_health", return_value=True)
    @patch("orchestrator.instance_warmup.probe_http_health", return_value=True)
    def test_warmup_runs_once_per_load(
        self,
        _mock_http: MagicMock,
        mock_gen: MagicMock,
    ) -> None:
        instance = _instance()
        first = maybe_run_load_warmup(instance)
        second = maybe_run_load_warmup(instance)
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        mock_gen.assert_called_once()

    @patch("orchestrator.instance_warmup.probe_generation_health", return_value=True)
    @patch("orchestrator.instance_warmup.probe_http_health", return_value=True)
    def test_warmup_reruns_after_mark_needs(
        self,
        _mock_http: MagicMock,
        mock_gen: MagicMock,
    ) -> None:
        instance = _instance()
        maybe_run_load_warmup(instance)
        mark_instance_needs_warmup(instance.pk)
        maybe_run_load_warmup(instance)
        self.assertEqual(mock_gen.call_count, 2)

    def test_warmup_skipped_when_disabled(self) -> None:
        self.assertIsNone(maybe_run_load_warmup(_instance(warmup=False)))

    @patch("orchestrator.instance_warmup.probe_generation_health")
    @patch("orchestrator.instance_warmup.probe_http_health", return_value=False)
    def test_warmup_skipped_when_health_not_ready(
        self,
        _mock_http: MagicMock,
        mock_gen: MagicMock,
    ) -> None:
        self.assertIsNone(maybe_run_load_warmup(_instance()))
        mock_gen.assert_not_called()
