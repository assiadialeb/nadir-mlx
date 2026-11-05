"""Tests for post-load warmup (MLX-126)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from orchestrator.instance_warmup import maybe_run_load_warmup


def _instance(*, warmup: bool = True, status: str = "LOADING") -> MagicMock:
    row = MagicMock()
    row.pk = 42
    row.model_name = "demo"
    row.port = 11442
    row.status = status
    row.launch_mode = "TEXT"
    row.server_config = {"ops": {"warmup_on_load": warmup}}
    return row


@patch("orchestrator.instance_warmup._record_warmup_activity")
class InstanceWarmupTests(SimpleTestCase):
    @patch("orchestrator.instance_warmup.probe_generation_health", return_value=True)
    def test_maybe_run_load_warmup_records_activity_on_success(
        self,
        mock_gen: MagicMock,
        mock_record: MagicMock,
    ) -> None:
        instance = _instance()
        self.assertIsNotNone(maybe_run_load_warmup(instance))
        mock_gen.assert_called_once_with(instance)
        mock_record.assert_called_once_with(instance)

    @patch("orchestrator.instance_warmup.probe_generation_health", return_value=False)
    def test_maybe_run_load_warmup_failure_does_not_count_as_activity(
        self,
        _mock_gen: MagicMock,
        mock_record: MagicMock,
    ) -> None:
        self.assertIsNotNone(maybe_run_load_warmup(_instance()))
        mock_record.assert_not_called()

    @patch(
        "orchestrator.instance_warmup.probe_generation_health",
        side_effect=RuntimeError("boom"),
    )
    def test_maybe_run_load_warmup_never_raises(
        self,
        _mock_gen: MagicMock,
        mock_record: MagicMock,
    ) -> None:
        self.assertIsNotNone(maybe_run_load_warmup(_instance()))
        mock_record.assert_not_called()

    @patch("orchestrator.instance_warmup.probe_generation_health")
    def test_maybe_run_load_warmup_skipped_when_disabled(
        self,
        mock_gen: MagicMock,
        mock_record: MagicMock,
    ) -> None:
        self.assertIsNone(maybe_run_load_warmup(_instance(warmup=False)))
        mock_gen.assert_not_called()
        mock_record.assert_not_called()
