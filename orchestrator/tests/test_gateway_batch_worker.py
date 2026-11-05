"""Tests for gateway-embedded batch worker loop."""

from __future__ import annotations

from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from orchestrator.gateway_batch_worker import (
    run_gateway_batch_worker_cycle,
    start_gateway_batch_worker_if_needed,
)


class GatewayBatchWorkerTests(SimpleTestCase):
    @patch("orchestrator.gateway_batch_worker.process_pending_batch_jobs", return_value=1)
    @patch("orchestrator.gateway_batch_worker.purge_expired_batch_jobs", return_value=0)
    @patch("orchestrator.gateway_batch_worker.gateway_batch_worker_lock")
    def test_run_cycle_processes_under_lock(
        self,
        lock_mock,
        purge_mock,
        process_mock,
    ) -> None:
        lock_mock.return_value.__enter__.return_value = None
        lock_mock.return_value.__exit__.return_value = None
        run_gateway_batch_worker_cycle()
        purge_mock.assert_called_once()
        process_mock.assert_called_once_with(max_jobs=1)

    @override_settings(NADIR_GATEWAY_BATCH_WORKER_ENABLED=False)
    @patch("orchestrator.gateway_batch_worker.threading.Thread")
    def test_start_respects_disabled_setting(self, thread_mock) -> None:
        start_gateway_batch_worker_if_needed()
        thread_mock.assert_not_called()
