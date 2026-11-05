"""Tests for gateway-embedded batch worker loop."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from django.test import SimpleTestCase, override_settings

from orchestrator.batch_services import create_batch_file, create_batch_job
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH
from orchestrator.gateway_batch_worker import (
    run_gateway_batch_worker_cycle,
    start_gateway_batch_worker_if_needed,
)
from orchestrator.models import GatewayBatchJob


class GatewayBatchWorkerTests(SimpleTestCase):
    @patch("orchestrator.gateway_batch_worker.process_pending_batch_jobs", return_value=1)
    @patch("orchestrator.gateway_batch_worker.purge_expired_batch_jobs", return_value=0)
    @patch("orchestrator.gateway_batch_worker.reconcile_stale_in_progress_jobs", return_value=0)
    @patch("orchestrator.gateway_batch_worker.gateway_batch_worker_lock")
    def test_run_cycle_processes_under_lock(
        self,
        lock_mock,
        reconcile_mock,
        purge_mock,
        process_mock,
    ) -> None:
        import orchestrator.gateway_batch_worker as worker_module

        worker_module._reconcile_stale_pending = True
        lock_mock.return_value.__enter__.return_value = None
        lock_mock.return_value.__exit__.return_value = None
        run_gateway_batch_worker_cycle()
        reconcile_mock.assert_called_once()
        purge_mock.assert_called_once()
        process_mock.assert_called_once_with(max_jobs=1)

    @override_settings(NADIR_GATEWAY_BATCH_WORKER_ENABLED=False)
    @patch("orchestrator.gateway_batch_worker.threading.Thread")
    def test_start_respects_disabled_setting(self, thread_mock) -> None:
        start_gateway_batch_worker_if_needed()
        thread_mock.assert_not_called()


@pytest.mark.django_db
def test_worker_first_cycle_reconciles_stale_in_progress(batch_files_tmp) -> None:
    import orchestrator.gateway_batch_worker as worker_module

    worker_module._reconcile_stale_pending = True
    record = create_batch_file(
        purpose="batch",
        filename="in.jsonl",
        data=b'{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"x","messages":[]}}\n',
    )
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)
    GatewayBatchJob.objects.filter(pk=job.pk).update(status=GatewayBatchJob.STATUS_IN_PROGRESS)

    with patch("orchestrator.gateway_batch_worker.gateway_batch_worker_lock") as lock_mock:
        lock_mock.return_value.__enter__.return_value = None
        lock_mock.return_value.__exit__.return_value = None
        with patch("orchestrator.gateway_batch_worker.purge_expired_batch_jobs", return_value=0):
            with patch("orchestrator.gateway_batch_worker.process_pending_batch_jobs", return_value=0):
                run_gateway_batch_worker_cycle()

    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_FAILED
