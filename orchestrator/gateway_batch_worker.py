"""Background OpenAI batch worker (runs inside ``run_gateway`` / ``serve``)."""

from __future__ import annotations

import logging
import threading
import time

from django.conf import settings

from orchestrator.batch_services import (
    process_pending_batch_jobs,
    purge_expired_batch_jobs,
    reconcile_stale_in_progress_jobs,
)
from orchestrator.env_utils import env_float
from orchestrator.instance_health import should_skip_watchdog
from orchestrator.instance_locks import BatchWorkerBusy, gateway_batch_worker_lock

logger = logging.getLogger(__name__)

_worker_started = False
_worker_lock = threading.Lock()
_reconcile_stale_pending = True


def run_gateway_batch_worker_cycle() -> None:
    """Process one validating batch if the cross-process worker lock is free."""
    global _reconcile_stale_pending
    try:
        with gateway_batch_worker_lock(blocking=False):
            if _reconcile_stale_pending:
                reconcile_stale_in_progress_jobs()
                _reconcile_stale_pending = False
            try:
                purge_expired_batch_jobs()
            except Exception:
                logger.exception("Batch retention purge failed")
            process_pending_batch_jobs(max_jobs=1)
    except BatchWorkerBusy:
        return
    except Exception:
        logger.exception("Gateway batch worker cycle failed")


def _batch_worker_loop() -> None:
    interval = env_float("NADIR_GATEWAY_BATCH_POLL_SECONDS", 15.0)
    interval = max(3.0, interval)
    while True:
        run_gateway_batch_worker_cycle()
        time.sleep(interval)


def start_gateway_batch_worker_if_needed() -> None:
    """Start a daemon thread that drains validating batch jobs (once per process)."""
    global _worker_started
    if not getattr(settings, "NADIR_GATEWAY_BATCH_WORKER_ENABLED", True):
        return
    if should_skip_watchdog():
        return
    with _worker_lock:
        if _worker_started:
            return
        import os
        import sys

        if "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "true":
            return
        _worker_started = True
        thread = threading.Thread(
            target=_batch_worker_loop,
            name="mlx-gateway-batch-worker",
            daemon=True,
        )
        thread.start()
        logger.info(
            "Gateway batch worker started (poll=%ss)",
            env_float("NADIR_GATEWAY_BATCH_POLL_SECONDS", 15.0),
        )
