"""Restore always_on instances after process or host restart."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

from orchestrator.instance_health import refresh_all_instance_health, should_skip_watchdog
from orchestrator.lifecycle_selectors import (
    LIFECYCLE_MODE_ALWAYS_ON,
    get_lifecycle_mode,
)
from orchestrator.models import InferenceInstance
from orchestrator.server_manager import is_manual_stop_in_progress, restart_instance

logger = logging.getLogger(__name__)

_reconcile_started = False
_reconcile_lock = threading.Lock()
_RESTARTABLE_STATUSES = ("FAILED", "STOPPED")
# Defer DB access until after AppConfig.ready() returns (avoids RuntimeWarning).
_RECONCILE_START_DELAY_SECONDS = 1.0


def _should_restore_on_startup(instance: InferenceInstance) -> bool:
    """Return True for always_on instances that should come back after boot."""
    if get_lifecycle_mode(instance.server_config) != LIFECYCLE_MODE_ALWAYS_ON:
        return False
    if instance.status not in _RESTARTABLE_STATUSES:
        return False
    if is_manual_stop_in_progress(instance):
        return False
    return True


def _restore_always_on_instance(instance: InferenceInstance) -> bool:
    """Restart one always_on instance; return True when a restart was attempted."""
    try:
        restart_instance(instance)
        logger.info(
            "Startup reconcile restored always_on instance %s on port %s",
            instance.model_name,
            instance.port,
        )
        return True
    except Exception as exc:
        logger.exception(
            "Startup reconcile failed for instance %s on port %s: %s",
            instance.model_name,
            instance.port,
            exc,
        )
        return False


def reconcile_instances_on_startup() -> dict[str, int]:
    """Reconcile health then restart always_on instances left STOPPED/FAILED.

    On-demand instances are left alone — gateway wake-on-request handles them.
    """
    refresh_all_instance_health()

    restored = 0
    skipped = 0
    failed = 0
    for instance in InferenceInstance.objects.all().order_by("id"):
        instance.refresh_from_db()
        if not _should_restore_on_startup(instance):
            skipped += 1
            continue
        if _restore_always_on_instance(instance):
            restored += 1
        else:
            failed += 1

    logger.info(
        "Startup reconcile finished: restored=%s skipped=%s failed=%s",
        restored,
        skipped,
        failed,
    )
    return {"restored": restored, "skipped": skipped, "failed": failed}


def _reconcile_once() -> None:
    time.sleep(_RECONCILE_START_DELAY_SECONDS)
    try:
        reconcile_instances_on_startup()
    except Exception:
        logger.exception("Startup instance reconcile failed")


def start_startup_reconcile_if_needed() -> None:
    """Run always_on restore once in a background thread (same guards as watchdog)."""
    global _reconcile_started
    from django.conf import settings

    if not getattr(settings, "INSTANCE_STARTUP_RECONCILE_ENABLED", True):
        return
    if should_skip_watchdog():
        return
    with _reconcile_lock:
        if _reconcile_started:
            return
        if "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "true":
            return
        _reconcile_started = True
        thread = threading.Thread(
            target=_reconcile_once,
            name="mlx-startup-reconcile",
            daemon=True,
        )
        thread.start()
        logger.info("Startup instance reconcile scheduled")
