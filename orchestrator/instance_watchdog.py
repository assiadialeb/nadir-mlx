"""Background health polling and optional auto-restart."""

from __future__ import annotations

import logging
import threading
import time
from datetime import timedelta, timezone as dt_timezone
from typing import Any

from django.conf import settings
from django.utils import timezone

from orchestrator.instance_health import refresh_all_instance_health, should_skip_watchdog
from orchestrator.models import InferenceInstance
from orchestrator.server_manager import is_manual_stop_in_progress, restart_instance

logger = logging.getLogger(__name__)

_watchdog_started = False
_watchdog_lock = threading.Lock()


def _ops_config(instance: InferenceInstance) -> dict[str, Any]:
    return dict((instance.server_config or {}).get("ops") or {})


def _auto_restart_enabled(instance: InferenceInstance) -> bool:
    ops = _ops_config(instance)
    if ops.get("auto_restart") is True:
        return True
    return bool((instance.server_config or {}).get("auto_restart"))


def _max_restart_retries(instance: InferenceInstance) -> int:
    ops = _ops_config(instance)
    raw = ops.get("auto_restart_max_retries", instance.server_config.get("auto_restart_max_retries", 3))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 3


def _restart_backoff_seconds(attempt: int) -> int:
    base = int(getattr(settings, "INSTANCE_AUTO_RESTART_BACKOFF_SECONDS", 30))
    return min(base * (2 ** max(0, attempt - 1)), 600)


def _is_restart_frozen(instance: InferenceInstance) -> bool:
    ops = _ops_config(instance)
    frozen_until = ops.get("restart_frozen_until")
    if not frozen_until:
        return False
    try:
        frozen_at = timezone.datetime.fromisoformat(str(frozen_until).replace("Z", "+00:00"))
        if timezone.is_naive(frozen_at):
            frozen_at = timezone.make_aware(frozen_at, dt_timezone.utc)
        return timezone.now() < frozen_at
    except ValueError:
        return False


def _persist_ops(instance: InferenceInstance, ops: dict[str, Any]) -> None:
    config = dict(instance.server_config or {})
    config["ops"] = ops
    instance.server_config = config
    instance.save(update_fields=["server_config"])


def _attempt_auto_restart(instance: InferenceInstance) -> None:
    if is_manual_stop_in_progress(instance):
        return
    if not _auto_restart_enabled(instance):
        return
    if instance.status not in ("FAILED", "STOPPED"):
        return
    if _is_restart_frozen(instance):
        return

    ops = _ops_config(instance)
    attempts = int(ops.get("restart_attempts") or 0)
    max_retries = _max_restart_retries(instance)
    if attempts >= max_retries:
        logger.warning(
            "Auto-restart frozen for instance %s on port %s after %s attempts",
            instance.model_name,
            instance.port,
            attempts,
        )
        ops["restart_frozen_until"] = (
            timezone.now() + timedelta(hours=1)
        ).isoformat()
        _persist_ops(instance, ops)
        return

    try:
        restart_instance(instance)
        ops["restart_attempts"] = attempts + 1
        ops["last_restart_at"] = timezone.now().isoformat()
        ops.pop("restart_frozen_until", None)
        _persist_ops(instance, ops)
        logger.info(
            "Auto-restarted instance %s on port %s (attempt %s)",
            instance.model_name,
            instance.port,
            attempts + 1,
        )
    except Exception as exc:
        logger.exception(
            "Auto-restart failed for instance %s on port %s: %s",
            instance.model_name,
            instance.port,
            exc,
        )
        ops["restart_attempts"] = attempts + 1
        ops["last_restart_error"] = str(exc)
        ops["restart_frozen_until"] = (
            timezone.now() + timedelta(seconds=_restart_backoff_seconds(attempts + 1))
        ).isoformat()
        _persist_ops(instance, ops)


def run_watchdog_cycle() -> None:
    """One health pass plus auto-restart for eligible instances."""
    refresh_all_instance_health()
    for instance in InferenceInstance.objects.filter(status__in=("FAILED", "STOPPED")):
        _attempt_auto_restart(instance)


def _watchdog_loop() -> None:
    interval = int(getattr(settings, "INSTANCE_HEALTH_CHECK_INTERVAL_SECONDS", 30))
    while True:
        try:
            run_watchdog_cycle()
        except Exception:
            logger.exception("Instance watchdog cycle failed")
        time.sleep(interval)


def start_watchdog_if_needed() -> None:
    """Start a daemon thread for periodic health checks (once per process)."""
    global _watchdog_started
    from django.conf import settings

    if not getattr(settings, "INSTANCE_WATCHDOG_ENABLED", True):
        return
    if should_skip_watchdog():
        return
    with _watchdog_lock:
        if _watchdog_started:
            return
        import os
        import sys

        if "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "true":
            return
        _watchdog_started = True
        thread = threading.Thread(target=_watchdog_loop, name="mlx-instance-watchdog", daemon=True)
        thread.start()
        logger.info("Instance watchdog started (interval=%ss)", getattr(settings, "INSTANCE_HEALTH_CHECK_INTERVAL_SECONDS", 30))
