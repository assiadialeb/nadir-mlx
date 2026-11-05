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
from orchestrator.instance_locks import InstanceSlotBusy
from orchestrator.lifecycle_selectors import is_manual_lifecycle
from orchestrator.models import InferenceInstance
from orchestrator.server_manager import is_stop_in_progress, restart_instance, stop_instance

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


def _persist_restart_ops(
    instance: InferenceInstance,
    updates: dict[str, Any],
    *,
    remove: tuple[str, ...] = (),
) -> None:
    """Merge restart bookkeeping into the latest ``ops`` (UI edits must not be lost)."""
    instance.refresh_from_db(fields=["server_config"])
    config = dict(instance.server_config or {})
    ops = dict(config.get("ops") or {})
    ops.update(updates)
    for key in remove:
        ops.pop(key, None)
    config["ops"] = ops
    instance.server_config = config
    instance.save(update_fields=["server_config"])


def _is_auto_restart_candidate(instance: InferenceInstance) -> bool:
    """Only unexpected failures of instances the operator wants running."""
    if instance.status != "FAILED":
        return False
    if instance.desired_state != InferenceInstance.DESIRED_STATE_RUNNING:
        return False
    if is_stop_in_progress(instance) or is_manual_lifecycle(instance.server_config):
        return False
    return _auto_restart_enabled(instance) and not _is_restart_frozen(instance)


def _attempt_auto_restart(instance: InferenceInstance) -> None:
    if not _is_auto_restart_candidate(instance):
        return

    attempts = int(_ops_config(instance).get("restart_attempts") or 0)
    if attempts >= _max_restart_retries(instance):
        logger.warning(
            "Auto-restart frozen for instance %s on port %s after %s attempts",
            instance.model_name,
            instance.port,
            attempts,
        )
        frozen_until = timezone.now() + timedelta(hours=1)
        _persist_restart_ops(instance, {"restart_frozen_until": frozen_until.isoformat()})
        return

    try:
        restart_instance(instance, automatic=True, wait_for_slot=False)
    except InstanceSlotBusy:
        return
    except Exception as exc:
        logger.exception(
            "Auto-restart failed for instance %s on port %s: %s",
            instance.model_name,
            instance.port,
            exc,
        )
        backoff = timedelta(seconds=_restart_backoff_seconds(attempts + 1))
        _persist_restart_ops(
            instance,
            {
                "restart_attempts": attempts + 1,
                "last_restart_error": str(exc),
                "restart_frozen_until": (timezone.now() + backoff).isoformat(),
            },
        )
        return

    _persist_restart_ops(
        instance,
        {
            "restart_attempts": attempts + 1,
            "last_restart_at": timezone.now().isoformat(),
        },
        remove=("restart_frozen_until",),
    )
    logger.info(
        "Auto-restarted instance %s on port %s (attempt %s)",
        instance.model_name,
        instance.port,
        attempts + 1,
    )


def _fail_orphaned_benchmarks() -> None:
    from orchestrator.benchmark_service import fail_orphaned_benchmark_runs

    interrupted = fail_orphaned_benchmark_runs()
    if interrupted:
        logger.warning("Marked %s orphaned benchmark run(s) as FAILED", interrupted)


def _converge_stop_intent(instance: InferenceInstance) -> None:
    """Stop an active row whose operator intent is ``stopped``.

    Covers user stops queued while the slot was briefly held (health probe,
    start finishing between its last cancel check and RUNNING).
    """
    if is_stop_in_progress(instance):
        return
    try:
        stop_instance(
            instance,
            desired_state=InferenceInstance.DESIRED_STATE_STOPPED,
            blocking=False,
        )
    except InstanceSlotBusy:
        return
    except RuntimeError as exc:
        logger.warning("Queued stop failed for %s on port %s: %s", instance.model_name, instance.port, exc)
        return
    logger.info("Applied queued stop for %s on port %s", instance.model_name, instance.port)


def _converge_stopped_intents() -> None:
    pending_stops = InferenceInstance.objects.filter(
        status__in=("LOADING", "RUNNING"),
        desired_state=InferenceInstance.DESIRED_STATE_STOPPED,
    )
    for instance in pending_stops:
        _converge_stop_intent(instance)


def run_watchdog_cycle() -> None:
    """Health pass, queued-stop convergence, then auto-restart for eligible instances."""
    refresh_all_instance_health()
    _converge_stopped_intents()
    candidates = InferenceInstance.objects.filter(
        status="FAILED",
        desired_state=InferenceInstance.DESIRED_STATE_RUNNING,
    )
    for instance in candidates:
        _attempt_auto_restart(instance)
    _fail_orphaned_benchmarks()


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
