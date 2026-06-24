"""Background idle offload for on_demand inference instances (MLX-43)."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import timedelta

from django.conf import settings
from django.utils import timezone

from orchestrator.gateway.route_cache import clear_gateway_route_cache
from orchestrator.gateway_aliases import instance_gateway_alias
from orchestrator.instance_health import should_skip_watchdog
from orchestrator.lifecycle_selectors import instance_idle_minutes, is_on_demand_lifecycle
from orchestrator.lifecycle_services import instance_activity_at, is_wake_in_progress
from orchestrator.models import InferenceInstance
from orchestrator.server_manager import is_manual_stop_in_progress, stop_instance

logger = logging.getLogger(__name__)

_idle_watcher_started = False
_idle_watcher_lock = threading.Lock()


def idle_offload_enabled() -> bool:
    """Return whether the idle offload watcher should run."""
    raw = os.environ.get("NADIR_IDLE_OFFLOAD_ENABLED")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(getattr(settings, "NADIR_IDLE_OFFLOAD_ENABLED", True))


def idle_check_interval_seconds() -> float:
    """Polling interval for idle offload evaluation."""
    raw = os.environ.get("NADIR_IDLE_CHECK_INTERVAL_SECONDS")
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return float(getattr(settings, "NADIR_IDLE_CHECK_INTERVAL_SECONDS", 60.0))


def _idle_deadline_reached(instance: InferenceInstance) -> bool:
    idle_minutes = instance_idle_minutes(instance)
    deadline = instance_activity_at(instance) + timedelta(minutes=idle_minutes)
    return timezone.now() >= deadline


def _should_skip_idle_offload(instance: InferenceInstance) -> bool:
    if not is_on_demand_lifecycle(instance.server_config):
        return True
    if instance.status != "RUNNING":
        return True
    if is_manual_stop_in_progress(instance):
        return True
    alias = instance_gateway_alias(instance)
    if is_wake_in_progress(alias):
        return True
    return not _idle_deadline_reached(instance)


def _attempt_idle_offload(instance: InferenceInstance) -> None:
    if _should_skip_idle_offload(instance):
        return

    instance.refresh_from_db()
    if _should_skip_idle_offload(instance):
        return

    alias = instance_gateway_alias(instance)
    try:
        stop_instance(instance)
        clear_gateway_route_cache()
        logger.info(
            "Idle-offloaded instance %s (alias=%s) on port %s",
            instance.model_name,
            alias,
            instance.port,
        )
    except Exception:
        logger.exception(
            "Idle offload failed for instance %s (alias=%s) on port %s",
            instance.model_name,
            alias,
            instance.port,
        )


def run_idle_offload_cycle() -> None:
    """Evaluate running on_demand instances and stop idle ones."""
    if not idle_offload_enabled():
        return

    for instance in InferenceInstance.objects.filter(status="RUNNING").order_by("id"):
        _attempt_idle_offload(instance)


def _idle_watcher_loop() -> None:
    interval = idle_check_interval_seconds()
    while True:
        try:
            run_idle_offload_cycle()
        except Exception:
            logger.exception("Instance idle watcher cycle failed")
        time.sleep(interval)


def start_idle_watcher_if_needed() -> None:
    """Start a daemon thread for idle offload (once per process)."""
    global _idle_watcher_started

    if not idle_offload_enabled():
        return
    if should_skip_watchdog():
        return

    with _idle_watcher_lock:
        if _idle_watcher_started:
            return
        import sys

        if "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "true":
            return
        _idle_watcher_started = True
        thread = threading.Thread(
            target=_idle_watcher_loop,
            name="mlx-instance-idle-watcher",
            daemon=True,
        )
        thread.start()
        logger.info(
            "Instance idle watcher started (interval=%ss)",
            idle_check_interval_seconds(),
        )
