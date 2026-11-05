"""Run slow instance transitions off the HTTP request thread (UI Start / Restart)."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from orchestrator.instance_locks import USER_ACTION_SLOT_WAIT_SECONDS, SlotLockHandoff
from orchestrator.models import InferenceInstance
from orchestrator.server_manager import (
    InstanceLaunchPlan,
    LaunchMode,
    get_free_port,
    launch_prepared_instance,
    prepare_instance_launch,
    restart_instance,
)
from orchestrator.security_utils import validated_launch_port

logger = logging.getLogger(__name__)


def _run_with_handoff(handoff: SlotLockHandoff, work: Callable[[], Any], *, name: str) -> None:
    def _worker() -> None:
        from django.db import close_old_connections

        try:
            with handoff.held_by_current_thread():
                work()
        except Exception:
            logger.exception("Background instance transition %s failed", name)
        finally:
            handoff.release()
            close_old_connections()

    threading.Thread(target=_worker, name=name, daemon=True).start()


def start_instance_in_background(
    model_name: str,
    port: int | None,
    launch_mode: LaunchMode,
    server_config: dict[str, Any] | None,
) -> InstanceLaunchPlan:
    """Validate and reserve the row now; load weights, health and warmup in a worker.

    Raises:
        ValueError: invalid model, config, or port.
        InstanceSlotBusy: another start/stop is in progress on this port.
    """
    resolved_port = validated_launch_port(int(port)) if port else get_free_port()
    handoff = SlotLockHandoff(resolved_port, timeout_seconds=USER_ACTION_SLOT_WAIT_SECONDS)
    try:
        with handoff.held_by_current_thread():
            plan = prepare_instance_launch(model_name, resolved_port, launch_mode, server_config)
    except Exception:
        handoff.release()
        raise
    if plan.already_active:
        handoff.release()
        return plan
    _run_with_handoff(
        handoff,
        lambda: launch_prepared_instance(plan),
        name=f"nadir-start-{resolved_port}",
    )
    return plan


def restart_instance_in_background(instance: InferenceInstance) -> None:
    """Reserve the slot now and stop + relaunch in a worker.

    Raises:
        InstanceSlotBusy: another start/stop is in progress on this port.
    """
    handoff = SlotLockHandoff(instance.port, timeout_seconds=USER_ACTION_SLOT_WAIT_SECONDS)
    _run_with_handoff(
        handoff,
        lambda: restart_instance(instance),
        name=f"nadir-restart-{instance.port}",
    )
