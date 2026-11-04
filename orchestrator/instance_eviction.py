"""LRU eviction of on_demand instances under memory pressure (MLX-112)."""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone

from orchestrator.gateway.route_cache import clear_gateway_route_cache
from orchestrator.gateway_aliases import instance_gateway_alias
from orchestrator.lifecycle_selectors import is_on_demand_lifecycle, is_pinned_lifecycle
from orchestrator.lifecycle_services import instance_activity_at, is_wake_in_progress
from orchestrator.memory_pressure import (
    estimate_instance_footprint_bytes,
    host_available_bytes,
    lru_eviction_enabled,
)
from orchestrator.models import InferenceInstance

logger = logging.getLogger(__name__)

_MAX_EVICTIONS_PER_LOAD = 3
_DEFAULT_EVICTION_GRACE_SECONDS = 60
_eviction_lock = threading.Lock()
_reserved_victim_ids: set[int] = set()


def eviction_grace_seconds() -> int:
    """Minimum idle age before an on_demand instance may be LRU-evicted.

    Defaults to a short grace window. Gateway traffic refreshes ``last_used_at``
    at request start/end and on a mid-request heartbeat (~15s) during long
    JSON calls and SSE streams (including prefill waits), so active generations
    stay non-evictable without a multi-minute lockout after every call. Idle
    offload uses the same timestamp.

    The effective grace is always at least ``2 ×`` the activity heartbeat so a
    misconfigured shorter grace cannot silently defeat mid-request protection.
    """
    import math

    from orchestrator.gateway.services.http_proxy import activity_heartbeat_seconds

    configured = _DEFAULT_EVICTION_GRACE_SECONDS
    raw = os.environ.get("NADIR_LRU_EVICTION_GRACE_SECONDS")
    if raw is not None and str(raw).strip():
        try:
            configured = max(0, int(raw))
        except ValueError:
            pass
    min_grace = max(0, math.ceil(2 * activity_heartbeat_seconds()))
    return max(configured, min_grace)


def list_lru_evictable_instances(*, exclude_id: int | None = None) -> list[InferenceInstance]:
    """Return RUNNING on_demand instances ordered oldest activity first."""
    rows = InferenceInstance.objects.filter(status="RUNNING").order_by("id")
    candidates: list[InferenceInstance] = []
    grace = timedelta(seconds=eviction_grace_seconds())
    now = datetime.now(timezone.utc)
    for instance in rows:
        if exclude_id is not None and instance.pk == exclude_id:
            continue
        if instance.pk in _reserved_victim_ids:
            continue
        if not is_on_demand_lifecycle(instance.server_config):
            continue
        if is_pinned_lifecycle(instance.server_config):
            continue
        alias = instance_gateway_alias(instance)
        if is_wake_in_progress(alias):
            continue
        if _within_grace_period(instance, now=now, grace=grace):
            continue
        candidates.append(instance)
    candidates.sort(key=instance_activity_at)
    return candidates


def evict_one_lru_instance(*, exclude_id: int | None = None) -> InferenceInstance | None:
    """Stop the least-recently-used evictable on_demand instance, if any."""
    from orchestrator.server_manager import is_manual_stop_in_progress, stop_instance

    candidates = list_lru_evictable_instances(exclude_id=exclude_id)
    for instance in candidates:
        if is_manual_stop_in_progress(instance):
            continue
        alias = instance_gateway_alias(instance)
        try:
            stop_instance(instance)
            clear_gateway_route_cache()
            logger.info(
                "LRU-evicted instance %s (alias=%s) on port %s to free memory",
                instance.model_name,
                alias,
                instance.port,
            )
            return instance
        except Exception:
            logger.exception(
                "LRU eviction failed for instance %s (alias=%s)",
                instance.model_name,
                alias,
            )
    return None


def ensure_headroom_for_load(pending: InferenceInstance) -> int:
    """Evict LRU on_demand instances until pending footprint fits, or give up.

    Returns the number of instances stopped. No-op when eviction is disabled
    or host available memory cannot be measured. Victim selection is serialized;
    ``stop_instance`` runs outside the lock to avoid blocking other planners.
    """
    if not lru_eviction_enabled():
        return 0

    needed = estimate_instance_footprint_bytes(pending.model_name, pending.server_config)
    with _eviction_lock:
        plan = _plan_evictions(pending, needed=needed)

    if plan is None:
        return 0

    stopped = 0
    for victim in plan["victims"]:
        try:
            if _stop_planned_victim(victim) is not None:
                stopped += 1
        finally:
            _release_victim_reservation(victim.pk)
    available = host_available_bytes()
    if stopped:
        logger.info(
            "LRU eviction freed %s instance(s) before loading %s (need≈%s MiB)",
            stopped,
            pending.model_name,
            needed // (1024**2),
        )
    if available is not None and available < needed:
        logger.warning(
            "Proceeding to load %s despite insufficient free memory "
            "(need≈%s MiB, available≈%s MiB) after %s eviction(s)",
            pending.model_name,
            needed // (1024**2),
            available // (1024**2),
            stopped,
        )
    return stopped


def _plan_evictions(
    pending: InferenceInstance,
    *,
    needed: int,
) -> dict[str, object] | None:
    available = host_available_bytes()
    if available is None:
        logger.debug("Skipping LRU eviction: host available memory unknown")
        return None

    from orchestrator.server_manager import is_manual_stop_in_progress

    victims: list[InferenceInstance] = []
    projected = available
    for candidate in list_lru_evictable_instances(exclude_id=pending.pk):
        if projected >= needed or len(victims) >= _MAX_EVICTIONS_PER_LOAD:
            break
        if is_manual_stop_in_progress(candidate):
            continue
        victims.append(candidate)
        _reserved_victim_ids.add(candidate.pk)
        projected += estimate_instance_footprint_bytes(
            candidate.model_name,
            candidate.server_config,
        )
    return {"victims": victims, "available": available}


def _release_victim_reservation(instance_id: int) -> None:
    with _eviction_lock:
        _reserved_victim_ids.discard(instance_id)


def _stop_planned_victim(instance: InferenceInstance) -> InferenceInstance | None:
    from orchestrator.server_manager import is_manual_stop_in_progress, stop_instance

    if is_manual_stop_in_progress(instance):
        return None
    alias = instance_gateway_alias(instance)
    try:
        stop_instance(instance)
        clear_gateway_route_cache()
        logger.info(
            "LRU-evicted instance %s (alias=%s) on port %s to free memory",
            instance.model_name,
            alias,
            instance.port,
        )
        return instance
    except Exception:
        logger.exception(
            "LRU eviction failed for instance %s (alias=%s)",
            instance.model_name,
            alias,
        )
        return None


def _within_grace_period(
    instance: InferenceInstance,
    *,
    now: datetime,
    grace: timedelta,
) -> bool:
    if grace.total_seconds() <= 0:
        return False
    activity = instance_activity_at(instance)
    if activity.tzinfo is None:
        activity = activity.replace(tzinfo=timezone.utc)
    return (now - activity) < grace
