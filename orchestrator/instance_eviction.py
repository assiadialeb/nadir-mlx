"""LRU eviction of on_demand instances under memory pressure (MLX-112)."""

from __future__ import annotations

import logging

from orchestrator.gateway.route_cache import clear_gateway_route_cache
from orchestrator.gateway_aliases import instance_gateway_alias
from orchestrator.lifecycle_selectors import is_on_demand_lifecycle
from orchestrator.lifecycle_services import instance_activity_at, is_wake_in_progress
from orchestrator.memory_pressure import (
    estimate_instance_footprint_bytes,
    host_available_bytes,
    lru_eviction_enabled,
)
from orchestrator.models import InferenceInstance

logger = logging.getLogger(__name__)

_MAX_EVICTIONS_PER_LOAD = 3


def list_lru_evictable_instances(*, exclude_id: int | None = None) -> list[InferenceInstance]:
    """Return RUNNING on_demand instances ordered oldest activity first."""
    rows = InferenceInstance.objects.filter(status="RUNNING").order_by("id")
    candidates: list[InferenceInstance] = []
    for instance in rows:
        if exclude_id is not None and instance.pk == exclude_id:
            continue
        if not is_on_demand_lifecycle(instance.server_config):
            continue
        alias = instance_gateway_alias(instance)
        if is_wake_in_progress(alias):
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
    or host available memory cannot be measured.
    """
    if not lru_eviction_enabled():
        return 0

    needed = estimate_instance_footprint_bytes(pending.model_name, pending.server_config)
    available = host_available_bytes()
    if available is None:
        logger.debug("Skipping LRU eviction: host available memory unknown")
        return 0

    stopped = 0
    while available < needed and stopped < _MAX_EVICTIONS_PER_LOAD:
        victim = evict_one_lru_instance(exclude_id=pending.pk)
        if victim is None:
            break
        stopped += 1
        available = host_available_bytes()
        if available is None:
            break
    if stopped:
        logger.info(
            "LRU eviction freed %s instance(s) before loading %s (need≈%s MiB)",
            stopped,
            pending.model_name,
            needed // (1024**2),
        )
    return stopped
