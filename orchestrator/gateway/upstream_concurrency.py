"""Limit concurrent upstream MLX requests per instance and across the gateway.

Each request takes its instance slot first, then a global slot, so a request
queued behind a busy instance never holds global capacity. Both levels serve
interactive waiters before batch ones (``X-Nadir-Priority: batch``).
"""

from __future__ import annotations

import time
from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator

from django.conf import settings

from orchestrator.env_utils import env_float, env_str
from orchestrator.gateway.gpu_scheduler import (
    PrioritySlotScheduler,
    RequestPriority,
    current_request_priority,
)
from orchestrator.gateway.router import GatewayTarget

MSG_QUEUE_TIMEOUT = (
    "Timed out waiting for an upstream inference slot. "
    "Increase NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM, "
    "NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL or the instance override."
)

_instance_schedulers: dict[int, PrioritySlotScheduler] = {}
_global_scheduler: PrioritySlotScheduler | None = None


def _read_positive_int(raw_value: object) -> int | None:
    if raw_value is None or raw_value == "":
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _limit_from_env(name: str) -> int | None:
    raw = env_str(name, "")
    if not raw:
        return _read_positive_int(getattr(settings, name))
    return _read_positive_int(raw)


def default_max_concurrent_upstream() -> int | None:
    """Per-instance default from NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM (0 = unlimited)."""
    return _limit_from_env("NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM")


def global_max_concurrent_upstream() -> int | None:
    """Cross-instance cap from NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL (0 = unlimited)."""
    return _limit_from_env("NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL")


def queue_timeout_seconds() -> float:
    raw = env_str("NADIR_GATEWAY_QUEUE_TIMEOUT_SECONDS", "")
    if not raw:
        return env_float(
            "NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS",
            float(settings.NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS),
        )
    return float(raw)


def _configured_instance_limit(target: GatewayTarget) -> int | None:
    if target.max_concurrent_upstream is None:
        return default_max_concurrent_upstream()
    if target.max_concurrent_upstream == 0:
        return None
    return target.max_concurrent_upstream


def resolve_max_concurrent_upstream(target: GatewayTarget) -> int | None:
    """Return the per-instance cap, or None when unlimited.

    Never exceeds the instance ``decode_concurrency``: requests above it would
    queue inside mlx-lm, outside the gateway priority ordering.
    """
    limit = _configured_instance_limit(target)
    if target.decode_concurrency is None:
        return limit
    if limit is None:
        return target.decode_concurrency
    return min(limit, target.decode_concurrency)


def _instance_scheduler(instance_id: int, limit: int) -> PrioritySlotScheduler:
    scheduler = _instance_schedulers.get(instance_id)
    if scheduler is None:
        scheduler = PrioritySlotScheduler(limit)
        _instance_schedulers[instance_id] = scheduler
    elif scheduler.capacity != limit:
        scheduler.resize(limit)
    return scheduler


def _shared_global_scheduler(limit: int) -> PrioritySlotScheduler:
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = PrioritySlotScheduler(limit)
    elif _global_scheduler.capacity != limit:
        _global_scheduler.resize(limit)
    return _global_scheduler


def _schedulers_for(target: GatewayTarget) -> list[PrioritySlotScheduler]:
    schedulers: list[PrioritySlotScheduler] = []
    instance_limit = resolve_max_concurrent_upstream(target)
    if instance_limit is not None:
        schedulers.append(_instance_scheduler(target.instance_id, instance_limit))
    global_limit = global_max_concurrent_upstream()
    if global_limit is not None:
        schedulers.append(_shared_global_scheduler(global_limit))
    return schedulers


@asynccontextmanager
async def _held_slot(
    scheduler: PrioritySlotScheduler,
    priority: RequestPriority,
    deadline: float,
) -> AsyncIterator[None]:
    await scheduler.acquire(priority, max(0.0, deadline - time.monotonic()))
    try:
        yield
    finally:
        scheduler.release()


@asynccontextmanager
async def upstream_concurrency_slot(target: GatewayTarget) -> AsyncIterator[None]:
    """Acquire instance + global slots; wait in queue until timeout."""
    from orchestrator.gateway.request_timing import get_active_collector

    collector = get_active_collector()
    priority = current_request_priority()
    deadline = time.monotonic() + queue_timeout_seconds()
    if collector is not None:
        collector.mark_queue_enter()
    async with AsyncExitStack() as stack:
        try:
            for scheduler in _schedulers_for(target):
                await stack.enter_async_context(
                    _held_slot(scheduler, priority, deadline),
                )
        except TimeoutError as exc:
            if collector is not None:
                collector.set_error("upstream_queue_timeout")
            raise UpstreamQueueTimeoutError(MSG_QUEUE_TIMEOUT) from exc
        if collector is not None:
            collector.mark_queue_acquired()
        yield


class UpstreamQueueTimeoutError(Exception):
    """Raised when a request waits too long for an upstream inference slot."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
