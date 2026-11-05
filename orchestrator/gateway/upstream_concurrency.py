"""Limit concurrent upstream MLX requests per instance and across the gateway.

A single coordinator reserves **global** and **per-instance** capacity together,
so a request waiting on a full instance never holds a global slot. Interactive
waiters are preferred; batch traffic is capped by ``NADIR_GATEWAY_BATCH_MAX_SHARE``.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from django.conf import settings

from orchestrator.env_utils import env_float, env_int, env_str
from orchestrator.gateway.gpu_scheduler import (
    GatewayUpstreamScheduler,
    RequestPriority,
    current_request_priority,
)
from orchestrator.gateway.router import GatewayTarget

MSG_QUEUE_TIMEOUT = (
    "Timed out waiting for an upstream inference slot. "
    "Increase NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM, "
    "NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL or the instance override."
)

_coordinator: GatewayUpstreamScheduler | None = None


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


def batch_line_parallelism() -> int:
    """Parallel upstream calls a batch worker should use (matches scheduler batch pool)."""
    global_limit = global_max_concurrent_upstream()
    if global_limit is None:
        return max(1, env_int("NADIR_BATCH_LINE_CONCURRENCY", 4))
    share = batch_max_share()
    if share <= 0.0:
        return 1
    if share >= 1.0:
        return global_limit
    return max(1, int(global_limit * share))


def batch_max_share() -> float:
    """Fraction of the global pool batch traffic may hold (0–1)."""
    raw = env_str("NADIR_GATEWAY_BATCH_MAX_SHARE", "")
    if not raw:
        value = getattr(settings, "NADIR_GATEWAY_BATCH_MAX_SHARE", 0.5)
    else:
        try:
            value = float(raw)
        except ValueError:
            value = 0.5
    return min(1.0, max(0.0, float(value)))


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


def _get_coordinator() -> GatewayUpstreamScheduler:
    global _coordinator
    global_limit = global_max_concurrent_upstream()
    share = batch_max_share()
    if _coordinator is None:
        _coordinator = GatewayUpstreamScheduler(global_limit, batch_max_share=share)
        return _coordinator
    if _coordinator.global_limit != global_limit or _coordinator.batch_max_share != share:
        _coordinator = GatewayUpstreamScheduler(global_limit, batch_max_share=share)
    return _coordinator


def reset_upstream_coordinator_for_tests() -> None:
    """Drop the process-wide coordinator (tests only)."""
    global _coordinator
    _coordinator = None


@asynccontextmanager
async def upstream_concurrency_slot(target: GatewayTarget) -> AsyncIterator[None]:
    """Acquire global + instance slots atomically; wait in queue until timeout."""
    from orchestrator.gateway.request_timing import get_active_collector

    collector = get_active_collector()
    priority: RequestPriority = current_request_priority()
    instance_limit = resolve_max_concurrent_upstream(target)
    deadline = time.monotonic() + queue_timeout_seconds()
    if collector is not None:
        collector.mark_queue_enter()
    coordinator = _get_coordinator()
    try:
        await coordinator.acquire(
            instance_id=target.instance_id,
            instance_limit=instance_limit,
            priority=priority,
            timeout_seconds=max(0.0, deadline - time.monotonic()),
        )
    except TimeoutError as exc:
        if collector is not None:
            collector.set_error("upstream_queue_timeout")
        raise UpstreamQueueTimeoutError(MSG_QUEUE_TIMEOUT) from exc
    if collector is not None:
        collector.mark_queue_acquired()
    try:
        yield
    finally:
        coordinator.release(
            instance_id=target.instance_id,
            instance_limit=instance_limit,
            priority=priority,
        )


class UpstreamQueueTimeoutError(Exception):
    """Raised when a request waits too long for an upstream inference slot."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
