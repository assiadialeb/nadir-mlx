"""Limit concurrent upstream MLX requests per instance (queue + backpressure)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from django.conf import settings

from orchestrator.env_utils import env_float, env_str
from orchestrator.gateway.router import GatewayTarget

MSG_QUEUE_TIMEOUT = (
    "Timed out waiting for an upstream inference slot. "
    "Increase NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM or the instance override."
)

_semaphores: dict[int, asyncio.Semaphore] = {}
_semaphore_limits: dict[int, int] = {}


def _read_positive_int(raw_value: object) -> int | None:
    if raw_value is None or raw_value == "":
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def default_max_concurrent_upstream() -> int | None:
    """Global gateway default from NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM (0 = unlimited)."""
    raw = env_str("NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM", "")
    if not raw:
        return _read_positive_int(settings.NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM)
    return _read_positive_int(raw)


def queue_timeout_seconds() -> float:
    raw = env_str("NADIR_GATEWAY_QUEUE_TIMEOUT_SECONDS", "")
    if not raw:
        return env_float(
            "NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS",
            float(settings.NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS),
        )
    return float(raw)


def resolve_max_concurrent_upstream(target: GatewayTarget) -> int | None:
    """Return the concurrency cap for an instance, or None when unlimited."""
    if target.max_concurrent_upstream is not None:
        if target.max_concurrent_upstream == 0:
            return None
        return target.max_concurrent_upstream
    return default_max_concurrent_upstream()


def _get_semaphore(instance_id: int, limit: int) -> asyncio.Semaphore:
    current_limit = _semaphore_limits.get(instance_id)
    if current_limit != limit:
        _semaphores[instance_id] = asyncio.Semaphore(limit)
        _semaphore_limits[instance_id] = limit
    return _semaphores[instance_id]


@asynccontextmanager
async def upstream_concurrency_slot(target: GatewayTarget) -> AsyncIterator[None]:
    """Acquire an upstream slot; wait in queue until timeout."""
    limit = resolve_max_concurrent_upstream(target)
    if limit is None:
        yield
        return

    semaphore = _get_semaphore(target.instance_id, limit)
    try:
        async with asyncio.timeout(queue_timeout_seconds()):
            await semaphore.acquire()
    except TimeoutError as exc:
        raise UpstreamQueueTimeoutError(MSG_QUEUE_TIMEOUT) from exc
    try:
        yield
    finally:
        semaphore.release()


class UpstreamQueueTimeoutError(Exception):
    """Raised when a request waits too long for an upstream inference slot."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
