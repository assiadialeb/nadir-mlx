"""In-process route cache for gateway alias resolution.

Start/stop may happen in another process (UI runserver, MCP). Every
``clear_gateway_route_cache()`` also bumps a shared generation file so the
gateway drops its snapshot on the next request instead of waiting for the TTL.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from django.conf import settings

from orchestrator.env_utils import env_str
from orchestrator.gateway.router import GatewayTarget
from orchestrator.instance_locks import runtime_state_dir


@dataclass(frozen=True)
class RouteCacheSnapshot:
    """Immutable routing snapshot built from the orchestrator database."""

    built_at: float
    running_targets: dict[str, GatewayTarget]
    alias_status: dict[str, str]
    models_payload: dict[str, object]


_CACHE_LOCK = threading.Lock()
_GENERATION_FILENAME = "route-cache.generation"


@dataclass
class _RouteCacheState:
    snapshot: RouteCacheSnapshot | None = None
    expires_at: float = 0.0
    generation: int = -1


_ROUTE_CACHE = _RouteCacheState()


def _generation_path() -> str:
    return str(runtime_state_dir() / _GENERATION_FILENAME)


def _read_route_generation() -> int:
    try:
        return os.stat(_generation_path()).st_mtime_ns
    except OSError:
        return 0


def _bump_route_generation() -> None:
    path = _generation_path()
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("")
        now_ns = time.time_ns()
        os.utime(path, ns=(now_ns, now_ns))
    except OSError:
        pass


def gateway_route_cache_ttl_seconds() -> float:
    """Return TTL for the in-memory gateway route cache."""
    raw = env_str("NADIR_GATEWAY_ROUTE_CACHE_TTL_SECONDS", "")
    if not raw:
        return float(settings.NADIR_GATEWAY_ROUTE_CACHE_TTL_SECONDS)
    try:
        return max(1.0, float(raw))
    except ValueError:
        return float(settings.NADIR_GATEWAY_ROUTE_CACHE_TTL_SECONDS)


def clear_gateway_route_cache() -> None:
    """Drop the cached snapshot here and signal other processes to drop theirs."""
    with _CACHE_LOCK:
        _ROUTE_CACHE.snapshot = None
        _ROUTE_CACHE.expires_at = 0.0
    _bump_route_generation()


def get_route_snapshot(*, force_refresh: bool = False) -> RouteCacheSnapshot:
    """Return a fresh or cached routing snapshot."""
    from orchestrator.gateway.selectors import build_route_snapshot_from_db

    now = time.monotonic()
    generation = _read_route_generation()
    with _CACHE_LOCK:
        if (
            not force_refresh
            and _ROUTE_CACHE.snapshot is not None
            and now < _ROUTE_CACHE.expires_at
            and generation == _ROUTE_CACHE.generation
        ):
            return _ROUTE_CACHE.snapshot
        _ROUTE_CACHE.snapshot = build_route_snapshot_from_db()
        _ROUTE_CACHE.expires_at = now + gateway_route_cache_ttl_seconds()
        _ROUTE_CACHE.generation = generation
        return _ROUTE_CACHE.snapshot
