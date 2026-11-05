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
_CACHED_SNAPSHOT: RouteCacheSnapshot | None = None
_CACHE_EXPIRES_AT: float = 0.0
_CACHED_GENERATION: int = -1
_GENERATION_FILENAME = "route-cache.generation"


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
    global _CACHED_SNAPSHOT, _CACHE_EXPIRES_AT
    with _CACHE_LOCK:
        _CACHED_SNAPSHOT = None
        _CACHE_EXPIRES_AT = 0.0
    _bump_route_generation()


def get_route_snapshot(*, force_refresh: bool = False) -> RouteCacheSnapshot:
    """Return a fresh or cached routing snapshot."""
    global _CACHED_SNAPSHOT, _CACHE_EXPIRES_AT, _CACHED_GENERATION
    from orchestrator.gateway.selectors import build_route_snapshot_from_db

    now = time.monotonic()
    generation = _read_route_generation()
    with _CACHE_LOCK:
        if (
            not force_refresh
            and _CACHED_SNAPSHOT is not None
            and now < _CACHE_EXPIRES_AT
            and generation == _CACHED_GENERATION
        ):
            return _CACHED_SNAPSHOT
        _CACHED_SNAPSHOT = build_route_snapshot_from_db()
        _CACHE_EXPIRES_AT = now + gateway_route_cache_ttl_seconds()
        _CACHED_GENERATION = generation
        return _CACHED_SNAPSHOT
