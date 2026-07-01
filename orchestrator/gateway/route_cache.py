"""In-process route cache for gateway alias resolution."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from django.conf import settings

from orchestrator.env_utils import env_str
from orchestrator.gateway.router import GatewayTarget


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
    """Drop the cached snapshot (used in tests and after TTL expiry)."""
    global _CACHED_SNAPSHOT, _CACHE_EXPIRES_AT
    with _CACHE_LOCK:
        _CACHED_SNAPSHOT = None
        _CACHE_EXPIRES_AT = 0.0


def get_route_snapshot(*, force_refresh: bool = False) -> RouteCacheSnapshot:
    """Return a fresh or cached routing snapshot."""
    global _CACHED_SNAPSHOT, _CACHE_EXPIRES_AT
    from orchestrator.gateway.selectors import build_route_snapshot_from_db

    now = time.monotonic()
    with _CACHE_LOCK:
        if (
            not force_refresh
            and _CACHED_SNAPSHOT is not None
            and now < _CACHE_EXPIRES_AT
        ):
            return _CACHED_SNAPSHOT
        _CACHED_SNAPSHOT = build_route_snapshot_from_db()
        _CACHE_EXPIRES_AT = now + gateway_route_cache_ttl_seconds()
        return _CACHED_SNAPSHOT
