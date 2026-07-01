"""Wake on demand and instance readiness for the Nadir gateway (MLX-38 / MLX-40)."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from django.conf import settings
from django.utils import timezone

from orchestrator.env_utils import env_float, env_str

from orchestrator.gateway.router import GatewayRouteError, GatewayTarget
from orchestrator.gateway.route_cache import clear_gateway_route_cache
from orchestrator.gateway_aliases import find_instance_by_gateway_alias, normalize_gateway_alias
from orchestrator.instance_health import probe_http_health
from orchestrator.lifecycle_selectors import is_on_demand_lifecycle
from orchestrator.server_manager import start_instance

if TYPE_CHECKING:
    from datetime import datetime

    from orchestrator.models import InferenceInstance

_WAKE_LOCKS: dict[str, threading.Lock] = {}
_WAKE_LOCKS_GUARD = threading.Lock()


def gateway_wake_timeout_seconds() -> float:
    """Max seconds to wait for an on_demand instance to become ready after wake."""
    raw = env_str("NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS", "")
    if not raw:
        return float(settings.NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS)
    try:
        return max(1.0, float(raw))
    except ValueError:
        return float(settings.NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS)


def gateway_wake_poll_interval_seconds() -> float:
    """Polling interval while waiting for a waking instance."""
    raw = env_str("NADIR_GATEWAY_WAKE_POLL_INTERVAL_SECONDS", "")
    if not raw:
        return float(settings.NADIR_GATEWAY_WAKE_POLL_INTERVAL_SECONDS)
    try:
        return max(0.1, float(raw))
    except ValueError:
        return float(settings.NADIR_GATEWAY_WAKE_POLL_INTERVAL_SECONDS)


def _wake_lock_for_alias(alias: str) -> threading.Lock:
    alias_key = normalize_gateway_alias(alias).casefold()
    with _WAKE_LOCKS_GUARD:
        lock = _WAKE_LOCKS.get(alias_key)
        if lock is None:
            lock = threading.Lock()
            _WAKE_LOCKS[alias_key] = lock
        return lock


def is_wake_in_progress(alias: str) -> bool:
    """Return True when a gateway wake singleflight lock is held for the alias."""
    alias_key = normalize_gateway_alias(alias).casefold()
    with _WAKE_LOCKS_GUARD:
        lock = _WAKE_LOCKS.get(alias_key)
    return bool(lock and lock.locked())


def _gateway_target_from_instance(instance: InferenceInstance) -> GatewayTarget:
    from orchestrator.gateway.selectors import _gateway_target_from_instance as build_target

    return build_target(instance)


def _instance_is_ready(instance: InferenceInstance) -> bool:
    return instance.status == "RUNNING" and probe_http_health(instance)


def _route_error_for_missing_alias(alias: str) -> GatewayRouteError:
    return GatewayRouteError(
        status_code=404,
        code="model_not_found",
        message=f"No inference instance is registered for alias '{alias}'.",
    )


def _route_error_unavailable(alias: str, *, detail: str = "") -> GatewayRouteError:
    suffix = f" {detail}" if detail else ""
    return GatewayRouteError(
        status_code=503,
        code="model_unavailable",
        message=f"Instance '{alias}' is not ready.{suffix}".strip(),
    )


def _route_error_wake_timeout(alias: str) -> GatewayRouteError:
    return GatewayRouteError(
        status_code=503,
        code="model_waking_timeout",
        message=(
            f"Instance '{alias}' did not become ready within "
            f"{int(gateway_wake_timeout_seconds())} seconds."
        ),
    )


def _mark_wake_timed_out(instance: InferenceInstance) -> None:
    instance.refresh_from_db()
    if instance.status != "LOADING":
        return
    instance.status = "STOPPED"
    instance.pid = None
    instance.stopped_at = timezone.now()
    instance.save(update_fields=["status", "pid", "stopped_at"])
    clear_gateway_route_cache()


def _wait_until_ready(instance: InferenceInstance, alias: str) -> GatewayTarget:
    deadline = time.monotonic() + gateway_wake_timeout_seconds()
    poll_seconds = gateway_wake_poll_interval_seconds()

    while time.monotonic() < deadline:
        instance.refresh_from_db()
        if instance.status == "FAILED":
            raise _route_error_unavailable(alias, detail="Startup failed.")
        if instance.status == "STOPPED":
            raise _route_error_unavailable(alias, detail="Instance stopped during wake.")
        if _instance_is_ready(instance):
            clear_gateway_route_cache()
            return _gateway_target_from_instance(instance)
        time.sleep(poll_seconds)

    _mark_wake_timed_out(instance)
    raise _route_error_wake_timeout(alias)


def _wake_on_demand_instance(instance: InferenceInstance, alias: str) -> GatewayTarget:
    if not is_on_demand_lifecycle(instance.server_config):
        raise _route_error_unavailable(
            alias,
            detail="Instance is stopped and lifecycle_mode is always_on.",
        )

    try:
        start_instance(
            instance.model_name,
            instance.port,
            instance.launch_mode,
            instance.server_config,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        clear_gateway_route_cache()
        raise _route_error_unavailable(alias, detail=str(exc)) from exc

    instance.refresh_from_db()
    clear_gateway_route_cache()
    if _instance_is_ready(instance):
        return _gateway_target_from_instance(instance)
    return _wait_until_ready(instance, alias)


def _ensure_under_wake_lock(instance: InferenceInstance, alias: str) -> GatewayTarget:
    instance.refresh_from_db()
    if _instance_is_ready(instance):
        return _gateway_target_from_instance(instance)

    if instance.status == "LOADING":
        return _wait_until_ready(instance, alias)

    if instance.status in ("STOPPED", "FAILED"):
        return _wake_on_demand_instance(instance, alias)

    raise _route_error_unavailable(alias, detail=f"Status is {instance.status.lower()}.")


def touch_instance_last_used_at(instance_id: int) -> None:
    """Record gateway activity for idle offload (MLX-42)."""
    from orchestrator.models import InferenceInstance

    InferenceInstance.objects.filter(pk=instance_id).update(last_used_at=timezone.now())


def instance_activity_at(instance: InferenceInstance) -> datetime:
    """Return the timestamp used to evaluate idle offload eligibility."""
    if instance.last_used_at is not None:
        return instance.last_used_at
    if instance.stopped_at is not None:
        return instance.stopped_at
    return instance.created_at


def ensure_instance_ready(alias: str) -> GatewayTarget:
    """Wake on_demand instances when needed and return a routable gateway target."""
    cleaned = normalize_gateway_alias(alias)
    if not cleaned:
        raise GatewayRouteError(
            status_code=400,
            code="invalid_model",
            message="The model field is required.",
        )

    instance = find_instance_by_gateway_alias(cleaned)
    if instance is None:
        raise _route_error_for_missing_alias(cleaned)

    instance.refresh_from_db()
    if _instance_is_ready(instance):
        return _gateway_target_from_instance(instance)

    wake_lock = _wake_lock_for_alias(cleaned)
    with wake_lock:
        return _ensure_under_wake_lock(instance, cleaned)
