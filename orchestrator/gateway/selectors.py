"""Resolve gateway aliases to running MLX inference instances."""

from __future__ import annotations

import time
from dataclasses import replace

from orchestrator.gateway.router import (
    LAUNCH_MODE_API_PATH,
    GatewayRouteError,
    GatewayTarget,
)
from orchestrator.gateway.route_cache import RouteCacheSnapshot, get_route_snapshot
from orchestrator.gateway_aliases import instance_gateway_alias, instance_gateway_aliases
from orchestrator.lifecycle_selectors import get_lifecycle_mode
from orchestrator.models import InferenceInstance
from orchestrator.security_utils import validate_server_bind_host

_UNAVAILABLE_STATUSES = frozenset({"STOPPED", "FAILED", "LOADING"})


def _connect_host(instance: InferenceInstance) -> str:
    raw_host = str((instance.server_config or {}).get("host") or "127.0.0.1")
    if raw_host == "0.0.0.0":
        return "127.0.0.1"
    return validate_server_bind_host(raw_host)


def _downstream_model(instance: InferenceInstance, alias: str) -> str:
    """Return the model field value expected by the upstream MLX backend."""
    if instance.launch_mode == "TEXT":
        return "default_model"
    config = instance.server_config or {}
    return str(config.get("model_id") or instance.model_name or alias)


def _instance_max_concurrent_upstream(instance: InferenceInstance) -> int | None:
    """Per-instance override; None inherits gateway default, 0 disables the cap."""
    config = instance.server_config or {}
    raw = config.get("max_concurrent_upstream")
    if raw is None or raw == "":
        return None
    return int(raw)


def _gateway_target_from_instance(instance: InferenceInstance) -> GatewayTarget:
    resolved_alias = instance_gateway_alias(instance)
    api_path = LAUNCH_MODE_API_PATH.get(instance.launch_mode)
    if not api_path:
        raise GatewayRouteError(
            status_code=500,
            code="unsupported_launch_mode",
            message=f"Launch mode '{instance.launch_mode}' is not supported by the gateway.",
        )
    return GatewayTarget(
        alias=resolved_alias,
        instance_id=instance.id,
        launch_mode=instance.launch_mode,
        host=_connect_host(instance),
        port=instance.port,
        upstream_model=_downstream_model(instance, resolved_alias),
        api_path=api_path,
        max_concurrent_upstream=_instance_max_concurrent_upstream(instance),
    )


def _nadir_model_metadata(instance: InferenceInstance) -> dict[str, object]:
    return {
        "launch_mode": instance.launch_mode,
        "nadir": {
            "lifecycle_mode": get_lifecycle_mode(instance.server_config),
            "status": instance.status.lower(),
        },
    }


def _record_alias_status(
    alias_status: dict[str, str],
    instance: InferenceInstance,
) -> None:
    for alias in instance_gateway_aliases(instance):
        alias_key = alias.casefold()
        if alias_key not in alias_status:
            alias_status[alias_key] = instance.status


def _register_running_aliases(
    running_targets: dict[str, GatewayTarget],
    instance: InferenceInstance,
) -> None:
    if instance.status != "RUNNING":
        return
    try:
        base_target = _gateway_target_from_instance(instance)
    except GatewayRouteError:
        return
    for alias in instance_gateway_aliases(instance):
        alias_key = alias.casefold()
        if alias_key in running_targets:
            continue
        running_targets[alias_key] = replace(base_target, alias=alias)


def _append_model_entry(
    model_entries: list[dict[str, object]],
    seen_model_aliases: set[str],
    instance: InferenceInstance,
    created_at: int,
) -> None:
    for alias in instance_gateway_aliases(instance):
        alias_key = alias.casefold()
        if alias_key in seen_model_aliases:
            continue
        seen_model_aliases.add(alias_key)
        model_entries.append(
            {
                "id": alias,
                "object": "model",
                "created": created_at,
                "owned_by": "nadir",
                "metadata": _nadir_model_metadata(instance),
            }
        )


def build_route_snapshot_from_db() -> RouteCacheSnapshot:
    """Load all gateway routing data from the database in one pass."""
    running_targets: dict[str, GatewayTarget] = {}
    alias_status: dict[str, str] = {}
    model_entries: list[dict[str, object]] = []
    seen_model_aliases: set[str] = set()
    created_at = int(time.time())

    instances = list(InferenceInstance.objects.all())
    for instance in sorted(instances, key=lambda row: row.created_at, reverse=True):
        _record_alias_status(alias_status, instance)
        _register_running_aliases(running_targets, instance)

    for instance in sorted(instances, key=lambda row: row.model_name):
        _append_model_entry(model_entries, seen_model_aliases, instance, created_at)

    return RouteCacheSnapshot(
        built_at=time.monotonic(),
        running_targets=running_targets,
        alias_status=alias_status,
        models_payload={"object": "list", "data": model_entries},
    )


def _route_error_for_alias(snapshot: RouteCacheSnapshot, alias: str) -> GatewayRouteError:
    alias_key = alias.casefold()
    status = snapshot.alias_status.get(alias_key)
    if status is None:
        return GatewayRouteError(
            status_code=404,
            code="model_not_found",
            message=f"No inference instance is registered for alias '{alias}'.",
        )
    if status in _UNAVAILABLE_STATUSES:
        return GatewayRouteError(
            status_code=503,
            code="model_unavailable",
            message=(
                f"Instance '{alias}' is {status.lower()} and cannot serve requests."
            ),
        )
    return GatewayRouteError(
        status_code=503,
        code="model_unavailable",
        message=f"Instance '{alias}' is not ready.",
    )


def resolve_gateway_target(alias: str) -> GatewayTarget:
    """Map a client-facing gateway alias to a running inference instance."""
    cleaned = (alias or "").strip()
    if not cleaned:
        raise GatewayRouteError(
            status_code=400,
            code="invalid_model",
            message="The model field is required.",
        )

    snapshot = get_route_snapshot()
    target = snapshot.running_targets.get(cleaned.casefold())
    if target is not None:
        return target
    raise _route_error_for_alias(snapshot, cleaned)


def list_running_gateway_models() -> dict[str, object]:
    """Build an OpenAI-compatible model list from registered gateway aliases."""
    snapshot = get_route_snapshot()
    return snapshot.models_payload
