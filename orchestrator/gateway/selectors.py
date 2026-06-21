"""Resolve gateway aliases to running MLX inference instances."""

from __future__ import annotations

from orchestrator.gateway.router import (
    LAUNCH_MODE_API_PATH,
    GatewayRouteError,
    GatewayTarget,
)
from orchestrator.gateway_aliases import find_instance_by_gateway_alias, instance_gateway_alias
from orchestrator.models import InferenceInstance

_UNAVAILABLE_STATUSES = frozenset({"STOPPED", "FAILED", "LOADING"})


def _connect_host(instance: InferenceInstance) -> str:
    host = str((instance.server_config or {}).get("host") or "127.0.0.1")
    if host == "0.0.0.0":
        return "127.0.0.1"
    return host


def _downstream_model(instance: InferenceInstance, alias: str) -> str:
    """Return the model field value expected by the upstream MLX backend."""
    if instance.launch_mode == "TEXT":
        return "default_model"
    config = instance.server_config or {}
    return str(config.get("model_id") or instance.model_name or alias)


def resolve_gateway_target(alias: str) -> GatewayTarget:
    """Map a client-facing gateway alias to a running inference instance."""
    cleaned = (alias or "").strip()
    if not cleaned:
        raise GatewayRouteError(
            status_code=400,
            code="invalid_model",
            message="The model field is required.",
        )

    instance = find_instance_by_gateway_alias(cleaned)
    if instance is None:
        raise GatewayRouteError(
            status_code=404,
            code="model_not_found",
            message=f"No inference instance is registered for alias '{cleaned}'.",
        )

    if instance.status in _UNAVAILABLE_STATUSES:
        raise GatewayRouteError(
            status_code=503,
            code="model_unavailable",
            message=(
                f"Instance '{instance_gateway_alias(instance)}' is "
                f"{instance.status.lower()} and cannot serve requests."
            ),
        )

    if instance.status != "RUNNING":
        raise GatewayRouteError(
            status_code=503,
            code="model_unavailable",
            message=f"Instance '{instance_gateway_alias(instance)}' is not ready.",
        )

    api_path = LAUNCH_MODE_API_PATH.get(instance.launch_mode)
    if not api_path:
        raise GatewayRouteError(
            status_code=500,
            code="unsupported_launch_mode",
            message=f"Launch mode '{instance.launch_mode}' is not supported by the gateway.",
        )

    resolved_alias = instance_gateway_alias(instance)
    return GatewayTarget(
        alias=resolved_alias,
        instance_id=instance.id,
        launch_mode=instance.launch_mode,
        host=_connect_host(instance),
        port=instance.port,
        upstream_model=_downstream_model(instance, resolved_alias),
        api_path=api_path,
    )
