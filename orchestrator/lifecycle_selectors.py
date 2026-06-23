"""Read lifecycle policy from instance server_config (MLX-38 / MLX-39)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orchestrator.models import InferenceInstance

LIFECYCLE_MODE_ALWAYS_ON = "always_on"
LIFECYCLE_MODE_ON_DEMAND = "on_demand"

LIFECYCLE_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    (LIFECYCLE_MODE_ALWAYS_ON, "Always on"),
    (LIFECYCLE_MODE_ON_DEMAND, "On demand (idle offload)"),
)

DEFAULT_IDLE_MINUTES = 30
MIN_IDLE_MINUTES = 5
MAX_IDLE_MINUTES = 1440


def _ops_section(server_config: dict[str, Any] | None) -> dict[str, Any]:
    if not server_config:
        return {}
    ops = server_config.get("ops")
    return ops if isinstance(ops, dict) else {}


def get_lifecycle_mode(server_config: dict[str, Any] | None) -> str:
    """Return normalized lifecycle mode; missing values default to always_on."""
    raw = str(_ops_section(server_config).get("lifecycle_mode") or LIFECYCLE_MODE_ALWAYS_ON)
    if raw not in {LIFECYCLE_MODE_ALWAYS_ON, LIFECYCLE_MODE_ON_DEMAND}:
        return LIFECYCLE_MODE_ALWAYS_ON
    return raw


def get_idle_minutes(server_config: dict[str, Any] | None) -> int:
    """Return idle offload threshold in minutes (on_demand instances only)."""
    raw = _ops_section(server_config).get("idle_minutes", DEFAULT_IDLE_MINUTES)
    try:
        minutes = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_IDLE_MINUTES
    return max(MIN_IDLE_MINUTES, min(MAX_IDLE_MINUTES, minutes))


def is_on_demand_lifecycle(server_config: dict[str, Any] | None) -> bool:
    return get_lifecycle_mode(server_config) == LIFECYCLE_MODE_ON_DEMAND


def instance_lifecycle_mode(instance: InferenceInstance) -> str:
    return get_lifecycle_mode(instance.server_config)


def instance_idle_minutes(instance: InferenceInstance) -> int:
    return get_idle_minutes(instance.server_config)
