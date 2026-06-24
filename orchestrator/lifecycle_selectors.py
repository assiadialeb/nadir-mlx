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


def instance_status_badge(instance: InferenceInstance) -> tuple[str, str]:
    """Return UI label and variant key for the instance list status badge."""
    if is_on_demand_lifecycle(instance.server_config):
        on_demand_labels = {
            "STOPPED": ("Sleeping", "sleeping"),
            "LOADING": ("Waking", "waking"),
            "RUNNING": ("Ready", "ready"),
        }
        if instance.status in on_demand_labels:
            return on_demand_labels[instance.status]

    default_labels = {
        "RUNNING": ("Running", "running"),
        "LOADING": ("Loading", "loading"),
        "STOPPED": ("Stopped", "stopped"),
        "FAILED": ("Failed", "failed"),
    }
    return default_labels.get(instance.status, (instance.status.title(), "stopped"))


def lifecycle_policy_summary(server_config: dict[str, Any] | None) -> str:
    """Short lifecycle policy label for the servers list."""
    if get_lifecycle_mode(server_config) == LIFECYCLE_MODE_ON_DEMAND:
        return f"On demand · idle {get_idle_minutes(server_config)} min"
    return "Always on"


def enrich_instance_lifecycle_ui(instance: InferenceInstance) -> None:
    """Attach lifecycle display attributes for templates (MLX-44)."""
    instance.lifecycle_on_demand = is_on_demand_lifecycle(instance.server_config)
    label, variant = instance_status_badge(instance)
    instance.status_badge_label = label
    instance.status_badge_variant = variant
    instance.lifecycle_policy_label = lifecycle_policy_summary(instance.server_config)
    instance.idle_minutes_display = instance_idle_minutes(instance)
