"""Read lifecycle policy from instance server_config (MLX-38 / MLX-39)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orchestrator.models import InferenceInstance

LIFECYCLE_MODE_ALWAYS_ON = "always_on"
LIFECYCLE_MODE_ON_DEMAND = "on_demand"
LIFECYCLE_MODE_MANUAL = "manual"

LIFECYCLE_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    (LIFECYCLE_MODE_ALWAYS_ON, "Always on"),
    (LIFECYCLE_MODE_ON_DEMAND, "On demand (idle offload)"),
    (LIFECYCLE_MODE_MANUAL, "Manual (UI start only)"),
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
    allowed = {
        LIFECYCLE_MODE_ALWAYS_ON,
        LIFECYCLE_MODE_ON_DEMAND,
        LIFECYCLE_MODE_MANUAL,
    }
    if raw not in allowed:
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


def _ops_flag(server_config: dict[str, Any] | None, name: str, *, default: bool) -> bool:
    raw = _ops_section(server_config).get(name, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw) if raw is not None else default


def is_pinned_lifecycle(server_config: dict[str, Any] | None) -> bool:
    """Pinned on_demand instances are never LRU-evicted under memory pressure."""
    return _ops_flag(server_config, "pinned", default=False)


def warmup_on_load_enabled(server_config: dict[str, Any] | None) -> bool:
    """When True, run a 1-token generation after load to pay Metal compile once."""
    return _ops_flag(server_config, "warmup_on_load", default=True)


def speculative_wake_enabled(server_config: dict[str, Any] | None) -> bool:
    """When True, discovery probes may wake a stopped on_demand instance."""
    return _ops_flag(server_config, "speculative_wake", default=False)


def is_on_demand_lifecycle(server_config: dict[str, Any] | None) -> bool:
    return get_lifecycle_mode(server_config) == LIFECYCLE_MODE_ON_DEMAND


def is_manual_lifecycle(server_config: dict[str, Any] | None) -> bool:
    """Manual instances are never auto-started or gateway-woken."""
    return get_lifecycle_mode(server_config) == LIFECYCLE_MODE_MANUAL


def instance_lifecycle_mode(instance: InferenceInstance) -> str:
    return get_lifecycle_mode(instance.server_config)


def instance_idle_minutes(instance: InferenceInstance) -> int:
    return get_idle_minutes(instance.server_config)


def instance_is_pinned(instance: InferenceInstance) -> bool:
    return is_pinned_lifecycle(instance.server_config)


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
    mode = get_lifecycle_mode(server_config)
    if mode == LIFECYCLE_MODE_ON_DEMAND:
        parts = [f"On demand · idle {get_idle_minutes(server_config)} min"]
        if is_pinned_lifecycle(server_config):
            parts.append("pinned")
        return " · ".join(parts)
    if mode == LIFECYCLE_MODE_MANUAL:
        return "Manual · UI start only"
    if is_pinned_lifecycle(server_config):
        return "Always on · pinned"
    return "Always on"


def enrich_instance_lifecycle_ui(instance: InferenceInstance) -> None:
    """Attach lifecycle display attributes for templates (MLX-44)."""
    instance.lifecycle_on_demand = is_on_demand_lifecycle(instance.server_config)
    instance.lifecycle_pinned = is_pinned_lifecycle(instance.server_config)
    label, variant = instance_status_badge(instance)
    instance.status_badge_label = label
    instance.status_badge_variant = variant
    instance.lifecycle_policy_label = lifecycle_policy_summary(instance.server_config)
    instance.idle_minutes_display = instance_idle_minutes(instance)
