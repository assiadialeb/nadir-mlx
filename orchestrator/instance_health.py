"""Health probing for inference instances."""

from __future__ import annotations

import os
import time
from typing import Any, Literal

import httpx
from django.utils import timezone

from orchestrator.env_utils import env_int, env_str
from orchestrator.models import InferenceInstance
from orchestrator.security_utils import build_validated_http_url, validate_server_bind_host
from orchestrator.server_manager import (
    _find_listener_pids,
    _is_process_alive,
    check_instance_status,
    is_manual_stop_in_progress,
)

HealthStatus = Literal["HEALTHY", "DEGRADED", "DOWN", "UNKNOWN"]

HEALTH_PATH = "/health"
HEALTH_TIMEOUT_SECONDS = 2.0
GENERATION_PROBE_TIMEOUT_SECONDS = 30.0
_last_generation_probe_at: dict[int, float] = {}


def _connect_host(instance: InferenceInstance) -> str:
    raw_host = str((instance.server_config or {}).get("host") or "127.0.0.1")
    if raw_host == "0.0.0.0":
        return "127.0.0.1"
    return validate_server_bind_host(raw_host)


def probe_http_health(instance: InferenceInstance) -> bool:
    """Return True when the instance responds on /health."""
    url = build_validated_http_url(_connect_host(instance), instance.port, HEALTH_PATH)
    try:
        response = httpx.get(url, timeout=HEALTH_TIMEOUT_SECONDS)
        return response.status_code < 500
    except httpx.HTTPError:
        return False


def _generation_probe_payload(launch_mode: str) -> tuple[str, dict[str, Any]] | None:
    if launch_mode in ("TEXT", "MULTIMODAL"):
        return "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
        }
    if launch_mode == "EMBEDDING":
        return "/v1/embeddings", {"input": "health probe"}
    if launch_mode == "RERANKER":
        return "/v1/rerank", {
            "query": "health",
            "documents": ["probe"],
        }
    return None


def deep_generation_health_enabled() -> bool:
    return env_str("NADIR_DEEP_INSTANCE_HEALTH", "") == "1"


def _generation_probe_interval_seconds() -> int:
    return max(30, env_int("NADIR_DEEP_HEALTH_INTERVAL_SECONDS", 300))


def _should_probe_generation(instance: InferenceInstance) -> bool:
    if not deep_generation_health_enabled():
        return False
    if instance.status != "RUNNING":
        return False
    if _generation_probe_payload(instance.launch_mode) is None:
        return False

    now = time.monotonic()
    last_probe = _last_generation_probe_at.get(instance.id, 0.0)
    return (now - last_probe) >= _generation_probe_interval_seconds()


def probe_generation_health(instance: InferenceInstance) -> bool:
    """Return True when a minimal inference request succeeds."""
    probe = _generation_probe_payload(instance.launch_mode)
    if probe is None:
        return True

    path, payload = probe
    url = build_validated_http_url(_connect_host(instance), instance.port, path)
    try:
        response = httpx.post(url, json=payload, timeout=GENERATION_PROBE_TIMEOUT_SECONDS)
        _last_generation_probe_at[instance.id] = time.monotonic()
        return response.status_code < 500
    except httpx.HTTPError:
        _last_generation_probe_at[instance.id] = time.monotonic()
        return False


def evaluate_instance_health(instance: InferenceInstance) -> HealthStatus:
    """Derive health from process, port, and optional HTTP /health."""
    if instance.status in ("STOPPED", "FAILED"):
        return "DOWN" if instance.status == "FAILED" else "UNKNOWN"

    if instance.status == "LOADING":
        if instance.pid and _is_process_alive(instance.pid):
            return "DEGRADED"
        return "DOWN"

    pid_alive = bool(instance.pid and _is_process_alive(instance.pid))
    port_listening = bool(_find_listener_pids(instance.port))

    if not pid_alive and not port_listening:
        return "DOWN"
    if not pid_alive or not port_listening:
        return "DEGRADED"
    if not probe_http_health(instance):
        return "DEGRADED"
    if _should_probe_generation(instance) and not probe_generation_health(instance):
        return "DEGRADED"
    return "HEALTHY"


def refresh_instance_health(instance: InferenceInstance) -> HealthStatus:
    """Sync runtime status, update health fields, and persist."""
    if is_manual_stop_in_progress(instance):
        return instance.health_status or "UNKNOWN"

    check_instance_status(instance)
    instance.refresh_from_db(fields=["status", "pid", "stopped_at"])

    if instance.status in ("STOPPED", "FAILED"):
        health = "DOWN" if instance.status == "FAILED" else "UNKNOWN"
    else:
        health = evaluate_instance_health(instance)
        if health == "DOWN" and instance.status in ("RUNNING", "LOADING"):
            instance.status = "FAILED"
            instance.pid = None
            instance.stopped_at = timezone.now()

    instance.health_status = health
    instance.last_health_check_at = timezone.now()
    update_fields = ["health_status", "last_health_check_at"]
    if instance.status == "FAILED":
        update_fields.extend(["status", "pid", "stopped_at"])
    instance.save(update_fields=update_fields)
    return health


def refresh_all_instance_health() -> dict[str, int]:
    """Refresh health for every instance; return counts by status."""
    counts = {"HEALTHY": 0, "DEGRADED": 0, "DOWN": 0, "UNKNOWN": 0}
    for instance in InferenceInstance.objects.all().order_by("id"):
        status = refresh_instance_health(instance)
        counts[status] = counts.get(status, 0) + 1
    return counts


def should_skip_watchdog() -> bool:
    """Avoid background health loops during tests or management commands."""
    import sys

    if os.environ.get("MLX_DISABLE_INSTANCE_WATCHDOG") == "1":
        return True
    blocked = {"test", "migrate", "makemigrations", "shell", "collectstatic"}
    return any(arg in sys.argv for arg in blocked)
