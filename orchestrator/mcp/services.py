"""Sync helpers used by Nadir MCP tools (Django ORM / server_manager)."""

from __future__ import annotations

import os
from typing import Any

import httpx
from asgiref.sync import sync_to_async

from orchestrator.gateway_aliases import instance_gateway_aliases, instance_primary_gateway_alias
from orchestrator.models import InferenceInstance
from orchestrator.server_manager import request_instance_stop, restart_instance
from orchestrator.ui_selectors import list_installed_models


def _serialize_instance(instance: InferenceInstance) -> dict[str, Any]:
    return {
        "id": instance.id,
        "model_name": instance.model_name,
        "port": instance.port,
        "launch_mode": instance.launch_mode,
        "status": instance.status,
        "health_status": instance.health_status,
        "pid": instance.pid,
        "gateway_alias": instance_primary_gateway_alias(instance),
        "gateway_aliases": instance_gateway_aliases(instance),
        "last_used_at": instance.last_used_at.isoformat() if instance.last_used_at else None,
        "stopped_at": instance.stopped_at.isoformat() if instance.stopped_at else None,
    }


def list_instances_sync() -> list[dict[str, Any]]:
    """Return all inference instances for MCP listing."""
    rows = InferenceInstance.objects.all().order_by("model_name", "port")
    return [_serialize_instance(row) for row in rows]


def get_instance_sync(instance_id: int) -> dict[str, Any]:
    """Return one instance or raise InferenceInstance.DoesNotExist."""
    instance = InferenceInstance.objects.get(pk=instance_id)
    return _serialize_instance(instance)


def start_instance_sync(instance_id: int) -> dict[str, Any]:
    """Start or restart a registered instance by id."""
    instance = InferenceInstance.objects.get(pk=instance_id)
    if instance.status == "RUNNING":
        return {
            "ok": True,
            "message": "Instance already running.",
            "instance": _serialize_instance(instance),
        }
    updated = restart_instance(instance)
    return {
        "ok": True,
        "message": f"Instance started on port {updated.port}.",
        "instance": _serialize_instance(updated),
    }


def stop_instance_sync(instance_id: int) -> dict[str, Any]:
    """Stop a registered instance by id."""
    instance = InferenceInstance.objects.get(pk=instance_id)
    if instance.status == "STOPPED":
        return {
            "ok": True,
            "message": "Instance already stopped.",
            "instance": _serialize_instance(instance),
        }
    stopped_now = request_instance_stop(instance)
    instance.refresh_from_db()
    message = (
        f"Instance on port {instance.port} stopped."
        if stopped_now
        else f"Stop queued for port {instance.port}; a start/stop is in progress."
    )
    return {
        "ok": True,
        "message": message,
        "instance": _serialize_instance(instance),
    }


def list_downloaded_models_sync() -> list[dict[str, Any]]:
    """Return downloaded model folders with capability flags."""
    return list_installed_models()


def gateway_base_url() -> str:
    """Build the local gateway base URL from env."""
    host = (os.environ.get("NADIR_GATEWAY_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    port = (os.environ.get("NADIR_GATEWAY_PORT") or "11380").strip() or "11380"
    return f"http://{host}:{port}"


def gateway_health_sync() -> dict[str, Any]:
    """Probe gateway /health (and optionally /v1/models)."""
    base = gateway_base_url()
    from orchestrator.gateway_client_auth import gateway_auth_headers

    headers = gateway_auth_headers()
    try:
        with httpx.Client(timeout=5.0) as client:
            health = client.get(f"{base}/health", headers=headers)
            models = client.get(f"{base}/v1/models", headers=headers)
        model_ids: list[str] = []
        if models.status_code == 200:
            payload = models.json()
            model_ids = [str(item.get("id")) for item in payload.get("data") or [] if item.get("id")]
        return {
            "ok": health.status_code == 200,
            "gateway_url": base,
            "health_status": health.status_code,
            "health_body": health.text[:500],
            "models_status": models.status_code,
            "model_ids": model_ids,
        }
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "gateway_url": base,
            "error": str(exc),
        }


list_instances = sync_to_async(list_instances_sync, thread_sensitive=True)
get_instance = sync_to_async(get_instance_sync, thread_sensitive=True)
start_instance = sync_to_async(start_instance_sync, thread_sensitive=True)
stop_instance_async = sync_to_async(stop_instance_sync, thread_sensitive=True)
list_downloaded_models = sync_to_async(list_downloaded_models_sync, thread_sensitive=True)
gateway_health = sync_to_async(gateway_health_sync, thread_sensitive=True)
