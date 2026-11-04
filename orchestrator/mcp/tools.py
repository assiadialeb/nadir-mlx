"""MCP tool definitions for Nadir instance and gateway control."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from orchestrator.mcp import services
from orchestrator.models import InferenceInstance


def _json_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


def _json_error(message: str) -> str:
    return _json_result({"data": None, "error": message, "meta": {}})


async def nadir_instances_list() -> str:
    """List inference instances (id, alias, status, port, launch mode)."""
    rows = await services.list_instances()
    return _json_result({"data": rows, "error": None, "meta": {"total": len(rows)}})


async def nadir_instance_get(instance_id: int) -> str:
    """Fetch one inference instance by numeric database id."""
    try:
        row = await services.get_instance(instance_id)
    except InferenceInstance.DoesNotExist:
        return _json_error(f"Instance {instance_id} not found.")
    return _json_result({"data": row, "error": None, "meta": {}})


async def nadir_instance_start(instance_id: int) -> str:
    """Start (or restart) a registered instance by id. No-op if already running."""
    try:
        result = await services.start_instance(instance_id)
    except InferenceInstance.DoesNotExist:
        return _json_error(f"Instance {instance_id} not found.")
    except (ValueError, RuntimeError) as exc:
        return _json_error(str(exc))
    return _json_result({"data": result, "error": None, "meta": {}})


async def nadir_instance_stop(instance_id: int) -> str:
    """Stop a registered instance by id. No-op if already stopped."""
    try:
        result = await services.stop_instance_async(instance_id)
    except InferenceInstance.DoesNotExist:
        return _json_error(f"Instance {instance_id} not found.")
    except RuntimeError as exc:
        return _json_error(str(exc))
    return _json_result({"data": result, "error": None, "meta": {}})


async def nadir_models_list() -> str:
    """List downloaded model folders under ./models with capability flags."""
    rows = await services.list_downloaded_models()
    return _json_result({"data": rows, "error": None, "meta": {"total": len(rows)}})


async def nadir_gateway_health() -> str:
    """Probe Nadir Gateway /health and list currently advertised model aliases."""
    result = await services.gateway_health()
    return _json_result({"data": result, "error": None if result.get("ok") else result.get("error"), "meta": {}})


def register_tools(mcp: FastMCP) -> None:
    """Register Nadir MCP tools on the given FastMCP instance."""
    mcp.tool()(nadir_instances_list)
    mcp.tool()(nadir_instance_get)
    mcp.tool()(nadir_instance_start)
    mcp.tool()(nadir_instance_stop)
    mcp.tool()(nadir_models_list)
    mcp.tool()(nadir_gateway_health)
