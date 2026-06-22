"""Qualification helpers for OpenAI chat extensions relayed by Nadir Gateway."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

STRUCTURED_OUTPUT_TYPES: Final[frozenset[str]] = frozenset({"json_object", "json_schema"})


def prepare_chat_upstream_body(body: dict[str, Any], upstream_model: str) -> dict[str, Any]:
    """Clone the client payload and rewrite only the model field for mlx-lm."""
    upstream = deepcopy(body)
    upstream["model"] = upstream_model
    return upstream


def has_tool_definitions(body: dict[str, Any]) -> bool:
    """Return True when the client sent a non-empty tools array."""
    tools = body.get("tools")
    return isinstance(tools, list) and len(tools) > 0


def structured_output_type(body: dict[str, Any]) -> str | None:
    """Return json_object/json_schema when response_format requests structured output."""
    response_format = body.get("response_format")
    if not isinstance(response_format, dict):
        return None
    mode = response_format.get("type")
    if not isinstance(mode, str):
        return None
    normalized = mode.strip().lower()
    if normalized in STRUCTURED_OUTPUT_TYPES:
        return normalized
    return None
