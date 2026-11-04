"""Byte-stable JSON helpers for gateway façades (MLX-125 prompt-cache prefixes)."""

from __future__ import annotations

import json
from typing import Any


def canonicalize_json_value(value: Any) -> Any:
    """Return a deep copy with object keys sorted (lists keep client order)."""
    if isinstance(value, dict):
        return {key: canonicalize_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonicalize_json_value(item) for item in value]
    return value


def stable_json_dumps(value: Any) -> str:
    """Serialize ``value`` with sorted keys and compact separators."""
    return json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    )


def stable_tool_arguments(arguments: Any) -> str:
    """Normalize tool-call argument payloads to a byte-stable JSON string."""
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
        return stable_json_dumps(parsed)
    return stable_json_dumps(arguments if arguments is not None else {})


def stabilize_chat_tools(tools: Any) -> list[Any] | Any:
    """Keep tool list order; canonicalize nested schema/parameter objects.

    mlx-lm prefix KV reuse requires the shared system+tools prefix to be
    identical across turns. Clients must keep tool *order* stable; Nadir
    only normalizes object key order inside each tool definition.
    """
    if not isinstance(tools, list):
        return tools
    stabilized: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            stabilized.append(tool)
            continue
        stabilized.append(_stabilize_one_tool(tool))
    return stabilized


def _stabilize_one_tool(tool: dict[str, Any]) -> dict[str, Any]:
    out = dict(tool)
    function = out.get("function")
    if isinstance(function, dict):
        fn = dict(function)
        if isinstance(fn.get("parameters"), dict):
            fn["parameters"] = canonicalize_json_value(fn["parameters"])
        out["function"] = fn
        return out
    if isinstance(out.get("parameters"), dict):
        out["parameters"] = canonicalize_json_value(out["parameters"])
    if isinstance(out.get("input_schema"), dict):
        out["input_schema"] = canonicalize_json_value(out["input_schema"])
    return out
