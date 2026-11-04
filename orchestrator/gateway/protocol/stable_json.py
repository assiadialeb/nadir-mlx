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
    """Normalize tool-call argument payloads to a byte-stable JSON string.

    Used when re-injecting tool results / calls into history, where clients
    may emit non-deterministic key order. Do **not** use this on tool *schemas*
    sent to the model (property order can affect argument generation).
    """
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
        return stable_json_dumps(parsed)
    return stable_json_dumps(arguments if arguments is not None else {})


def stabilize_chat_tools(tools: Any) -> list[Any] | Any:
    """Preserve client tool list and schema key order.

    Prefix KV reuse requires the shared system+tools prefix to be identical
    across turns. Claude Code / Codex already send stable JSON; sorting
    ``parameters`` / ``input_schema`` keys would only change the order the
    model sees for argument generation without helping the cache.
    """
    if not isinstance(tools, list):
        return tools
    stabilized: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            stabilized.append(tool)
            continue
        stabilized.append(_copy_tool_shallow(tool))
    return stabilized


def _copy_tool_shallow(tool: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy tool wrappers without reordering nested schema keys."""
    out = dict(tool)
    function = out.get("function")
    if isinstance(function, dict):
        out["function"] = dict(function)
    return out
