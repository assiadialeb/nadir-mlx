"""Qualification helpers for OpenAI chat extensions relayed by Nadir Gateway."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

STRUCTURED_OUTPUT_TYPES: Final[frozenset[str]] = frozenset({"json_object", "json_schema"})
VISION_IMAGE_CONTENT_TYPES: Final[frozenset[str]] = frozenset({"image_url", "input_image"})


def prepare_chat_upstream_body(body: dict[str, Any], upstream_model: str) -> dict[str, Any]:
    """Clone the client payload and rewrite only the model field for mlx-lm / mlx-vlm."""
    upstream = deepcopy(body)
    upstream["model"] = upstream_model
    return upstream


def iter_message_content_parts(messages: Any) -> list[dict[str, Any]]:
    """Return OpenAI-style content parts from chat messages."""
    parts: list[dict[str, Any]] = []
    if not isinstance(messages, list):
        return parts
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict):
                parts.append(part)
    return parts


def has_vision_content(messages: Any) -> bool:
    """Return True when messages include image content blocks."""
    return count_vision_images(messages) > 0


def count_vision_images(messages: Any) -> int:
    """Count image_url / input_image blocks in multimodal messages."""
    count = 0
    for part in iter_message_content_parts(messages):
        part_type = part.get("type")
        if isinstance(part_type, str) and part_type in VISION_IMAGE_CONTENT_TYPES:
            count += 1
    return count


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
