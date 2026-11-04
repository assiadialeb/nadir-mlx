"""Normalize upstream chat payloads for OpenAI-compatible clients."""

from __future__ import annotations

from typing import Any


def restore_response_model(payload: Any, client_model_alias: str) -> Any:
    """Rewrite upstream ``model`` back to the gateway alias for OpenAI clients."""
    if not isinstance(payload, dict) or "model" not in payload:
        return payload
    restored = dict(payload)
    restored["model"] = client_model_alias
    return restored


def dual_write_reasoning_fields(payload: Any) -> Any:
    """Copy ``reasoning`` to ``reasoning_content`` on message/delta blocks."""
    if not isinstance(payload, dict):
        return payload
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload

    normalized_choices: list[Any] = []
    changed = False
    for choice in choices:
        rewritten, choice_changed = _dual_write_choice(choice)
        normalized_choices.append(rewritten)
        changed = changed or choice_changed
    if not changed:
        return payload
    restored = dict(payload)
    restored["choices"] = normalized_choices
    return restored


def _dual_write_choice(choice: Any) -> tuple[Any, bool]:
    if not isinstance(choice, dict):
        return choice, False
    rewritten = dict(choice)
    changed = False
    for key in ("message", "delta"):
        block = rewritten.get(key)
        if not isinstance(block, dict):
            continue
        if "reasoning" not in block or "reasoning_content" in block:
            continue
        block = dict(block)
        block["reasoning_content"] = block["reasoning"]
        rewritten[key] = block
        changed = True
    return rewritten, changed


def normalize_client_chat_payload(payload: Any, client_model_alias: str) -> Any:
    """Restore gateway alias and dual-write reasoning fields for clients."""
    restored = restore_response_model(payload, client_model_alias)
    return dual_write_reasoning_fields(restored)
