"""Anthropic Messages API ↔ Chat Completions translation."""

from __future__ import annotations

import json
import uuid
from typing import Any

from orchestrator.gateway.router import GatewayRouteError


def anthropic_request_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    """Translate Anthropic ``/v1/messages`` body to chat.completions."""
    model = _require_model(body)
    max_tokens = _require_max_tokens(body)
    messages_in = _require_messages(body)

    messages: list[dict[str, Any]] = []
    _append_system_message(messages, body.get("system"))
    for item in messages_in:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        messages.append({"role": role, "content": _normalize_content(item.get("content"))})

    chat: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    _copy_optional_sampling_fields(chat, body)
    return chat


def chat_completion_to_anthropic(
    chat_payload: dict[str, Any],
    *,
    client_model_alias: str,
) -> dict[str, Any]:
    """Translate chat.completion JSON into Anthropic Messages response."""
    text = ""
    stop_reason = "end_turn"
    choices = chat_payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            text = message["content"]
        finish = choices[0].get("finish_reason")
        if finish == "length":
            stop_reason = "max_tokens"
        elif finish == "stop":
            stop_reason = "end_turn"
    usage = chat_payload.get("usage") if isinstance(chat_payload.get("usage"), dict) else {}
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": client_model_alias,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _require_model(body: dict[str, Any]) -> str:
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise GatewayRouteError(
            status_code=400,
            code="invalid_model",
            message="The model field is required.",
            param="model",
        )
    return model.strip()


def _require_max_tokens(body: dict[str, Any]) -> int:
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise GatewayRouteError(
            status_code=400,
            code="invalid_request",
            message="max_tokens is required and must be a positive integer.",
            param="max_tokens",
        )
    return max_tokens


def _require_messages(body: dict[str, Any]) -> list[Any]:
    messages_in = body.get("messages")
    if not isinstance(messages_in, list) or not messages_in:
        raise GatewayRouteError(
            status_code=400,
            code="invalid_request",
            message="messages must be a non-empty array.",
            param="messages",
        )
    return messages_in


def _append_system_message(messages: list[dict[str, Any]], system: Any) -> None:
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})
        return
    if isinstance(system, list):
        text = _blocks_to_text(system)
        if text:
            messages.append({"role": "system", "content": text})


def _copy_optional_sampling_fields(chat: dict[str, Any], body: dict[str, Any]) -> None:
    if "temperature" in body:
        chat["temperature"] = body["temperature"]
    if "top_p" in body:
        chat["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        chat["stop"] = body["stop_sequences"]
    if body.get("stream"):
        chat["stream"] = True


def _normalize_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _normalize_content_blocks(content)
    return ""


def _normalize_content_blocks(content: list[Any]) -> Any:
    text = _blocks_to_text(content)
    images = [part for part in content if isinstance(part, dict) and part.get("type") == "image"]
    if not images:
        return text
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    for image in images:
        converted = _image_block_to_openai(image)
        if converted is not None:
            parts.append(converted)
    return parts or text


def _image_block_to_openai(image: dict[str, Any]) -> dict[str, Any] | None:
    source = image.get("source") if isinstance(image.get("source"), dict) else {}
    if source.get("type") != "base64" or "data" not in source:
        return None
    media = source.get("media_type") or "image/png"
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{media};base64,{source['data']}",
        },
    }


def _blocks_to_text(blocks: list[Any]) -> str:
    texts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(str(block.get("text") or ""))
        elif isinstance(block, str):
            texts.append(block)
    return "".join(texts)


def new_anthropic_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def anthropic_sse_event(event_name: str, payload: dict[str, Any]) -> bytes:
    """Encode one Anthropic Messages SSE event."""
    body = json.dumps(payload, separators=(",", ":"))
    return f"event: {event_name}\ndata: {body}\n\n".encode()


def anthropic_stream_message_start(
    *,
    message_id: str,
    client_model_alias: str,
) -> bytes:
    return anthropic_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": client_model_alias,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )


def anthropic_stream_content_block_start() -> bytes:
    return anthropic_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )


def anthropic_stream_text_delta(text: str) -> bytes:
    return anthropic_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
    )


def anthropic_stream_content_block_stop() -> bytes:
    return anthropic_sse_event(
        "content_block_stop",
        {"type": "content_block_stop", "index": 0},
    )


def anthropic_stream_message_delta(stop_reason: str) -> bytes:
    return anthropic_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 0},
        },
    )


def anthropic_stream_message_stop() -> bytes:
    return anthropic_sse_event("message_stop", {"type": "message_stop"})


def chat_sse_delta_text(line: bytes) -> str | None:
    """Extract assistant text delta from one OpenAI chat SSE ``data:`` line."""
    if not line.startswith(b"data:"):
        return None
    raw = line[5:].strip()
    if not raw or raw == b"[DONE]":
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return None
    text = delta.get("content")
    if not isinstance(text, str) or text == "":
        return None
    return text


def chat_sse_finish_reason(line: bytes) -> str | None:
    """Return OpenAI ``finish_reason`` from a chat SSE line when present."""
    if not line.startswith(b"data:"):
        return None
    raw = line[5:].strip()
    if not raw or raw == b"[DONE]":
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    finish = choices[0].get("finish_reason")
    return finish if isinstance(finish, str) and finish else None


def openai_finish_to_anthropic_stop(finish_reason: str | None) -> str:
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"
