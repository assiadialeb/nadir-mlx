"""Stateless OpenAI Responses API ↔ Chat Completions translation."""

from __future__ import annotations

import time
import uuid
from typing import Any

from orchestrator.gateway.router import GatewayRouteError

UNSUPPORTED_RESPONSE_FIELDS = frozenset(
    {
        "previous_response_id",
        "conversation",
        "background",
        "prompt",
        "truncation",
        "reasoning",
        "max_tool_calls",
        "tool_choice",
    }
)

PASSTHROUGH_CHAT_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "stream",
        "tools",
        "user",
        "metadata",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
    }
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def responses_request_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    """Translate a Responses create body into a chat.completions body."""
    unsupported = sorted(UNSUPPORTED_RESPONSE_FIELDS.intersection(body))
    if unsupported:
        raise GatewayRouteError(
            status_code=400,
            code="invalid_request",
            message=(
                "Unsupported Responses fields for Nadir MVP: "
                + ", ".join(unsupported)
            ),
            param=unsupported[0],
        )

    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise GatewayRouteError(
            status_code=400,
            code="invalid_model",
            message="The model field is required.",
            param="model",
        )

    messages = _input_to_messages(body.get("input"), body.get("instructions"))
    chat: dict[str, Any] = {"model": model.strip(), "messages": messages}

    max_output_tokens = body.get("max_output_tokens")
    if max_output_tokens is not None:
        chat["max_tokens"] = max_output_tokens

    text = body.get("text")
    if isinstance(text, dict):
        fmt = text.get("format")
        if isinstance(fmt, dict) and fmt.get("type") in {"json_object", "json_schema"}:
            chat["response_format"] = fmt

    for key in PASSTHROUGH_CHAT_FIELDS:
        if key in body:
            chat[key] = body[key]
    return chat


def _input_to_messages(input_value: Any, instructions: Any) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})

    if input_value is None:
        raise GatewayRouteError(
            status_code=400,
            code="invalid_request",
            message="The input field is required.",
            param="input",
        )
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages
    if not isinstance(input_value, list):
        raise GatewayRouteError(
            status_code=400,
            code="invalid_request",
            message="input must be a string or an array of messages.",
            param="input",
        )

    for item in input_value:
        message = _coerce_input_item(item)
        if message is not None:
            messages.append(message)
    if not any(msg.get("role") != "system" for msg in messages):
        raise GatewayRouteError(
            status_code=400,
            code="invalid_request",
            message="input must include at least one user/assistant message.",
            param="input",
        )
    return messages


def _coerce_input_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    if item.get("type") in {None, "message"} and "role" in item:
        role = str(item.get("role") or "user")
        content = item.get("content", "")
        return {"role": role, "content": _flatten_content(content)}
    if "role" in item and "content" in item:
        return {
            "role": str(item["role"]),
            "content": _flatten_content(item.get("content")),
        }
    return None


def _flatten_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    texts: list[str] = []
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text"} and "text" in part:
            texts.append(str(part["text"]))
            continue
        if part_type in {"input_image", "image_url"}:
            parts.append(part)
            continue
        parts.append(part)
    if parts:
        converted: list[dict[str, Any]] = []
        for text in texts:
            converted.append({"type": "text", "text": text})
        converted.extend(parts)
        return converted
    return "".join(texts)


def chat_completion_to_response(
    chat_payload: dict[str, Any],
    *,
    client_model_alias: str,
) -> dict[str, Any]:
    """Translate a chat.completion JSON body into a Responses object."""
    text = _assistant_text_from_chat(chat_payload)
    message_id = _new_id("msg")
    response_id = str(chat_payload.get("id") or _new_id("resp")).replace(
        "chatcmpl-",
        "resp_",
        1,
    )
    created = int(chat_payload.get("created") or time.time())
    usage = chat_payload.get("usage") if isinstance(chat_payload.get("usage"), dict) else {}
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": client_model_alias,
        "output": [
            {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def _assistant_text_from_chat(chat_payload: dict[str, Any]) -> str:
    choices = chat_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        return content if isinstance(content, str) else ""
    text = first.get("text")
    return text if isinstance(text, str) else ""


def build_response_stream_events(
    *,
    client_model_alias: str,
    response_id: str,
    output_index: int = 0,
) -> dict[str, Any]:
    """Return shared ids used while translating chat SSE into Responses events."""
    return {
        "response_id": response_id,
        "message_id": _new_id("msg"),
        "model": client_model_alias,
        "output_index": output_index,
        "created_at": int(time.time()),
    }
