"""Anthropic Messages API ↔ Chat Completions translation."""

from __future__ import annotations

import json
import uuid
from typing import Any

from orchestrator.gateway.protocol.chat_sse import (  # noqa: F401 — re-export
    chat_sse_delta_text,
    chat_sse_finish_reason,
)
from orchestrator.gateway.protocol.stable_json import (
    canonicalize_json_value,
    stable_json_dumps,
    stabilize_chat_tools,
)
from orchestrator.gateway.router import GatewayRouteError


def anthropic_request_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    """Translate Anthropic ``/v1/messages`` body to chat.completions."""
    model = _require_model(body)
    max_tokens = _require_max_tokens(body)
    messages_in = _require_messages(body)

    messages: list[dict[str, Any]] = []
    _append_system_message(messages, body.get("system"))
    for item in messages_in:
        converted = _anthropic_message_to_chat(item)
        if converted is None:
            continue
        if isinstance(converted, list):
            messages.extend(converted)
        else:
            messages.append(converted)

    chat: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    _copy_optional_sampling_fields(chat, body)
    _copy_tools(chat, body)
    return chat


def chat_completion_to_anthropic(
    chat_payload: dict[str, Any],
    *,
    client_model_alias: str,
    include_thinking: bool = False,
) -> dict[str, Any]:
    """Translate chat.completion JSON into Anthropic Messages response."""
    message, stop_reason = _first_choice_message_and_stop(chat_payload)
    content = _assistant_content_blocks(message, include_thinking=include_thinking)
    usage = chat_payload.get("usage") if isinstance(chat_payload.get("usage"), dict) else {}
    return {
        "id": new_anthropic_message_id(),
        "type": "message",
        "role": "assistant",
        "model": client_model_alias,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def estimate_anthropic_token_count(body: dict[str, Any]) -> dict[str, Any]:
    """Best-effort token estimate for ``/v1/messages/count_tokens`` (MVP)."""
    model = _require_model(body)
    blob_parts = _token_count_blob_parts(body)
    text = "\n".join(blob_parts)
    # Rough heuristic without loading a tokenizer (~4 chars / token).
    input_tokens = max(1, (len(text) + 3) // 4) if text else 0
    return {"input_tokens": input_tokens, "model": model}


def _first_choice_message_and_stop(
    chat_payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    message: dict[str, Any] = {}
    stop_reason = "end_turn"
    choices = chat_payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return message, stop_reason
    raw_message = choices[0].get("message")
    if isinstance(raw_message, dict):
        message = raw_message
    stop_reason = _finish_to_stop_reason(choices[0].get("finish_reason"), message)
    return message, stop_reason


def _assistant_content_blocks(
    message: dict[str, Any],
    *,
    include_thinking: bool = False,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if include_thinking:
        thinking = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(thinking, str) and thinking:
            content.append({"type": "thinking", "thinking": thinking})
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            block = _tool_call_to_tool_use(call)
            if block is not None:
                content.append(block)
    if not content:
        content.append({"type": "text", "text": ""})
    return content


def _token_count_blob_parts(body: dict[str, Any]) -> list[str]:
    blob_parts: list[str] = []
    system = body.get("system")
    if isinstance(system, str):
        blob_parts.append(system)
    elif isinstance(system, list):
        blob_parts.append(_blocks_to_text(system))
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for item in messages:
        if not isinstance(item, dict):
            continue
        blob_parts.extend(_token_parts_from_message_content(item.get("content")))
    return blob_parts


def _token_parts_from_message_content(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    parts = [_blocks_to_text(content)]
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            parts.append(stable_json_dumps(block.get("input") or {}))
        elif block.get("type") == "tool_result":
            parts.append(str(block.get("content") or ""))
    return parts


def anthropic_error_envelope(message: str, error_type: str = "invalid_request_error") -> dict[str, Any]:
    """Build Anthropic-shaped ``{"type":"error","error":{...}}`` payload."""
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def anthropic_error_type_for_code(code: str) -> str:
    mapping = {
        "invalid_api_key": "authentication_error",
        "invalid_model": "invalid_request_error",
        "invalid_request": "invalid_request_error",
        "not_implemented": "invalid_request_error",
        "model_not_found": "not_found_error",
        "model_unavailable": "api_error",
        "gateway_timeout": "api_error",
        "bad_gateway": "api_error",
        "gateway_error": "api_error",
        "upstream_error": "api_error",
    }
    return mapping.get(code, "api_error")


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


def _copy_tools(chat: dict[str, Any], body: dict[str, Any]) -> None:
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        chat["tools"] = stabilize_chat_tools(
            [_anthropic_tool_to_openai(tool) for tool in tools if isinstance(tool, dict)]
        )
    if "tool_choice" in body:
        chat["tool_choice"] = _anthropic_tool_choice_to_openai(body.get("tool_choice"))


def _anthropic_tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    parameters = (
        tool.get("input_schema")
        if isinstance(tool.get("input_schema"), dict)
        else {}
    )
    return {
        "type": "function",
        "function": {
            "name": tool.get("name"),
            "description": tool.get("description") or "",
            "parameters": canonicalize_json_value(parameters),
        },
    }


def _anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    if tool_choice == "any":
        return "required"
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name")}}
    if tool_choice in {"auto", "none"}:
        return tool_choice
    return tool_choice


def _anthropic_message_to_chat(item: Any) -> dict[str, Any] | list[dict[str, Any]] | None:
    if not isinstance(item, dict):
        return None
    role = str(item.get("role") or "user")
    content = item.get("content")
    if isinstance(content, str):
        return {"role": role, "content": content}
    if not isinstance(content, list):
        return {"role": role, "content": ""}

    tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
    if tool_results:
        return _tool_results_and_followup(role, content, tool_results)

    tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
    if tool_uses:
        return _assistant_tool_use_message(content, tool_uses)

    return _role_content_with_optional_thinking(role, content)


def _tool_results_and_followup(
    role: str,
    content: list[Any],
    tool_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map tool_result blocks, then keep any trailing text/image as a user message."""
    messages: list[dict[str, Any]] = [
        {
            "role": "tool",
            "tool_call_id": str(block.get("tool_use_id") or ""),
            "content": _tool_result_content(block.get("content")),
        }
        for block in tool_results
    ]
    followup = [
        block
        for block in content
        if not (isinstance(block, dict) and block.get("type") == "tool_result")
    ]
    if not followup:
        return messages
    followup_message = _role_content_with_optional_thinking(role, followup)
    if (
        followup_message.get("content") not in (None, "")
        or followup_message.get("reasoning_content")
    ):
        messages.append(followup_message)
    return messages


def _assistant_tool_use_message(
    content: list[Any],
    tool_uses: list[dict[str, Any]],
) -> dict[str, Any]:
    text = _blocks_to_text(content)
    thinking = _blocks_to_thinking(content)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [
            {
                "id": str(block.get("id") or new_anthropic_message_id()),
                "type": "function",
                "function": {
                    "name": str(block.get("name") or ""),
                    "arguments": stable_json_dumps(block.get("input") or {}),
                },
            }
            for block in tool_uses
        ],
    }
    if thinking:
        message["reasoning_content"] = thinking
    return message


def _role_content_with_optional_thinking(role: str, content: list[Any]) -> dict[str, Any]:
    thinking = _blocks_to_thinking(content)
    images = [b for b in content if isinstance(b, dict) and b.get("type") == "image"]
    if images:
        result: dict[str, Any] = {"role": role, "content": _normalize_content_blocks(content)}
    else:
        result = {"role": role, "content": _blocks_to_text(content)}
    if thinking:
        result["reasoning_content"] = thinking
    return result


def _tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _blocks_to_text(content)
    if content is None:
        return ""
    return stable_json_dumps(content)


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


def _blocks_to_thinking(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "thinking":
            parts.append(str(block.get("thinking") or ""))
    return "".join(parts)


def _finish_to_stop_reason(finish: Any, message: dict[str, Any]) -> str:
    if finish == "length":
        return "max_tokens"
    if finish == "tool_calls" or (
        isinstance(message.get("tool_calls"), list) and message.get("tool_calls")
    ):
        return "tool_use"
    return "end_turn"


def _tool_call_to_tool_use(call: Any) -> dict[str, Any] | None:
    if not isinstance(call, dict):
        return None
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = function.get("arguments")
    parsed: Any = {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = {"raw": arguments}
    elif isinstance(arguments, dict):
        parsed = arguments
    return {
        "type": "tool_use",
        "id": str(call.get("id") or new_anthropic_message_id()),
        "name": name,
        "input": parsed if isinstance(parsed, dict) else {"value": parsed},
    }


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


def anthropic_stream_content_block_start(*, index: int = 0) -> bytes:
    return anthropic_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        },
    )


def anthropic_stream_thinking_block_start(*, index: int) -> bytes:
    return anthropic_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "thinking", "thinking": ""},
        },
    )


def anthropic_stream_tool_use_block_start(
    *,
    index: int,
    tool_id: str,
    name: str,
) -> bytes:
    return anthropic_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": {},
            },
        },
    )


def anthropic_stream_text_delta(text: str, *, index: int = 0) -> bytes:
    return anthropic_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        },
    )


def anthropic_stream_thinking_delta(text: str, *, index: int) -> bytes:
    return anthropic_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": text},
        },
    )


def anthropic_stream_input_json_delta(partial_json: str, *, index: int) -> bytes:
    return anthropic_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        },
    )


def anthropic_stream_content_block_stop(*, index: int = 0) -> bytes:
    return anthropic_sse_event(
        "content_block_stop",
        {"type": "content_block_stop", "index": index},
    )


def anthropic_stream_message_delta(
    stop_reason: str,
    *,
    output_tokens: int = 0,
    input_tokens: int = 0,
) -> bytes:
    return anthropic_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        },
    )


def anthropic_stream_message_stop() -> bytes:
    return anthropic_sse_event("message_stop", {"type": "message_stop"})


def anthropic_stream_error(message: str) -> bytes:
    return anthropic_sse_event(
        "error",
        {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        },
    )


def openai_finish_to_anthropic_stop(finish_reason: str | None) -> str:
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    return "end_turn"
