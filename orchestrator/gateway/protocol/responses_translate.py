"""Stateless OpenAI Responses API ↔ Chat Completions translation."""

from __future__ import annotations

import json
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
        "max_tool_calls",
    }
)

PASSTHROUGH_CHAT_FIELDS = frozenset(

    {
        "temperature",
        "top_p",
        "stream",
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
            chat["response_format"] = _responses_format_to_chat(fmt)

    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        chat_tools = [
            _responses_tool_to_chat(tool)
            for tool in tools
            if isinstance(tool, dict) and _is_function_tool(tool)
        ]
        if chat_tools:
            chat["tools"] = chat_tools

    if "tool_choice" in body:
        chat["tool_choice"] = _responses_tool_choice_to_chat(body.get("tool_choice"))

    for key in PASSTHROUGH_CHAT_FIELDS:
        if key in body:
            chat[key] = body[key]
    return chat


def _responses_format_to_chat(fmt: dict[str, Any]) -> dict[str, Any]:
    """Map Responses ``text.format`` into Chat Completions ``response_format``."""
    if fmt.get("type") != "json_schema":
        return fmt
    if isinstance(fmt.get("json_schema"), dict):
        return fmt
    return {
        "type": "json_schema",
        "json_schema": {
            "name": fmt.get("name") or "response",
            "schema": fmt.get("schema") if isinstance(fmt.get("schema"), dict) else {},
            "strict": fmt.get("strict"),
        },
    }


def _is_function_tool(tool: dict[str, Any]) -> bool:
    tool_type = tool.get("type")
    if tool_type not in (None, "function"):
        return False
    if isinstance(tool.get("function"), dict) and tool["function"].get("name"):
        return True
    name = tool.get("name")
    return isinstance(name, str) and bool(name)


def _responses_tool_to_chat(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return tool
    name = tool.get("name")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description") or "",
            "parameters": tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {},
        },
    }


def _responses_tool_choice_to_chat(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice.get("name")
        if isinstance(tool_choice.get("function"), dict):
            return tool_choice
        return {"type": "function", "function": {"name": name}}
    return tool_choice


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

    pending_tool_calls: list[dict[str, Any]] = []

    def flush_tool_calls() -> None:
        if not pending_tool_calls:
            return
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": list(pending_tool_calls),
            }
        )
        pending_tool_calls.clear()

    for item in input_value:
        if isinstance(item, dict) and item.get("type") == "function_call":
            call = _function_call_to_tool_call(item)
            if call is not None:
                pending_tool_calls.append(call)
            continue
        flush_tool_calls()
        message = _coerce_input_item(item)
        if message is not None:
            messages.append(message)
    flush_tool_calls()
    if not any(msg.get("role") != "system" for msg in messages):
        raise GatewayRouteError(
            status_code=400,
            code="invalid_request",
            message="input must include at least one user/assistant/tool message.",
            param="input",
        )
    return messages


def _normalize_role(role: str) -> str:
    if role == "developer":
        return "system"
    return role


def _coerce_input_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    item_type = item.get("type")
    if item_type == "function_call_output":
        return _function_call_output_to_tool_message(item)
    if item_type in {None, "message"} and "role" in item:
        role = _normalize_role(str(item.get("role") or "user"))
        content = item.get("content", "")
        return {"role": role, "content": _flatten_content(content)}
    if "role" in item and "content" in item:
        return {
            "role": _normalize_role(str(item["role"])),
            "content": _flatten_content(item.get("content")),
        }
    return None


def _function_call_to_tool_call(item: dict[str, Any]) -> dict[str, Any] | None:
    name = str(item.get("name") or "")
    if not name:
        return None
    call_id = str(item.get("call_id") or item.get("id") or _new_id("call"))
    arguments = item.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, separators=(",", ":"))
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _function_call_output_to_tool_message(item: dict[str, Any]) -> dict[str, Any]:
    call_id = str(item.get("call_id") or "")
    output = item.get("output")
    if not isinstance(output, str):
        output = json.dumps(output, separators=(",", ":")) if output is not None else ""
    return {"role": "tool", "tool_call_id": call_id, "content": output}


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
            parts.append(_normalize_image_part(part))
            continue
        parts.append(part)
    if parts:
        converted: list[dict[str, Any]] = [{"type": "text", "text": text} for text in texts]
        converted.extend(parts)
        return converted
    return "".join(texts)


def _normalize_image_part(part: dict[str, Any]) -> dict[str, Any]:
    image_url = part.get("image_url")
    if isinstance(image_url, str):
        return {"type": "image_url", "image_url": {"url": image_url}}
    if isinstance(image_url, dict) and "url" in image_url:
        return {"type": "image_url", "image_url": image_url}
    return part


def chat_completion_to_response(
    chat_payload: dict[str, Any],
    *,
    client_model_alias: str,
) -> dict[str, Any]:
    """Translate a chat.completion JSON body into a Responses object."""
    message_id = _new_id("msg")
    response_id = str(chat_payload.get("id") or _new_id("resp")).replace(
        "chatcmpl-",
        "resp_",
        1,
    )
    created = int(chat_payload.get("created") or time.time())
    usage = chat_payload.get("usage") if isinstance(chat_payload.get("usage"), dict) else {}
    output = _output_items_from_chat(chat_payload, message_id=message_id)
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": client_model_alias,
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def _output_items_from_chat(chat_payload: dict[str, Any], *, message_id: str) -> list[dict[str, Any]]:
    choices = chat_payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return [_text_message_item(message_id, "")]
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return [_text_message_item(message_id, _assistant_text_from_chat(chat_payload))]

    items: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        items.append(
            {
                "id": f"rs_{message_id[4:]}" if message_id.startswith("msg_") else _new_id("rs"),
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning}],
            }
        )
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            converted = _tool_call_to_function_call_item(call)
            if converted is not None:
                items.append(converted)
    text = message.get("content")
    if isinstance(text, str) and text:
        items.append(_text_message_item(message_id, text))
    elif not items:
        items.append(_text_message_item(message_id, _assistant_text_from_chat(chat_payload)))
    return items


def _tool_call_to_function_call_item(call: Any) -> dict[str, Any] | None:
    if not isinstance(call, dict):
        return None
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, separators=(",", ":"))
    return {
        "id": str(call.get("id") or _new_id("fc")),
        "type": "function_call",
        "call_id": str(call.get("id") or _new_id("call")),
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _text_message_item(message_id: str, text: str) -> dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text}],
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
        "sequence_number": 0,
        "text": "",
        "failed": False,
        "error_message": "",
        "usage": {},
    }


def next_sequence(meta: dict[str, Any]) -> int:
    meta["sequence_number"] = int(meta.get("sequence_number") or 0) + 1
    return int(meta["sequence_number"])
