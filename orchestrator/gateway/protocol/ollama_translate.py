"""Ollama HTTP API ↔ OpenAI-compatible translation helpers."""

from __future__ import annotations

import json
from typing import Any

from orchestrator.gateway.protocol.chat_sse import (  # noqa: F401 — re-export
    chat_sse_delta_text,
    tool_calls_as_ollama,
)
from orchestrator.gateway.router import GatewayRouteError

_FINISH_TO_DONE_REASON = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "stop",
}

_DEFAULT_OLLAMA_VERSION = "0.6.0"


def ollama_chat_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Translate Ollama ``/api/chat`` body to chat.completions."""
    model = _require_model(body)
    messages_in = body.get("messages")
    if not isinstance(messages_in, list) or not messages_in:
        raise GatewayRouteError(
            status_code=400,
            code="invalid_request",
            message="messages must be a non-empty array.",
            param="messages",
        )
    messages = _normalize_ollama_messages(messages_in)
    if not messages:
        raise GatewayRouteError(
            status_code=400,
            code="invalid_request",
            message="messages must be a non-empty array.",
            param="messages",
        )
    chat: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(body.get("stream", True)),
    }
    _apply_ollama_options(chat, body.get("options"))
    _apply_ollama_format(chat, body.get("format"))
    _copy_tools(chat, body)
    return chat


def ollama_generate_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Translate Ollama ``/api/generate`` body to chat.completions."""
    model = _require_model(body)
    prompt = body.get("prompt")
    if not isinstance(prompt, str):
        raise GatewayRouteError(
            status_code=400,
            code="invalid_request",
            message="prompt is required.",
            param="prompt",
        )
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})
    user_message: dict[str, Any] = {"role": "user", "content": prompt}
    images = body.get("images")
    if isinstance(images, list) and images:
        user_message = _normalize_ollama_message(
            {"role": "user", "content": prompt, "images": images}
        )
    messages.append(user_message)
    chat: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(body.get("stream", True)),
    }
    _apply_ollama_options(chat, body.get("options"))
    _apply_ollama_format(chat, body.get("format"))
    return chat


def openai_chat_to_ollama(
    chat_payload: dict[str, Any],
    *,
    client_model_alias: str,
) -> dict[str, Any]:
    """Translate chat.completion JSON into Ollama chat response."""
    content, finish_reason, usage = _extract_chat_fields(chat_payload)
    message: dict[str, Any] = {"role": "assistant", "content": content}
    tool_calls = _extract_tool_calls(chat_payload)
    if tool_calls:
        message["tool_calls"] = tool_calls
    payload: dict[str, Any] = {
        "model": client_model_alias,
        "message": message,
        "done": True,
        "done_reason": _FINISH_TO_DONE_REASON.get(finish_reason or "stop", "stop"),
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }
    return payload


def openai_chat_to_ollama_generate(
    chat_payload: dict[str, Any],
    *,
    client_model_alias: str,
) -> dict[str, Any]:
    """Translate chat.completion JSON into Ollama generate response."""
    content, finish_reason, usage = _extract_chat_fields(chat_payload)
    return {
        "model": client_model_alias,
        "response": content,
        "done": True,
        "done_reason": _FINISH_TO_DONE_REASON.get(finish_reason or "stop", "stop"),
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }


def ollama_embed_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Translate Ollama ``/api/embed`` / ``/api/embeddings`` body to embeddings."""
    model = _require_model(body)
    if "input" in body:
        inputs = body["input"]
    elif "prompt" in body:
        inputs = body["prompt"]
    else:
        raise GatewayRouteError(
            status_code=400,
            code="invalid_request",
            message="input or prompt is required.",
            param="input",
        )
    return {"model": model, "input": inputs}


def openai_embeddings_to_ollama(
    embeddings_payload: dict[str, Any],
    *,
    client_model_alias: str,
    legacy: bool = False,
) -> dict[str, Any]:
    """Translate embeddings JSON into Ollama embed response.

    ``legacy=True`` targets ``POST /api/embeddings`` singular ``embedding``.
    """
    vectors: list[list[float]] = []
    data = embeddings_payload.get("data")
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict) and isinstance(row.get("embedding"), list):
                vectors.append(row["embedding"])
    if legacy:
        first = vectors[0] if vectors else []
        return {"embedding": first}
    return {"model": client_model_alias, "embeddings": vectors}


def models_list_to_ollama_tags(models_payload: dict[str, Any]) -> dict[str, Any]:
    """Translate ``GET /v1/models`` payload into Ollama ``/api/tags``."""
    return {"models": _models_from_list_payload(models_payload)}


def models_list_to_ollama_ps(models_payload: dict[str, Any]) -> dict[str, Any]:
    """Translate running models into Ollama ``/api/ps`` (MVP)."""
    return {"models": _models_from_list_payload(models_payload)}


def ollama_show_stub(
    name: str,
    *,
    launch_mode: str = "TEXT",
    tool_parser_type: str = "",
) -> dict[str, Any]:
    """Minimal ``/api/show`` payload so Open WebUI bootstrap probes succeed."""
    return {
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "nadir",
            "families": ["nadir"],
            "parameter_size": "",
            "quantization_level": "",
        },
        "model_info": {"general.architecture": "nadir"},
        "capabilities": ollama_capabilities_for_mode(
            launch_mode,
            tool_parser_type=tool_parser_type,
        ),
        "modified_at": "",
        "name": name,
        "model": name,
    }


def ollama_capabilities_for_mode(
    launch_mode: str,
    *,
    tool_parser_type: str = "",
) -> list[str]:
    mode = (launch_mode or "TEXT").upper()
    if mode == "EMBEDDING":
        return ["embedding"]
    capabilities = ["completion"]
    if mode == "MULTIMODAL":
        capabilities.append("vision")
    if tool_parser_type:
        capabilities.append("tools")
    return capabilities


def ollama_version_payload(version: str = _DEFAULT_OLLAMA_VERSION) -> dict[str, str]:
    return {"version": version}


def ollama_chat_stream_chunk(
    *,
    client_model_alias: str,
    content: str,
    done: bool,
    done_reason: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    thinking: str | None = None,
    prompt_eval_count: int | None = None,
    eval_count: int | None = None,
    error: str | None = None,
) -> bytes:
    """Encode one Ollama ``/api/chat`` NDJSON stream line."""
    payload: dict[str, Any] = {
        "model": client_model_alias,
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    if thinking:
        payload["message"]["thinking"] = thinking
    if tool_calls:
        payload["message"]["tool_calls"] = tool_calls
    if done and done_reason is not None:
        payload["done_reason"] = done_reason
    if prompt_eval_count is not None:
        payload["prompt_eval_count"] = prompt_eval_count
    if eval_count is not None:
        payload["eval_count"] = eval_count
    if error:
        payload["error"] = error
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode()


def ollama_generate_stream_chunk(
    *,
    client_model_alias: str,
    content: str,
    done: bool,
    done_reason: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_eval_count: int | None = None,
    eval_count: int | None = None,
    error: str | None = None,
) -> bytes:
    """Encode one Ollama ``/api/generate`` NDJSON stream line."""
    payload: dict[str, Any] = {
        "model": client_model_alias,
        "response": content,
        "done": done,
    }
    if done and done_reason is not None:
        payload["done_reason"] = done_reason
    if prompt_eval_count is not None:
        payload["prompt_eval_count"] = prompt_eval_count
    if eval_count is not None:
        payload["eval_count"] = eval_count
    if error:
        payload["error"] = error
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode()


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


def _normalize_ollama_messages(messages_in: list[Any]) -> list[dict[str, Any]]:
    """Normalize Ollama chat history, preserving assistant tool_calls for mlx-lm."""
    messages: list[dict[str, Any]] = []
    pending_tool_ids: dict[str, list[str]] = {}
    for item in messages_in:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_ollama_message(item, pending_tool_ids=pending_tool_ids)
        messages.append(normalized)
        pending_tool_ids = _pending_ids_after_message(normalized, pending_tool_ids)
    return messages


def _pending_ids_after_message(
    normalized: dict[str, Any],
    pending_tool_ids: dict[str, list[str]],
) -> dict[str, list[str]]:
    if normalized.get("role") != "assistant":
        return pending_tool_ids
    tool_calls = normalized.get("tool_calls")
    if not isinstance(tool_calls, list):
        return pending_tool_ids
    refreshed: dict[str, list[str]] = {}
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name")
        call_id = call.get("id")
        if isinstance(name, str) and name and isinstance(call_id, str):
            refreshed.setdefault(name, []).append(call_id)
    return refreshed


def _normalize_ollama_message(
    item: dict[str, Any],
    *,
    pending_tool_ids: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    role = str(item.get("role") or "user")
    if role == "tool":
        return _normalize_ollama_tool_message(item, pending_tool_ids=pending_tool_ids or {})

    message = _base_role_content_message(role, item.get("content"), item.get("images"))
    if role == "assistant":
        _attach_assistant_extras(message, item)
    return message


def _base_role_content_message(role: str, content: Any, images: Any) -> dict[str, Any]:
    if isinstance(images, list) and images:
        parts = _image_content_parts(content, images)
        return {"role": role, "content": parts or (content if content is not None else "")}
    return {"role": role, "content": content if content is not None else ""}


def _image_content_parts(content: Any, images: list[Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if isinstance(content, str) and content:
        parts.append({"type": "text", "text": content})
    for image in images:
        if not isinstance(image, str) or not image:
            continue
        url = image if image.startswith("data:") else f"data:image/png;base64,{image}"
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _attach_assistant_extras(message: dict[str, Any], item: dict[str, Any]) -> None:
    tool_calls = _ollama_tool_calls_to_openai(item.get("tool_calls"))
    if tool_calls:
        message["tool_calls"] = tool_calls
        if not message.get("content"):
            message["content"] = None
    thinking = item.get("thinking")
    if isinstance(thinking, str) and thinking:
        message["reasoning_content"] = thinking


def _normalize_ollama_tool_message(
    item: dict[str, Any],
    *,
    pending_tool_ids: dict[str, list[str]],
) -> dict[str, Any]:
    content = item.get("content")
    return {
        "role": "tool",
        "tool_call_id": _resolve_tool_call_id(item, pending_tool_ids),
        "content": content if isinstance(content, str) else json.dumps(content or ""),
    }


def _resolve_tool_call_id(
    item: dict[str, Any],
    pending_tool_ids: dict[str, list[str]],
) -> str:
    tool_call_id = item.get("tool_call_id")
    if isinstance(tool_call_id, str) and tool_call_id:
        return tool_call_id
    tool_name = item.get("tool_name") or item.get("name")
    if isinstance(tool_name, str) and tool_name in pending_tool_ids:
        ids = pending_tool_ids[tool_name]
        return ids.pop(0) if ids else f"call_{tool_name}"
    if isinstance(tool_name, str) and tool_name:
        return f"call_{tool_name}"
    return "call_0"


def _ollama_tool_calls_to_openai(tool_calls: Any) -> list[dict[str, Any]]:
    """Convert Ollama tool_calls (args as objects) into OpenAI chat tool_calls (args as JSON strings)."""
    if not isinstance(tool_calls, list):
        return []
    converted: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls):
        converted_call = _one_ollama_tool_call_to_openai(call, index)
        if converted_call is not None:
            converted.append(converted_call)
    return converted


def _one_ollama_tool_call_to_openai(call: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(call, dict):
        return None
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, separators=(",", ":"))
    elif not isinstance(arguments, str):
        arguments = "{}"
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{index}"
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _apply_ollama_options(chat: dict[str, Any], options: Any) -> None:
    if not isinstance(options, dict):
        return
    if "temperature" in options:
        chat["temperature"] = options["temperature"]
    if "top_p" in options:
        chat["top_p"] = options["top_p"]
    if "num_predict" in options:
        chat["max_tokens"] = options["num_predict"]
    if "stop" in options:
        chat["stop"] = options["stop"]
    if "seed" in options:
        chat["seed"] = options["seed"]
    # num_ctx has no OpenAI chat equivalent on mlx-lm; ignored intentionally.


def _apply_ollama_format(chat: dict[str, Any], fmt: Any) -> None:
    if fmt == "json":
        chat["response_format"] = {"type": "json_object"}
        return
    if isinstance(fmt, dict):
        chat["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "ollama_schema", "schema": fmt},
        }


def _copy_tools(chat: dict[str, Any], body: dict[str, Any]) -> None:
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        chat["tools"] = tools
    if "tool_choice" in body:
        chat["tool_choice"] = body["tool_choice"]


def _extract_chat_fields(
    chat_payload: dict[str, Any],
) -> tuple[str, str | None, dict[str, Any]]:
    content = ""
    finish_reason: str | None = None
    choices = chat_payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            content = message["content"]
        finish = choices[0].get("finish_reason")
        if isinstance(finish, str):
            finish_reason = finish
    usage = chat_payload.get("usage") if isinstance(chat_payload.get("usage"), dict) else {}
    return content, finish_reason, usage


def _extract_tool_calls(chat_payload: dict[str, Any]) -> list[Any] | None:
    choices = chat_payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    converted = tool_calls_as_ollama([c for c in tool_calls if isinstance(c, dict)])
    return converted or None


def _models_from_list_payload(models_payload: dict[str, Any]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    data = models_payload.get("data")
    if not isinstance(data, list):
        return models
    for entry in data:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str):
            continue
        models.append(
            {
                "name": model_id,
                "model": model_id,
                "modified_at": "",
                "size": 0,
                "digest": "",
                "details": {},
            }
        )
    return models
