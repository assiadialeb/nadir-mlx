"""Shared OpenAI chat-completions SSE parsing helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatSseEvent:
    """Parsed fields from one ``data:`` SSE line."""

    text: str | None = None
    reasoning: str | None = None
    tool_call_deltas: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    done: bool = False


@dataclass
class AccumulatedToolCall:
    """Mutable OpenAI-style tool call assembled from stream deltas."""

    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""
    started: bool = False


class ToolCallAccumulator:
    """Accumulate ``delta.tool_calls`` fragments keyed by OpenAI index."""

    def __init__(self) -> None:
        self._by_index: dict[int, AccumulatedToolCall] = {}

    def ingest(self, deltas: list[dict[str, Any]]) -> list[tuple[AccumulatedToolCall, str]]:
        """Merge deltas; return ``(call, arguments_fragment)`` for each update."""
        updates: list[tuple[AccumulatedToolCall, str]] = []
        for delta in deltas:
            if not isinstance(delta, dict):
                continue
            raw_index = delta.get("index")
            index = int(raw_index) if isinstance(raw_index, int) else 0
            call = self._by_index.get(index)
            if call is None:
                call = AccumulatedToolCall(index=index)
                self._by_index[index] = call
            arg_fragment = _merge_tool_call_delta(call, delta)
            updates.append((call, arg_fragment))
        return updates

    def completed_calls(self) -> list[dict[str, Any]]:
        """Return OpenAI chat ``tool_calls`` payloads in index order (includes ``index``)."""
        return [
            {
                "index": call.index,
                "id": call.id or f"call_{call.index}",
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in sorted(self._by_index.values(), key=lambda item: item.index)
            if call.name
        ]

    def has_calls(self) -> bool:
        return any(call.name for call in self._by_index.values())


def parse_chat_sse_line(line: bytes) -> ChatSseEvent | None:
    """Parse one SSE line into a :class:`ChatSseEvent`, or ``None`` if noise."""
    payload = _load_sse_payload(line)
    if payload is None:
        return None
    if payload is _DONE:
        return ChatSseEvent(done=True)

    event = ChatSseEvent()
    error = payload.get("error")
    if isinstance(error, dict):
        event.error = error
        return event
    usage = payload.get("usage")
    if isinstance(usage, dict):
        event.usage = usage

    _fill_choice_fields(event, payload.get("choices"))
    if _event_is_empty(event):
        return None
    return event


_DONE = object()


def _load_sse_payload(line: bytes) -> dict[str, Any] | object | None:
    if not line.startswith(b"data:"):
        return None
    raw = line[5:].strip()
    if not raw:
        return None
    if raw == b"[DONE]":
        return _DONE
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _fill_choice_fields(event: ChatSseEvent, choices: Any) -> None:
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return
    first = choices[0]
    finish = first.get("finish_reason")
    if isinstance(finish, str) and finish:
        event.finish_reason = finish
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return
    text = delta.get("content")
    if isinstance(text, str) and text:
        event.text = text
    event.reasoning = _delta_reasoning(delta)
    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list):
        event.tool_call_deltas = [item for item in tool_calls if isinstance(item, dict)]


def _delta_reasoning(delta: dict[str, Any]) -> str | None:
    reasoning = delta.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    reasoning = delta.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    return None


def _event_is_empty(event: ChatSseEvent) -> bool:
    return (
        event.text is None
        and event.reasoning is None
        and not event.tool_call_deltas
        and event.finish_reason is None
        and event.usage is None
        and event.error is None
    )


def chat_sse_delta_text(line: bytes) -> str | None:
    """Extract assistant text delta from one OpenAI chat SSE ``data:`` line."""
    event = parse_chat_sse_line(line)
    if event is None:
        return None
    return event.text


def chat_sse_finish_reason(line: bytes) -> str | None:
    """Return OpenAI ``finish_reason`` from a chat SSE line when present."""
    event = parse_chat_sse_line(line)
    if event is None:
        return None
    return event.finish_reason


def tool_calls_as_ollama(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool_calls into Ollama objects (arguments as JSON objects)."""
    converted: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        converted_call = _openai_tool_call_to_ollama(call)
        if converted_call is not None:
            converted.append(converted_call)
    return converted


def _openai_tool_call_to_ollama(call: dict[str, Any]) -> dict[str, Any] | None:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return None
    return {"function": {"name": name, "arguments": _coerce_ollama_arguments(function.get("arguments"))}}


def _coerce_ollama_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(arguments, dict):
        return arguments
    return {}


def _merge_tool_call_delta(call: AccumulatedToolCall, delta: dict[str, Any]) -> str:
    call_id = delta.get("id")
    if isinstance(call_id, str) and call_id:
        call.id = call_id
    function = delta.get("function") if isinstance(delta.get("function"), dict) else {}
    name = function.get("name")
    if isinstance(name, str) and name:
        call.name = name
    arguments = function.get("arguments")
    fragment = arguments if isinstance(arguments, str) else ""
    if fragment:
        call.arguments += fragment
    return fragment
