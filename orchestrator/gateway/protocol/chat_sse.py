"""Shared OpenAI chat-completions SSE parsing helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatSseEvent:
    """Parsed fields from one ``data:`` SSE line."""

    text: str | None = None
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
        """Return OpenAI chat ``tool_calls`` payloads in index order."""
        return [
            {
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
    if not line.startswith(b"data:"):
        return None
    raw = line[5:].strip()
    if not raw:
        return None
    if raw == b"[DONE]":
        return ChatSseEvent(done=True)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    event = ChatSseEvent()
    error = payload.get("error")
    if isinstance(error, dict):
        event.error = error
        return event
    usage = payload.get("usage")
    if isinstance(usage, dict):
        event.usage = usage

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return event if event.usage is not None else None

    first = choices[0]
    finish = first.get("finish_reason")
    if isinstance(finish, str) and finish:
        event.finish_reason = finish

    delta = first.get("delta")
    if isinstance(delta, dict):
        text = delta.get("content")
        if isinstance(text, str) and text:
            event.text = text
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            event.tool_call_deltas = [item for item in tool_calls if isinstance(item, dict)]

    if (
        event.text is None
        and not event.tool_call_deltas
        and event.finish_reason is None
        and event.usage is None
    ):
        return None
    return event


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
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                parsed = {}
            arguments = parsed if isinstance(parsed, dict) else {}
        elif not isinstance(arguments, dict):
            arguments = {}
        converted.append({"function": {"name": name, "arguments": arguments}})
    return converted


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
