"""Proxy OpenAI Responses API requests via chat completions upstream."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from orchestrator.gateway.chat_extensions import prepare_chat_upstream_body
from orchestrator.gateway.protocol.chat_sse import ToolCallAccumulator, parse_chat_sse_line
from orchestrator.gateway.protocol.responses_translate import (
    build_response_stream_events,
    chat_completion_to_response,
    next_sequence,
    responses_request_to_chat,
)
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH
from orchestrator.gateway.services.chat_proxy import CHAT_LAUNCH_MODES
from orchestrator.gateway.services.http_proxy import (
    forward_request_headers,
    proxy_json_post,
    proxy_sse_post,
    resolve_target_from_body,
    upstream_url_for_path,
    validate_target_launch_mode,
)


async def proxy_responses_create(body: dict[str, Any], headers: Any) -> Response:
    """Translate Responses → chat completions, then map the result back."""
    chat_body = responses_request_to_chat(body)
    target = await resolve_target_from_body(chat_body)
    validate_target_launch_mode(target, CHAT_LAUNCH_MODES, "responses")
    request_headers = forward_request_headers(headers)
    url = upstream_url_for_path(target, CHAT_COMPLETIONS_PATH)
    upstream_body = prepare_chat_upstream_body(
        chat_body,
        target.upstream_model,
        force_include_usage=True,
    )

    if chat_body.get("stream"):
        upstream = await proxy_sse_post(target, url, upstream_body, request_headers)
        if upstream.status_code >= 400:
            return upstream
        return StreamingResponse(
            _translate_chat_sse_to_responses(
                upstream.body_iterator,
                client_model_alias=target.alias,
            ),
            media_type="text/event-stream",
            status_code=200,
        )

    upstream = await proxy_json_post(target, url, upstream_body, request_headers)
    if upstream.status_code >= 400:
        return upstream
    chat_payload = json.loads(upstream.body)
    response_payload = chat_completion_to_response(
        chat_payload,
        client_model_alias=target.alias,
    )
    return JSONResponse(status_code=200, content=response_payload)


async def _translate_chat_sse_to_responses(
    chunks: AsyncIterator[bytes],
    *,
    client_model_alias: str,
) -> AsyncIterator[bytes]:
    meta = build_response_stream_events(
        client_model_alias=client_model_alias,
        response_id="",
    )
    meta["response_id"] = f"resp_{meta['message_id'][4:]}"
    meta["tools"] = ToolCallAccumulator()
    meta["tool_item_ids"] = {}
    meta["tool_output_indexes"] = {}
    meta["tool_call_ids"] = {}
    meta["next_output_index"] = 0
    meta["message_opened"] = False
    meta["reasoning_opened"] = False
    meta["reasoning"] = ""
    meta["reasoning_id"] = f"rs_{meta['message_id'][4:]}"
    yield _sse_event(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": next_sequence(meta),
            "response": {
                "id": meta["response_id"],
                "object": "response",
                "created_at": meta["created_at"],
                "status": "in_progress",
                "model": client_model_alias,
                "output": [],
            },
        },
    )

    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        events, buffer, failed = _drain_sse_buffer(buffer, meta)
        for event in events:
            yield event
        if failed:
            yield _failed_event(meta, client_model_alias)
            return

    if buffer and not meta.get("failed"):
        for event in _map_chat_sse_line(buffer, meta):
            yield event
        if meta.get("failed"):
            yield _failed_event(meta, client_model_alias)
            return

    for event in _stream_close_events(meta, client_model_alias):
        yield event


def _drain_sse_buffer(
    buffer: bytes,
    meta: dict[str, Any],
) -> tuple[list[bytes], bytes, bool]:
    events: list[bytes] = []
    while b"\n" in buffer:
        line, buffer = buffer.split(b"\n", 1)
        events.extend(_map_chat_sse_line(line, meta))
        if meta.get("failed"):
            return events, buffer, True
    return events, buffer, False


def _ensure_message_opened(meta: dict[str, Any]) -> list[bytes]:
    if meta.get("message_opened"):
        return []
    meta["message_opened"] = True
    meta["output_index"] = int(meta["next_output_index"])
    meta["next_output_index"] = int(meta["output_index"]) + 1
    return [
        _sse_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": next_sequence(meta),
                "output_index": meta["output_index"],
                "item": {
                    "id": meta["message_id"],
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            },
        ),
        _sse_event(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "sequence_number": next_sequence(meta),
                "item_id": meta["message_id"],
                "output_index": meta["output_index"],
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            },
        ),
    ]


def _ensure_reasoning_opened(meta: dict[str, Any]) -> list[bytes]:
    if meta.get("reasoning_opened"):
        return []
    meta["reasoning_opened"] = True
    meta["reasoning_output_index"] = int(meta["next_output_index"])
    meta["next_output_index"] = int(meta["reasoning_output_index"]) + 1
    return [
        _sse_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": next_sequence(meta),
                "output_index": meta["reasoning_output_index"],
                "item": {
                    "id": meta["reasoning_id"],
                    "type": "reasoning",
                    "status": "in_progress",
                    "summary": [],
                },
            },
        )
    ]


def _stream_close_events(meta: dict[str, Any], client_model_alias: str) -> list[bytes]:
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    tools: ToolCallAccumulator = meta["tools"]
    events: list[bytes] = []
    events.extend(_close_reasoning_events(meta))
    if meta.get("message_opened"):
        events.extend(
            [
                _sse_event(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "sequence_number": next_sequence(meta),
                        "item_id": meta["message_id"],
                        "output_index": meta["output_index"],
                        "content_index": 0,
                        "text": meta.get("text", ""),
                    },
                ),
                _sse_event(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "sequence_number": next_sequence(meta),
                        "item_id": meta["message_id"],
                        "output_index": meta["output_index"],
                        "content_index": 0,
                        "part": {"type": "output_text", "text": meta.get("text", "")},
                    },
                ),
                _sse_event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "sequence_number": next_sequence(meta),
                        "output_index": meta["output_index"],
                        "item": {
                            "id": meta["message_id"],
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {"type": "output_text", "text": meta.get("text", "")}
                            ],
                        },
                    },
                ),
            ]
        )
    events.extend(_close_tool_call_events(meta))
    output_items: list[dict[str, Any]] = []
    if meta.get("reasoning_opened") and meta.get("reasoning"):
        output_items.append(
            {
                "id": meta["reasoning_id"],
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": meta.get("reasoning", "")}],
            }
        )
    if meta.get("message_opened"):
        output_items.append(
            {
                "id": meta["message_id"],
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": meta.get("text", "")}],
            }
        )
    output_items.extend(_function_call_output_items(meta, tools))
    events.append(
        _sse_event(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": next_sequence(meta),
                "response": {
                    "id": meta["response_id"],
                    "object": "response",
                    "created_at": meta["created_at"],
                    "status": "completed",
                    "model": client_model_alias,
                    "output": output_items,
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                },
            },
        )
    )
    return events


def _close_reasoning_events(meta: dict[str, Any]) -> list[bytes]:
    if not meta.get("reasoning_opened"):
        return []
    return [
        _sse_event(
            "response.reasoning_summary_text.done",
            {
                "type": "response.reasoning_summary_text.done",
                "sequence_number": next_sequence(meta),
                "item_id": meta["reasoning_id"],
                "output_index": meta["reasoning_output_index"],
                "text": meta.get("reasoning", ""),
            },
        ),
        _sse_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "sequence_number": next_sequence(meta),
                "output_index": meta["reasoning_output_index"],
                "item": {
                    "id": meta["reasoning_id"],
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [
                        {"type": "summary_text", "text": meta.get("reasoning", "")}
                    ],
                },
            },
        ),
    ]


def _close_tool_call_events(meta: dict[str, Any]) -> list[bytes]:
    tools: ToolCallAccumulator = meta["tools"]
    events: list[bytes] = []
    for call in tools.completed_calls():
        openai_index = int(call.get("index", 0))
        item_id = meta["tool_item_ids"].get(openai_index) or call["id"]
        output_index = meta["tool_output_indexes"].get(
            openai_index,
            meta["next_output_index"],
        )
        arguments = call["function"]["arguments"]
        events.append(
            _sse_event(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "sequence_number": next_sequence(meta),
                    "item_id": item_id,
                    "output_index": output_index,
                    "arguments": arguments,
                },
            )
        )
        events.append(
            _sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": next_sequence(meta),
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": arguments,
                        "status": "completed",
                    },
                },
            )
        )
    return events


def _function_call_output_items(
    meta: dict[str, Any],
    tools: ToolCallAccumulator,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for call in tools.completed_calls():
        openai_index = int(call.get("index", 0))
        item_id = meta["tool_item_ids"].get(openai_index) or call["id"]
        items.append(
            {
                "id": item_id,
                "type": "function_call",
                "call_id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
                "status": "completed",
            }
        )
    return items


def _failed_event(meta: dict[str, Any], client_model_alias: str) -> bytes:
    return _sse_event(
        "response.failed",
        {
            "type": "response.failed",
            "sequence_number": next_sequence(meta),
            "response": {
                "id": meta["response_id"],
                "object": "response",
                "created_at": meta["created_at"],
                "status": "failed",
                "model": client_model_alias,
                "output": [],
                "error": {
                    "message": meta.get("error_message") or "Upstream stream error",
                    "type": "server_error",
                },
            },
        },
    )


def _map_chat_sse_line(line: bytes, meta: dict[str, Any]) -> list[bytes]:
    parsed = parse_chat_sse_line(line)
    if parsed is None or parsed.done:
        return []
    if parsed.error is not None:
        meta["failed"] = True
        meta["error_message"] = str(parsed.error.get("message") or "Upstream stream error")
        return []
    if parsed.usage is not None:
        meta["usage"] = parsed.usage

    events: list[bytes] = []
    if parsed.reasoning:
        events.extend(_ensure_reasoning_opened(meta))
        meta["reasoning"] = str(meta.get("reasoning", "")) + parsed.reasoning
        events.append(
            _sse_event(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "sequence_number": next_sequence(meta),
                    "item_id": meta["reasoning_id"],
                    "output_index": meta["reasoning_output_index"],
                    "delta": parsed.reasoning,
                },
            )
        )
    if parsed.text:
        events.extend(_ensure_message_opened(meta))
        meta["text"] = str(meta.get("text", "")) + parsed.text
        events.append(
            _sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": next_sequence(meta),
                    "item_id": meta["message_id"],
                    "output_index": meta["output_index"],
                    "content_index": 0,
                    "delta": parsed.text,
                },
            )
        )
    if parsed.tool_call_deltas:
        events.extend(_map_tool_call_deltas(parsed.tool_call_deltas, meta))
    return events


def _map_tool_call_deltas(
    deltas: list[dict[str, Any]],
    meta: dict[str, Any],
) -> list[bytes]:
    tools: ToolCallAccumulator = meta["tools"]
    events: list[bytes] = []
    for call, fragment in tools.ingest(deltas):
        item_id = meta["tool_item_ids"].get(call.index)
        if item_id is None:
            if not call.name:
                continue
            item_id = f"fc_{call.id or call.index}"
            output_index = int(meta["next_output_index"])
            meta["next_output_index"] = output_index + 1
            meta["tool_item_ids"][call.index] = item_id
            meta["tool_output_indexes"][call.index] = output_index
            meta["tool_call_ids"][call.index] = call.id or f"call_{call.index}"
            events.append(
                _sse_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "sequence_number": next_sequence(meta),
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "function_call",
                            "call_id": call.id or f"call_{call.index}",
                            "name": call.name,
                            "arguments": "",
                            "status": "in_progress",
                        },
                    },
                )
            )
        if fragment:
            output_index = meta["tool_output_indexes"][call.index]
            events.append(
                _sse_event(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": next_sequence(meta),
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": fragment,
                    },
                )
            )
    return events


def _sse_event(event_name: str, payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"))
    return f"event: {event_name}\ndata: {body}\n\n".encode()
