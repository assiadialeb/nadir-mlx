"""Proxy OpenAI Responses API requests via chat completions upstream."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from orchestrator.gateway.chat_extensions import prepare_chat_upstream_body
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
    upstream_body = prepare_chat_upstream_body(chat_body, target.upstream_model)

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
    for event in _stream_open_events(meta, client_model_alias):
        yield event

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


def _stream_open_events(meta: dict[str, Any], client_model_alias: str) -> list[bytes]:
    events = [
        _sse_event(
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
        ),
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
    return events


def _stream_close_events(meta: dict[str, Any], client_model_alias: str) -> list[bytes]:
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    return [
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
                    "content": [{"type": "output_text", "text": meta.get("text", "")}],
                },
            },
        ),
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
                    "output": [
                        {
                            "id": meta["message_id"],
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": meta.get("text", "")}],
                        }
                    ],
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                },
            },
        ),
    ]


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
    if not line.startswith(b"data:"):
        return []
    raw = line[5:].strip()
    if not raw or raw == b"[DONE]":
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("error"), dict):
        meta["failed"] = True
        meta["error_message"] = str(payload["error"].get("message") or "Upstream stream error")
        return []
    if isinstance(payload.get("usage"), dict):
        meta["usage"] = payload["usage"]
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    if not isinstance(delta, dict):
        return []
    text = delta.get("content")
    if not isinstance(text, str) or text == "":
        return []
    meta["text"] = str(meta.get("text", "")) + text
    return [
        _sse_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": next_sequence(meta),
                "item_id": meta["message_id"],
                "output_index": meta["output_index"],
                "content_index": 0,
                "delta": text,
            },
        )
    ]


def _sse_event(event_name: str, payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"))
    return f"event: {event_name}\ndata: {body}\n\n".encode()
