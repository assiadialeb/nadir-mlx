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
    meta["text"] = ""
    yield _sse_event(
        "response.created",
        {
            "type": "response.created",
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
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            event = _map_chat_sse_line(line, meta)
            if event is not None:
                yield event
    if buffer:
        event = _map_chat_sse_line(buffer, meta)
        if event is not None:
            yield event
    yield _sse_event(
        "response.completed",
        {
            "type": "response.completed",
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
                        "content": [{"type": "output_text", "text": meta["text"]}],
                    }
                ],
            },
        },
    )


def _map_chat_sse_line(line: bytes, meta: dict[str, Any]) -> bytes | None:
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
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    if not isinstance(delta, dict):
        return None
    text = delta.get("content")
    if not isinstance(text, str) or text == "":
        return None
    meta["text"] = str(meta.get("text", "")) + text
    return _sse_event(
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "item_id": meta["message_id"],
            "output_index": meta["output_index"],
            "delta": text,
        },
    )


def _sse_event(event_name: str, payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"))
    return f"event: {event_name}\ndata: {body}\n\n".encode()
