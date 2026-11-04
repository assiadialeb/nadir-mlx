"""Proxy Anthropic Messages API via chat completions (TEXT/MULTIMODAL)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from orchestrator.gateway.chat_extensions import prepare_chat_upstream_body
from orchestrator.gateway.protocol.anthropic_translate import (
    anthropic_request_to_chat,
    anthropic_stream_content_block_start,
    anthropic_stream_content_block_stop,
    anthropic_stream_message_delta,
    anthropic_stream_message_start,
    anthropic_stream_message_stop,
    anthropic_stream_text_delta,
    chat_completion_to_anthropic,
    chat_sse_delta_text,
    chat_sse_finish_reason,
    new_anthropic_message_id,
    openai_finish_to_anthropic_stop,
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


async def proxy_anthropic_messages(body: dict[str, Any], headers: Any) -> Response:
    """Handle Anthropic ``/v1/messages`` (JSON or SSE when ``stream`` is true)."""
    chat_body = anthropic_request_to_chat(body)
    target = await resolve_target_from_body(chat_body)
    validate_target_launch_mode(target, CHAT_LAUNCH_MODES, "anthropic messages")
    request_headers = forward_request_headers(headers)
    url = upstream_url_for_path(target, CHAT_COMPLETIONS_PATH)
    upstream_body = prepare_chat_upstream_body(chat_body, target.upstream_model)

    if chat_body.get("stream"):
        upstream = await proxy_sse_post(target, url, upstream_body, request_headers)
        if upstream.status_code >= 400:
            return upstream
        return StreamingResponse(
            _translate_chat_sse_to_anthropic(
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
    return JSONResponse(
        status_code=200,
        content=chat_completion_to_anthropic(
            chat_payload,
            client_model_alias=target.alias,
        ),
    )


async def _translate_chat_sse_to_anthropic(
    chunks: AsyncIterator[bytes],
    *,
    client_model_alias: str,
) -> AsyncIterator[bytes]:
    message_id = new_anthropic_message_id()
    stop_reason = "end_turn"
    yield anthropic_stream_message_start(
        message_id=message_id,
        client_model_alias=client_model_alias,
    )
    yield anthropic_stream_content_block_start()

    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            stop_reason = _emit_chat_line(line, stop_reason)
            event = _delta_event_for_line(line)
            if event is not None:
                yield event
    if buffer:
        stop_reason = _emit_chat_line(buffer, stop_reason)
        event = _delta_event_for_line(buffer)
        if event is not None:
            yield event

    yield anthropic_stream_content_block_stop()
    yield anthropic_stream_message_delta(stop_reason)
    yield anthropic_stream_message_stop()


def _delta_event_for_line(line: bytes) -> bytes | None:
    text = chat_sse_delta_text(line)
    if text is None:
        return None
    return anthropic_stream_text_delta(text)


def _emit_chat_line(line: bytes, current_stop: str) -> str:
    finish = chat_sse_finish_reason(line)
    if finish is None:
        return current_stop
    return openai_finish_to_anthropic_stop(finish)
