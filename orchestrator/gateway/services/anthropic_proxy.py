"""Proxy Anthropic Messages API via chat completions (TEXT/MULTIMODAL)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from orchestrator.gateway.chat_extensions import prepare_chat_upstream_body
from orchestrator.gateway.protocol.anthropic_translate import (
    anthropic_error_envelope,
    anthropic_request_to_chat,
    anthropic_stream_content_block_stop,
    anthropic_stream_error,
    anthropic_stream_input_json_delta,
    anthropic_stream_message_delta,
    anthropic_stream_message_start,
    anthropic_stream_message_stop,
    anthropic_stream_text_delta,
    anthropic_stream_thinking_block_start,
    anthropic_stream_thinking_delta,
    anthropic_stream_tool_use_block_start,
    anthropic_stream_content_block_start,
    chat_completion_to_anthropic,
    new_anthropic_message_id,
    openai_finish_to_anthropic_stop,
)
from orchestrator.gateway.protocol.chat_sse import ToolCallAccumulator, parse_chat_sse_line
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
    upstream_body = prepare_chat_upstream_body(
        chat_body,
        target.upstream_model,
        force_include_usage=True,
    )

    if chat_body.get("stream"):
        upstream = await proxy_sse_post(target, url, upstream_body, request_headers)
        if upstream.status_code >= 400:
            return _as_anthropic_error(upstream)
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
        return _as_anthropic_error(upstream)
    chat_payload = json.loads(upstream.body)
    return JSONResponse(
        status_code=200,
        content=chat_completion_to_anthropic(
            chat_payload,
            client_model_alias=target.alias,
        ),
    )


def _as_anthropic_error(upstream: Response) -> JSONResponse:
    """Rewrite OpenAI/gateway error envelopes into Anthropic Messages shape."""
    message = "Upstream request failed"
    error_type = "api_error"
    payload: Any = None
    body = getattr(upstream, "body", None)
    if isinstance(body, (bytes, bytearray)) and body:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            message = body.decode("utf-8", errors="replace") or message
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or message)
            raw_type = error.get("type")
            if isinstance(raw_type, str) and raw_type:
                error_type = raw_type if raw_type.endswith("_error") else "api_error"
        elif isinstance(payload.get("type"), str) and payload.get("type") == "error":
            return JSONResponse(status_code=upstream.status_code, content=payload)
    return JSONResponse(
        status_code=upstream.status_code,
        content=anthropic_error_envelope(message, error_type),
    )


async def _translate_chat_sse_to_anthropic(
    chunks: AsyncIterator[bytes],
    *,
    client_model_alias: str,
) -> AsyncIterator[bytes]:
    message_id = new_anthropic_message_id()
    state = _AnthropicStreamState()
    yield anthropic_stream_message_start(
        message_id=message_id,
        client_model_alias=client_model_alias,
    )

    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            for event in _events_for_chat_line(line, state):
                yield event
            if state.failed:
                return
    if buffer and not state.failed:
        for event in _events_for_chat_line(buffer, state):
            yield event
        if state.failed:
            return

    for event in _close_anthropic_stream(state):
        yield event


class _AnthropicStreamState:
    def __init__(self) -> None:
        self.stop_reason = "end_turn"
        self.block_index = 0
        self.open_block_index: int | None = None
        self.tools = ToolCallAccumulator()
        self.tool_block_indexes: dict[int, int] = {}
        self.thinking_index: int | None = None
        self.text_index: int | None = None
        self.usage: dict[str, Any] = {}
        self.failed = False


def _events_for_chat_line(line: bytes, state: _AnthropicStreamState) -> list[bytes]:
    parsed = parse_chat_sse_line(line)
    if parsed is None:
        return []
    if parsed.error is not None:
        state.failed = True
        message = str(parsed.error.get("message") or "Upstream stream error")
        return [anthropic_stream_error(message)]
    if parsed.usage is not None:
        state.usage = parsed.usage
    if parsed.finish_reason:
        state.stop_reason = openai_finish_to_anthropic_stop(parsed.finish_reason)

    events: list[bytes] = []
    if parsed.reasoning:
        events.extend(_emit_thinking_delta(parsed.reasoning, state))
    if parsed.text:
        events.extend(_emit_text_delta(parsed.text, state))
    if parsed.tool_call_deltas:
        events.extend(_emit_tool_deltas(parsed.tool_call_deltas, state))
    return events


def _close_open_block(state: _AnthropicStreamState) -> list[bytes]:
    if state.open_block_index is None:
        return []
    index = state.open_block_index
    state.open_block_index = None
    return [anthropic_stream_content_block_stop(index=index)]


def _open_block(state: _AnthropicStreamState, start_event: bytes) -> list[bytes]:
    events = _close_open_block(state)
    events.append(start_event)
    state.open_block_index = state.block_index
    state.block_index += 1
    return events


def _emit_thinking_delta(text: str, state: _AnthropicStreamState) -> list[bytes]:
    events: list[bytes] = []
    if state.thinking_index is None:
        events.extend(
            _open_block(
                state,
                anthropic_stream_thinking_block_start(index=state.block_index),
            )
        )
        state.thinking_index = state.open_block_index
    assert state.thinking_index is not None
    events.append(anthropic_stream_thinking_delta(text, index=state.thinking_index))
    return events


def _emit_text_delta(text: str, state: _AnthropicStreamState) -> list[bytes]:
    events: list[bytes] = []
    if state.text_index is None:
        events.extend(
            _open_block(
                state,
                anthropic_stream_content_block_start(index=state.block_index),
            )
        )
        state.text_index = state.open_block_index
    assert state.text_index is not None
    events.append(anthropic_stream_text_delta(text, index=state.text_index))
    return events


def _emit_tool_deltas(
    deltas: list[dict[str, Any]],
    state: _AnthropicStreamState,
) -> list[bytes]:
    events: list[bytes] = []
    for call, fragment in state.tools.ingest(deltas):
        block_index = state.tool_block_indexes.get(call.index)
        if block_index is None:
            if not call.name:
                continue
            events.extend(
                _open_block(
                    state,
                    anthropic_stream_tool_use_block_start(
                        index=state.block_index,
                        tool_id=call.id or f"toolu_{call.index}",
                        name=call.name,
                    ),
                )
            )
            block_index = state.open_block_index
            assert block_index is not None
            state.tool_block_indexes[call.index] = block_index
            call.started = True
        if fragment:
            events.append(anthropic_stream_input_json_delta(fragment, index=block_index))
    return events


def _close_anthropic_stream(state: _AnthropicStreamState) -> list[bytes]:
    events = _close_open_block(state)
    if state.tools.has_calls() and state.stop_reason == "end_turn":
        state.stop_reason = "tool_use"
    events.append(
        anthropic_stream_message_delta(
            state.stop_reason,
            output_tokens=int(state.usage.get("completion_tokens") or 0),
            input_tokens=int(state.usage.get("prompt_tokens") or 0),
        )
    )
    events.append(anthropic_stream_message_stop())
    return events
