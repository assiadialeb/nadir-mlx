"""Proxy Ollama-compatible endpoints through Nadir gateway routing."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable

from asgiref.sync import sync_to_async
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.responses import Response

from orchestrator.gateway.chat_extensions import prepare_chat_upstream_body
from orchestrator.gateway.protocol.chat_sse import (
    ToolCallAccumulator,
    parse_chat_sse_line,
    tool_calls_as_ollama,
)
from orchestrator.gateway.protocol.ollama_translate import (
    models_list_to_ollama_ps,
    models_list_to_ollama_tags,
    ollama_chat_stream_chunk,
    ollama_chat_to_openai,
    ollama_embed_to_openai,
    ollama_generate_stream_chunk,
    ollama_generate_to_openai,
    ollama_show_stub,
    ollama_version_payload,
    openai_chat_to_ollama,
    openai_chat_to_ollama_generate,
    openai_embeddings_to_ollama,
)
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, EMBEDDINGS_PATH, GatewayRouteError
from orchestrator.gateway.route_cache import get_route_snapshot
from orchestrator.gateway.selectors import get_gateway_model, list_running_gateway_models
from orchestrator.gateway.services.chat_proxy import CHAT_LAUNCH_MODES
from orchestrator.gateway.services.http_proxy import (
    forward_request_headers,
    prepare_upstream_body,
    proxy_json_post,
    proxy_sse_post,
    resolve_target_from_body,
    upstream_url_for_path,
    validate_target_launch_mode,
)

EMBEDDING_MODES = frozenset({"EMBEDDING"})
StreamChunkFn = Callable[..., bytes]


def proxy_ollama_root() -> Response:
    return PlainTextResponse("Ollama is running")


def proxy_ollama_version() -> Response:
    return JSONResponse(content=ollama_version_payload())


async def proxy_ollama_tags() -> Response:
    payload = await sync_to_async(list_running_gateway_models, thread_sensitive=False)()
    return JSONResponse(content=models_list_to_ollama_tags(payload))


async def proxy_ollama_ps() -> Response:
    snapshot = await sync_to_async(get_route_snapshot, thread_sensitive=False)()
    running_aliases = sorted({target.alias for target in snapshot.running_targets.values()})
    payload = {
        "object": "list",
        "data": [
            {"id": alias, "object": "model", "created": 0, "owned_by": "nadir"}
            for alias in running_aliases
        ],
    }
    return JSONResponse(content=models_list_to_ollama_ps(payload))


async def proxy_ollama_show(body: dict[str, Any]) -> Response:
    name = body.get("name") or body.get("model")
    if not isinstance(name, str) or not name.strip():
        raise GatewayRouteError(
            status_code=400,
            code="invalid_model",
            message="name or model is required.",
            param="name",
        )
    cleaned = name.strip()
    model = await sync_to_async(get_gateway_model, thread_sensitive=False)(cleaned)
    metadata = model.get("metadata") if isinstance(model.get("metadata"), dict) else {}
    launch_mode = str(metadata.get("launch_mode") or "TEXT")
    tool_parser_type = str(metadata.get("tool_parser_type") or "")
    return JSONResponse(
        content=ollama_show_stub(
            cleaned,
            launch_mode=launch_mode,
            tool_parser_type=tool_parser_type,
        )
    )


async def proxy_ollama_chat(body: dict[str, Any], headers: Any) -> Response:
    return await _proxy_chat_like(
        body,
        headers,
        translate_request=ollama_chat_to_openai,
        translate_response=openai_chat_to_ollama,
        stream_chunk=ollama_chat_stream_chunk,
        route_label="ollama chat",
    )


async def proxy_ollama_generate(body: dict[str, Any], headers: Any) -> Response:
    return await _proxy_chat_like(
        body,
        headers,
        translate_request=ollama_generate_to_openai,
        translate_response=openai_chat_to_ollama_generate,
        stream_chunk=ollama_generate_stream_chunk,
        route_label="ollama generate",
    )


async def proxy_ollama_embed(
    body: dict[str, Any],
    headers: Any,
    *,
    legacy: bool = False,
) -> Response:
    embed_body = ollama_embed_to_openai(body)
    target = await resolve_target_from_body(embed_body)
    validate_target_launch_mode(target, EMBEDDING_MODES, "ollama embed")
    upstream = await proxy_json_post(
        target,
        upstream_url_for_path(target, EMBEDDINGS_PATH),
        prepare_upstream_body(embed_body, target),
        forward_request_headers(headers),
    )
    if upstream.status_code >= 400:
        return upstream
    return JSONResponse(
        content=openai_embeddings_to_ollama(
            json.loads(upstream.body),
            client_model_alias=target.alias,
            legacy=legacy,
        )
    )


def _ollama_thinking_enabled(body: dict[str, Any]) -> bool:
    return body.get("think") is True


async def _proxy_chat_like(
    body: dict[str, Any],
    headers: Any,
    *,
    translate_request: Callable[[dict[str, Any]], dict[str, Any]],
    translate_response: Callable[..., dict[str, Any]],
    stream_chunk: StreamChunkFn,
    route_label: str,
) -> Response:
    chat_body = translate_request(body)
    thinking_enabled = _ollama_thinking_enabled(body)
    target = await resolve_target_from_body(chat_body)
    validate_target_launch_mode(target, CHAT_LAUNCH_MODES, route_label)
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
            _translate_chat_sse(
                upstream.body_iterator,
                client_model_alias=target.alias,
                stream_chunk=stream_chunk,
                thinking_enabled=thinking_enabled,
            ),
            media_type="application/x-ndjson",
            status_code=200,
        )

    upstream = await proxy_json_post(target, url, upstream_body, request_headers)
    if upstream.status_code >= 400:
        return upstream
    kwargs: dict[str, Any] = {"client_model_alias": target.alias}
    if translate_response is openai_chat_to_ollama:
        kwargs["include_thinking"] = thinking_enabled
    return JSONResponse(
        content=translate_response(json.loads(upstream.body), **kwargs)
    )


async def _translate_chat_sse(
    chunks: AsyncIterator[bytes],
    *,
    client_model_alias: str,
    stream_chunk: StreamChunkFn,
    thinking_enabled: bool = False,
) -> AsyncIterator[bytes]:
    state = _OllamaStreamState(thinking_enabled=thinking_enabled)
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            for event in _events_for_ollama_line(line, client_model_alias, stream_chunk, state):
                yield event
            if state.failed:
                return
    if buffer and not state.failed:
        for event in _events_for_ollama_line(buffer, client_model_alias, stream_chunk, state):
            yield event
        if state.failed:
            return
    yield _final_ollama_chunk(client_model_alias, stream_chunk, state)


class _OllamaStreamState:
    def __init__(self, *, thinking_enabled: bool = False) -> None:
        self.tools = ToolCallAccumulator()
        self.done_reason = "stop"
        self.usage: dict[str, Any] = {}
        self.failed = False
        self.error_message = ""
        self.thinking_enabled = thinking_enabled


def _events_for_ollama_line(
    line: bytes,
    client_model_alias: str,
    stream_chunk: StreamChunkFn,
    state: _OllamaStreamState,
) -> list[bytes]:
    parsed = parse_chat_sse_line(line)
    if parsed is None:
        return []
    if parsed.error is not None:
        state.failed = True
        state.error_message = str(parsed.error.get("message") or "Upstream stream error")
        return [
            stream_chunk(
                client_model_alias=client_model_alias,
                content="",
                done=True,
                done_reason="error",
                error=state.error_message,
            )
        ]
    if parsed.usage is not None:
        state.usage = parsed.usage
    if parsed.finish_reason:
        mapped = "stop" if parsed.finish_reason == "tool_calls" else parsed.finish_reason
        state.done_reason = mapped
    if parsed.tool_call_deltas:
        state.tools.ingest(parsed.tool_call_deltas)
    events: list[bytes] = []
    if (
        parsed.reasoning
        and state.thinking_enabled
        and stream_chunk is ollama_chat_stream_chunk
    ):
        events.append(
            stream_chunk(
                client_model_alias=client_model_alias,
                content="",
                thinking=parsed.reasoning,
                done=False,
            )
        )
    if parsed.text:
        events.append(
            stream_chunk(
                client_model_alias=client_model_alias,
                content=parsed.text,
                done=False,
            )
        )
    return events


def _final_ollama_chunk(
    client_model_alias: str,
    stream_chunk: StreamChunkFn,
    state: _OllamaStreamState,
) -> bytes:
    tool_calls = tool_calls_as_ollama(state.tools.completed_calls())
    kwargs: dict[str, Any] = {
        "client_model_alias": client_model_alias,
        "content": "",
        "done": True,
        "done_reason": state.done_reason or "stop",
        "prompt_eval_count": int(state.usage.get("prompt_tokens") or 0),
        "eval_count": int(state.usage.get("completion_tokens") or 0),
    }
    if tool_calls and stream_chunk is ollama_chat_stream_chunk:
        kwargs["tool_calls"] = tool_calls
    return stream_chunk(**kwargs)
