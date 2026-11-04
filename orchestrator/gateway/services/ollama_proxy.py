"""Proxy Ollama-compatible endpoints through Nadir gateway routing."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable

from asgiref.sync import sync_to_async
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.responses import Response

from orchestrator.gateway.chat_extensions import prepare_chat_upstream_body
from orchestrator.gateway.protocol.ollama_translate import (
    chat_sse_delta_text,
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
from orchestrator.gateway.selectors import list_running_gateway_models
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
    payload = await sync_to_async(list_running_gateway_models, thread_sensitive=False)()
    return JSONResponse(content=models_list_to_ollama_ps(payload))


def proxy_ollama_show(body: dict[str, Any]) -> Response:
    name = body.get("name") or body.get("model")
    if not isinstance(name, str) or not name.strip():
        raise GatewayRouteError(
            status_code=400,
            code="invalid_model",
            message="name or model is required.",
            param="name",
        )
    return JSONResponse(content=ollama_show_stub(name.strip()))


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
    target = await resolve_target_from_body(chat_body)
    validate_target_launch_mode(target, CHAT_LAUNCH_MODES, route_label)
    request_headers = forward_request_headers(headers)
    url = upstream_url_for_path(target, CHAT_COMPLETIONS_PATH)
    upstream_body = prepare_chat_upstream_body(chat_body, target.upstream_model)

    if chat_body.get("stream"):
        upstream = await proxy_sse_post(target, url, upstream_body, request_headers)
        if upstream.status_code >= 400:
            return upstream
        return StreamingResponse(
            _translate_chat_sse(
                upstream.body_iterator,
                client_model_alias=target.alias,
                stream_chunk=stream_chunk,
            ),
            media_type="application/x-ndjson",
            status_code=200,
        )

    upstream = await proxy_json_post(target, url, upstream_body, request_headers)
    if upstream.status_code >= 400:
        return upstream
    return JSONResponse(
        content=translate_response(
            json.loads(upstream.body),
            client_model_alias=target.alias,
        )
    )


async def _translate_chat_sse(
    chunks: AsyncIterator[bytes],
    *,
    client_model_alias: str,
    stream_chunk: StreamChunkFn,
) -> AsyncIterator[bytes]:
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            text = chat_sse_delta_text(line)
            if text is not None:
                yield stream_chunk(
                    client_model_alias=client_model_alias,
                    content=text,
                    done=False,
                )
    if buffer:
        text = chat_sse_delta_text(buffer)
        if text is not None:
            yield stream_chunk(
                client_model_alias=client_model_alias,
                content=text,
                done=False,
            )
    yield stream_chunk(
        client_model_alias=client_model_alias,
        content="",
        done=True,
        done_reason="stop",
    )
