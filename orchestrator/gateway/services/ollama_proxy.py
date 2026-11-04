"""Proxy Ollama-compatible endpoints through Nadir gateway routing."""

from __future__ import annotations

import json
from typing import Any

from asgiref.sync import sync_to_async
from fastapi.responses import JSONResponse
from starlette.responses import Response

from orchestrator.gateway.chat_extensions import prepare_chat_upstream_body
from orchestrator.gateway.protocol.ollama_translate import (
    models_list_to_ollama_tags,
    ollama_chat_to_openai,
    ollama_embed_to_openai,
    openai_chat_to_ollama,
    openai_embeddings_to_ollama,
)
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, EMBEDDINGS_PATH, GatewayRouteError
from orchestrator.gateway.selectors import list_running_gateway_models
from orchestrator.gateway.services.chat_proxy import CHAT_LAUNCH_MODES
from orchestrator.gateway.services.http_proxy import (
    forward_request_headers,
    prepare_upstream_body,
    proxy_json_post,
    resolve_target_from_body,
    upstream_url_for_path,
    validate_target_launch_mode,
)

EMBEDDING_MODES = frozenset({"EMBEDDING"})


async def proxy_ollama_tags() -> Response:
    payload = await sync_to_async(list_running_gateway_models, thread_sensitive=False)()
    return JSONResponse(content=models_list_to_ollama_tags(payload))


async def proxy_ollama_chat(body: dict[str, Any], headers: Any) -> Response:
    if body.get("stream"):
        raise GatewayRouteError(
            status_code=501,
            code="not_implemented",
            message="Streaming Ollama chat is not implemented in this Nadir MVP.",
            param="stream",
        )
    chat_body = ollama_chat_to_openai(body)
    target = await resolve_target_from_body(chat_body)
    validate_target_launch_mode(target, CHAT_LAUNCH_MODES, "ollama chat")
    upstream = await proxy_json_post(
        target,
        upstream_url_for_path(target, CHAT_COMPLETIONS_PATH),
        prepare_chat_upstream_body(chat_body, target.upstream_model),
        forward_request_headers(headers),
    )
    if upstream.status_code >= 400:
        return upstream
    return JSONResponse(
        content=openai_chat_to_ollama(
            json.loads(upstream.body),
            client_model_alias=target.alias,
        )
    )


async def proxy_ollama_embed(body: dict[str, Any], headers: Any) -> Response:
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
        )
    )
