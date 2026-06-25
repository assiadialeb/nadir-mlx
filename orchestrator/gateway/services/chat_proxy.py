"""HTTP proxy helpers for chat and completion endpoints."""

from __future__ import annotations

from typing import Any

from fastapi.responses import StreamingResponse
from starlette.responses import Response

from orchestrator.gateway.chat_extensions import prepare_chat_upstream_body
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, COMPLETIONS_PATH
from orchestrator.gateway.services.http_proxy import (
    forward_request_headers,
    prepare_upstream_body,
    proxy_json_post,
    resolve_target_from_body,
    stream_upstream_chunks,
    upstream_url_for_path,
    validate_target_launch_mode,
)

CHAT_LAUNCH_MODES = frozenset({"TEXT", "MULTIMODAL"})
TEXT_ONLY_LAUNCH_MODES = frozenset({"TEXT"})


async def proxy_chat_completions(body: dict[str, Any], headers: Any) -> Response:
    """Proxy a chat completion request to a TEXT or MULTIMODAL instance."""
    target = await resolve_target_from_body(body)
    validate_target_launch_mode(target, CHAT_LAUNCH_MODES, "chat completions")
    upstream_body = prepare_chat_upstream_body(body, target.upstream_model)
    request_headers = forward_request_headers(headers)
    url = upstream_url_for_path(target, CHAT_COMPLETIONS_PATH)

    if body.get("stream"):
        return StreamingResponse(
            stream_upstream_chunks(target, url, upstream_body, request_headers),
            media_type="text/event-stream",
        )
    return await proxy_json_post(target, url, upstream_body, request_headers)


async def proxy_text_completions(body: dict[str, Any], headers: Any) -> Response:
    """Proxy a legacy text completion request to a TEXT instance."""
    target = await resolve_target_from_body(body)
    validate_target_launch_mode(target, TEXT_ONLY_LAUNCH_MODES, "text completions")
    upstream_body = prepare_chat_upstream_body(body, target.upstream_model)
    request_headers = forward_request_headers(headers)
    url = upstream_url_for_path(target, COMPLETIONS_PATH)

    if body.get("stream"):
        return StreamingResponse(
            stream_upstream_chunks(target, url, upstream_body, request_headers),
            media_type="text/event-stream",
        )
    return await proxy_json_post(target, url, upstream_body, request_headers)
