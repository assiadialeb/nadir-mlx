"""Proxy Anthropic Messages API via chat completions (TEXT/MULTIMODAL)."""

from __future__ import annotations

import json
from typing import Any

from fastapi.responses import JSONResponse
from starlette.responses import Response

from orchestrator.gateway.chat_extensions import prepare_chat_upstream_body
from orchestrator.gateway.protocol.anthropic_translate import (
    anthropic_request_to_chat,
    chat_completion_to_anthropic,
)
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, GatewayRouteError
from orchestrator.gateway.services.chat_proxy import CHAT_LAUNCH_MODES
from orchestrator.gateway.services.http_proxy import (
    forward_request_headers,
    proxy_json_post,
    resolve_target_from_body,
    upstream_url_for_path,
    validate_target_launch_mode,
)


async def proxy_anthropic_messages(body: dict[str, Any], headers: Any) -> Response:
    """Handle Anthropic ``/v1/messages`` (non-stream MVP; stream returns 501)."""
    if body.get("stream"):
        raise GatewayRouteError(
            status_code=501,
            code="not_implemented",
            message="Streaming Anthropic messages is not implemented in this Nadir MVP.",
            param="stream",
        )

    chat_body = anthropic_request_to_chat(body)
    target = await resolve_target_from_body(chat_body)
    validate_target_launch_mode(target, CHAT_LAUNCH_MODES, "anthropic messages")
    upstream_body = prepare_chat_upstream_body(chat_body, target.upstream_model)
    upstream = await proxy_json_post(
        target,
        upstream_url_for_path(target, CHAT_COMPLETIONS_PATH),
        upstream_body,
        forward_request_headers(headers),
    )
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
