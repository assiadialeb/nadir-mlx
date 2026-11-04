"""HTTP proxy helpers for chat and completion endpoints."""

from __future__ import annotations

from typing import Any

from starlette.responses import Response

from orchestrator.gateway.chat_extensions import prepare_chat_upstream_body
from orchestrator.gateway.request_timing import reset_request_timing, start_request_timing
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, COMPLETIONS_PATH
from orchestrator.gateway.services.http_proxy import (
    forward_request_headers,
    proxy_json_post,
    proxy_sse_post,
    resolve_target_from_body,
    upstream_url_for_path,
    validate_target_launch_mode,
)
from orchestrator.gateway.thinking_control import kwargs_from_openai_chat

CHAT_LAUNCH_MODES = frozenset({"TEXT", "MULTIMODAL"})
TEXT_ONLY_LAUNCH_MODES = frozenset({"TEXT"})


def _force_usage_for_telemetry() -> bool:
    from django.conf import settings

    from orchestrator.env_utils import env_bool
    from orchestrator.gateway.request_timing import telemetry_enabled

    if not telemetry_enabled():
        return False
    return env_bool(
        "NADIR_GATEWAY_TELEMETRY_FORCE_USAGE",
        bool(getattr(settings, "NADIR_GATEWAY_TELEMETRY_FORCE_USAGE", True)),
    )


def _client_requested_include_usage(body: dict[str, Any]) -> bool:
    options = body.get("stream_options")
    return isinstance(options, dict) and bool(options.get("include_usage"))


def _configure_openai_stream_telemetry(collector: Any, body: dict[str, Any]) -> bool:
    """Force upstream usage for telemetry; strip usage-only chunks unless client opted in."""
    force = _force_usage_for_telemetry()
    if collector is not None and force and not _client_requested_include_usage(body):
        collector.strip_usage_only = True
    return force


async def proxy_chat_completions(body: dict[str, Any], headers: Any) -> Response:
    """Proxy a chat completion request to a TEXT or MULTIMODAL instance."""
    stream = bool(body.get("stream"))
    collector = start_request_timing(endpoint=CHAT_COMPLETIONS_PATH, stream=stream)
    try:
        target = await resolve_target_from_body(body)
        validate_target_launch_mode(target, CHAT_LAUNCH_MODES, "chat completions")
        force_usage = _configure_openai_stream_telemetry(collector, body)
        upstream_body = prepare_chat_upstream_body(
            body,
            target.upstream_model,
            force_include_usage=force_usage,
            chat_template_kwargs=kwargs_from_openai_chat(body),
        )
        request_headers = forward_request_headers(headers)
        url = upstream_url_for_path(target, CHAT_COMPLETIONS_PATH)

        if stream:
            return await proxy_sse_post(target, url, upstream_body, request_headers)
        return await proxy_json_post(target, url, upstream_body, request_headers)
    finally:
        # Non-stream: finalize now. Stream: finalize in proxy_sse_post iter_chunks.
        if not stream:
            reset_request_timing(collector)


async def proxy_text_completions(body: dict[str, Any], headers: Any) -> Response:
    """Proxy a legacy text completion request to a TEXT instance."""
    stream = bool(body.get("stream"))
    collector = start_request_timing(endpoint=COMPLETIONS_PATH, stream=stream)
    try:
        target = await resolve_target_from_body(body)
        validate_target_launch_mode(target, TEXT_ONLY_LAUNCH_MODES, "text completions")
        force_usage = _configure_openai_stream_telemetry(collector, body)
        upstream_body = prepare_chat_upstream_body(
            body,
            target.upstream_model,
            force_include_usage=force_usage,
            chat_template_kwargs=kwargs_from_openai_chat(body),
        )
        request_headers = forward_request_headers(headers)
        url = upstream_url_for_path(target, COMPLETIONS_PATH)

        if stream:
            return await proxy_sse_post(target, url, upstream_body, request_headers)
        return await proxy_json_post(target, url, upstream_body, request_headers)
    finally:
        if not stream:
            reset_request_timing(collector)
