"""Proxy helpers for non-chat OpenAI-compatible MLX endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import Request, UploadFile
from starlette.responses import Response

from orchestrator.gateway.router import (
    AUDIO_SPEECH_PATH,
    AUDIO_TRANSCRIPTIONS_PATH,
    EMBEDDINGS_PATH,
    IMAGES_PATH,
    RERANK_PATH,
    GatewayTarget,
)
from orchestrator.gateway.services.http_proxy import (
    forward_request_headers,
    gateway_error,
    passthrough_response_headers,
    prepare_upstream_body,
    proxy_json_post,
    proxy_timeout_seconds,
    read_upstream_error,
    resolve_target_from_body,
    resolve_target_from_model,
    upstream_url_for_path,
    validate_target_launch_mode,
)
import httpx

EMBEDDING_MODES = frozenset({"EMBEDDING"})
RERANKER_MODES = frozenset({"RERANKER"})
IMAGE_MODES = frozenset({"IMAGE"})
TTS_MODES = frozenset({"TTS"})
STT_MODES = frozenset({"STT"})


async def proxy_embeddings(body: dict[str, Any], headers: Any) -> Response:
    target = await resolve_target_from_body(body)
    validate_target_launch_mode(target, EMBEDDING_MODES, "embeddings")
    return await _proxy_json_for_target(
        target,
        EMBEDDINGS_PATH,
        prepare_upstream_body(body, target),
        headers,
    )


async def proxy_rerank(body: dict[str, Any], headers: Any) -> Response:
    target = await resolve_target_from_body(body)
    validate_target_launch_mode(target, RERANKER_MODES, "rerank")
    return await _proxy_json_for_target(
        target,
        RERANK_PATH,
        prepare_upstream_body(body, target),
        headers,
    )


async def proxy_image_generations(body: dict[str, Any], headers: Any) -> Response:
    target = await resolve_target_from_body(body)
    validate_target_launch_mode(target, IMAGE_MODES, "image generation")
    return await _proxy_json_for_target(
        target,
        IMAGES_PATH,
        prepare_upstream_body(body, target),
        headers,
    )


async def proxy_audio_speech(body: dict[str, Any], headers: Any) -> Response:
    target = await resolve_target_from_body(body)
    validate_target_launch_mode(target, TTS_MODES, "text-to-speech")
    return await _proxy_json_for_target(
        target,
        AUDIO_SPEECH_PATH,
        prepare_upstream_body(body, target),
        headers,
    )


async def proxy_audio_transcriptions(request: Request) -> Response:
    form = await request.form()
    model = form.get("model")
    target = await resolve_target_from_model(str(model) if model is not None else "")
    validate_target_launch_mode(target, STT_MODES, "speech-to-text")

    multipart_data: dict[str, str] = {}
    multipart_files: dict[str, tuple[str | None, bytes, str | None]] = {}
    for key, value in form.multi_items():
        if key == "model":
            multipart_data["model"] = target.upstream_model
            continue
        if isinstance(value, UploadFile):
            multipart_files[key] = (
                value.filename,
                await value.read(),
                value.content_type,
            )
            continue
        multipart_data[key] = str(value)

    timeout = proxy_timeout_seconds()
    url = upstream_url_for_path(target, AUDIO_TRANSCRIPTIONS_PATH)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                data=multipart_data,
                files=multipart_files or None,
            )
    except httpx.TimeoutException:
        return gateway_error(504, "gateway_timeout", "Upstream request timed out.")
    except httpx.HTTPError:
        return gateway_error(502, "bad_gateway", "Could not reach the inference backend.")

    if response.status_code >= 400:
        return await read_upstream_error(response)

    content_type = response.headers.get("content-type", "application/json")
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=passthrough_response_headers(response.headers),
        media_type=content_type,
    )


async def _proxy_json_for_target(
    target: GatewayTarget,
    upstream_path: str,
    upstream_body: dict[str, Any],
    headers: Any,
) -> Response:
    request_headers = forward_request_headers(headers)
    timeout = proxy_timeout_seconds()
    url = upstream_url_for_path(target, upstream_path)
    return await proxy_json_post(url, upstream_body, request_headers, timeout)
