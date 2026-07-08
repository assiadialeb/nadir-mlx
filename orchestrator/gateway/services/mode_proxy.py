"""Proxy helpers for non-chat OpenAI-compatible MLX endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.responses import Response, StreamingResponse

from orchestrator.gateway.router import (
    AUDIO_SPEECH_PATH,
    AUDIO_TRANSCRIPTIONS_PATH,
    AUDIO_TRANSCRIPTIONS_STREAM_PATH,
    AUDIO_TRANSLATIONS_PATH,
    EMBEDDINGS_PATH,
    IMAGE_EDITS_PATH,
    IMAGE_VARIATIONS_PATH,
    IMAGES_PATH,
    RERANK_PATH,
    GatewayTarget,
)
from orchestrator.gateway.services.http_proxy import (
    CONTENT_TYPE_JSON,
    MSG_BAD_GATEWAY,
    MSG_UPSTREAM_TIMEOUT,
    forward_request_headers,
    gateway_error,
    passthrough_response_headers,
    prepare_upstream_body,
    proxy_binary_post,
    proxy_json_post,
    httpx_client_timeout,
    proxy_timeout_seconds,
    read_upstream_error,
    resolve_target_from_body,
    resolve_target_from_model,
    upstream_url_for_path,
    validate_target_launch_mode,
)
from orchestrator.gateway.upstream_concurrency import (
    UpstreamQueueTimeoutError,
    upstream_concurrency_slot,
)
import httpx

EMBEDDING_MODES = frozenset({"EMBEDDING"})
RERANKER_MODES = frozenset({"RERANKER"})
IMAGE_MODES = frozenset({"IMAGE"})
TTS_MODES = frozenset({"TTS"})
STT_MODES = frozenset({"STT"})


def _is_upload_file(value: object) -> bool:
    return isinstance(value, (StarletteUploadFile,))


async def _parse_multipart_payload(
    form: Any,
    target: GatewayTarget,
) -> tuple[dict[str, str], dict[str, tuple[str | None, bytes, str | None]]]:
    multipart_data: dict[str, str] = {}
    multipart_files: dict[str, tuple[str | None, bytes, str | None]] = {}
    for key, value in form.multi_items():
        if key == "model":
            multipart_data["model"] = target.upstream_model
            continue
        if _is_upload_file(value):
            multipart_files[key] = (
                value.filename,
                await value.read(),
                value.content_type,
            )
            continue
        multipart_data[key] = str(value)
    return multipart_data, multipart_files


def _gateway_transport_error(exc: Exception) -> Response:
    if isinstance(exc, UpstreamQueueTimeoutError):
        return gateway_error(503, "upstream_queue_timeout", exc.message)
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return gateway_error(504, "gateway_timeout", MSG_UPSTREAM_TIMEOUT)
    return gateway_error(502, "bad_gateway", MSG_BAD_GATEWAY)


async def _upstream_stream_error_response(
    response: httpx.Response,
    client: httpx.AsyncClient,
) -> Response:
    error_body = await response.aread()
    await response.aclose()
    await client.aclose()
    content_type = response.headers.get("content-type", CONTENT_TYPE_JSON)
    if CONTENT_TYPE_JSON in content_type:
        try:
            payload = json.loads(error_body)
        except json.JSONDecodeError:
            payload = {"error": {"message": error_body.decode(), "type": "upstream_error"}}
        return JSONResponse(status_code=response.status_code, content=payload)
    return Response(
        content=error_body,
        status_code=response.status_code,
        media_type=content_type,
    )


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


async def _proxy_multipart_request(
    request: Request,
    upstream_path: str,
    *,
    allowed_modes: frozenset[str],
    endpoint_label: str,
) -> Response:
    form = await request.form()
    model = form.get("model")
    target = await resolve_target_from_model(str(model) if model is not None else "")
    validate_target_launch_mode(target, allowed_modes, endpoint_label)

    multipart_data, multipart_files = await _parse_multipart_payload(form, target)
    timeout = proxy_timeout_seconds()
    url = upstream_url_for_path(target, upstream_path)
    try:
        async with upstream_concurrency_slot(target):
            async with asyncio.timeout(timeout):
                async with httpx.AsyncClient(timeout=httpx_client_timeout()) as client:
                    response = await client.post(
                        url,
                        data=multipart_data,
                        files=multipart_files or None,
                    )
    except UpstreamQueueTimeoutError as exc:
        return gateway_error(503, "upstream_queue_timeout", exc.message)
    except (TimeoutError, httpx.TimeoutException):
        return gateway_error(504, "gateway_timeout", MSG_UPSTREAM_TIMEOUT)
    except httpx.HTTPError:
        return gateway_error(502, "bad_gateway", MSG_BAD_GATEWAY)

    if response.status_code >= 400:
        return read_upstream_error(response)

    content_type = response.headers.get("content-type", CONTENT_TYPE_JSON)
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=passthrough_response_headers(response.headers),
        media_type=content_type,
    )


async def _proxy_multipart_stream_request(
    request: Request,
    upstream_path: str,
    *,
    allowed_modes: frozenset[str],
    endpoint_label: str,
) -> Response:
    form = await request.form()
    model = form.get("model")
    target = await resolve_target_from_model(str(model) if model is not None else "")
    validate_target_launch_mode(target, allowed_modes, endpoint_label)

    multipart_data, multipart_files = await _parse_multipart_payload(form, target)

    timeout = proxy_timeout_seconds()
    url = upstream_url_for_path(target, upstream_path)
    client = httpx.AsyncClient(timeout=httpx_client_timeout())
    try:
        async with upstream_concurrency_slot(target):
            async with asyncio.timeout(timeout):
                upstream_request = client.build_request(
                    "POST",
                    url,
                    data=multipart_data,
                    files=multipart_files or None,
                )
                response = await client.send(upstream_request, stream=True)
    except (UpstreamQueueTimeoutError, TimeoutError, httpx.TimeoutException, httpx.HTTPError) as exc:
        await client.aclose()
        return _gateway_transport_error(exc)

    if response.status_code >= 400:
        return await _upstream_stream_error_response(response, client)

    media_type = response.headers.get("content-type", "text/event-stream")
    passthrough = passthrough_response_headers(response.headers)

    async def iter_upstream() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        iter_upstream(),
        status_code=response.status_code,
        media_type=media_type,
        headers=passthrough,
    )


async def proxy_image_edits(request: Request) -> Response:
    return await _proxy_multipart_request(
        request,
        IMAGE_EDITS_PATH,
        allowed_modes=IMAGE_MODES,
        endpoint_label="image edits",
    )


async def proxy_image_variations(request: Request) -> Response:
    return await _proxy_multipart_request(
        request,
        IMAGE_VARIATIONS_PATH,
        allowed_modes=IMAGE_MODES,
        endpoint_label="image variations",
    )


async def proxy_audio_speech(body: dict[str, Any], headers: Any) -> Response:
    target = await resolve_target_from_body(body)
    validate_target_launch_mode(target, TTS_MODES, "text-to-speech")
    request_headers = forward_request_headers(headers)
    url = upstream_url_for_path(target, AUDIO_SPEECH_PATH)
    return await proxy_binary_post(
        target,
        url,
        prepare_upstream_body(body, target),
        request_headers,
    )


async def proxy_audio_transcriptions(request: Request) -> Response:
    return await _proxy_multipart_request(
        request,
        AUDIO_TRANSCRIPTIONS_PATH,
        allowed_modes=STT_MODES,
        endpoint_label="speech-to-text",
    )


async def proxy_audio_transcriptions_stream(request: Request) -> Response:
    return await _proxy_multipart_stream_request(
        request,
        AUDIO_TRANSCRIPTIONS_STREAM_PATH,
        allowed_modes=STT_MODES,
        endpoint_label="speech-to-text streaming",
    )


async def proxy_audio_translations(request: Request) -> Response:
    return await _proxy_multipart_request(
        request,
        AUDIO_TRANSLATIONS_PATH,
        allowed_modes=STT_MODES,
        endpoint_label="speech-to-text",
    )


async def _proxy_json_for_target(
    target: GatewayTarget,
    upstream_path: str,
    upstream_body: dict[str, Any],
    headers: Any,
) -> Response:
    request_headers = forward_request_headers(headers)
    url = upstream_url_for_path(target, upstream_path)
    return await proxy_json_post(target, url, upstream_body, request_headers)
