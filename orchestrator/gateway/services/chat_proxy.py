"""HTTP proxy helpers for chat and completion endpoints."""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

import httpx
from asgiref.sync import sync_to_async
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from orchestrator.gateway.router import (
    CHAT_COMPLETIONS_PATH,
    COMPLETIONS_PATH,
    GatewayRouteError,
    GatewayTarget,
)

CHAT_LAUNCH_MODES = frozenset({"TEXT", "MULTIMODAL"})
TEXT_ONLY_LAUNCH_MODES = frozenset({"TEXT"})

FORWARD_REQUEST_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "content-type",
        "user-agent",
    }
)

HOP_BY_HOP_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "content-encoding",
    }
)


def proxy_timeout_seconds() -> float:
    return float(os.environ.get("NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS", "300"))


def forward_request_headers(headers: Any) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in FORWARD_REQUEST_HEADERS:
            forwarded[key] = value
    if "content-type" not in {name.lower() for name in forwarded}:
        forwarded["content-type"] = "application/json"
    return forwarded


def passthrough_response_headers(headers: httpx.Headers) -> dict[str, str]:
    passthrough: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() not in HOP_BY_HOP_RESPONSE_HEADERS:
            passthrough[key] = value
    return passthrough


def prepare_upstream_body(body: dict[str, Any], target: GatewayTarget) -> dict[str, Any]:
    upstream = dict(body)
    upstream["model"] = target.upstream_model
    return upstream


def validate_chat_target(target: GatewayTarget) -> None:
    if target.launch_mode not in CHAT_LAUNCH_MODES:
        raise GatewayRouteError(
            status_code=400,
            code="unsupported_endpoint",
            message=(
                f"Alias '{target.alias}' ({target.launch_mode}) does not support "
                "chat completions."
            ),
        )


def validate_completions_target(target: GatewayTarget) -> None:
    if target.launch_mode not in TEXT_ONLY_LAUNCH_MODES:
        raise GatewayRouteError(
            status_code=400,
            code="unsupported_endpoint",
            message=(
                f"Alias '{target.alias}' ({target.launch_mode}) does not support "
                "text completions."
            ),
        )


def upstream_url_for_path(target: GatewayTarget, path: str) -> str:
    return f"{target.base_url}{path}"


async def _read_upstream_error(response: httpx.Response) -> JSONResponse:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = {"error": {"message": response.text, "type": "upstream_error"}}
    else:
        payload = {"error": {"message": response.text, "type": "upstream_error"}}
    return JSONResponse(status_code=response.status_code, content=payload)


async def _proxy_buffered(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> Response:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body, headers=headers)
    except httpx.TimeoutException:
        return _gateway_error(504, "gateway_timeout", "Upstream request timed out.")
    except httpx.HTTPError:
        return _gateway_error(502, "bad_gateway", "Could not reach the inference backend.")

    if response.status_code >= 400:
        return await _read_upstream_error(response)

    return JSONResponse(
        status_code=response.status_code,
        content=response.json(),
        headers=passthrough_response_headers(response.headers),
    )


async def _stream_upstream_chunks(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> AsyncIterator[bytes]:
    client = httpx.AsyncClient(timeout=timeout)
    try:
        request = client.build_request("POST", url, json=body, headers=headers)
        response = await client.send(request, stream=True)
    except httpx.TimeoutException:
        await client.aclose()
        yield _encode_sse_error("Upstream request timed out.")
        return
    except httpx.HTTPError:
        await client.aclose()
        yield _encode_sse_error("Could not reach the inference backend.")
        return

    if response.status_code >= 400:
        error_body = await response.aread()
        await response.aclose()
        await client.aclose()
        yield error_body
        return

    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()
        await client.aclose()


def _encode_sse_error(message: str) -> bytes:
    payload = json.dumps({"error": {"message": message, "type": "gateway_error"}})
    return f"data: {payload}\n\n".encode()


def _gateway_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": code, "code": code}},
    )


async def proxy_chat_completions(
    body: dict[str, Any],
    headers: Any,
) -> Response:
    """Proxy a chat completion request to a TEXT or MULTIMODAL instance."""
    target = await _resolve_from_body(body)
    validate_chat_target(target)
    upstream_body = prepare_upstream_body(body, target)
    request_headers = forward_request_headers(headers)
    timeout = proxy_timeout_seconds()
    url = upstream_url_for_path(target, CHAT_COMPLETIONS_PATH)

    if body.get("stream"):
        return StreamingResponse(
            _stream_upstream_chunks(url, upstream_body, request_headers, timeout),
            media_type="text/event-stream",
        )
    return await _proxy_buffered(url, upstream_body, request_headers, timeout)


async def proxy_text_completions(
    body: dict[str, Any],
    headers: Any,
) -> Response:
    """Proxy a legacy text completion request to a TEXT instance."""
    target = await _resolve_from_body(body)
    validate_completions_target(target)
    upstream_body = prepare_upstream_body(body, target)
    request_headers = forward_request_headers(headers)
    timeout = proxy_timeout_seconds()
    url = upstream_url_for_path(target, COMPLETIONS_PATH)

    if body.get("stream"):
        return StreamingResponse(
            _stream_upstream_chunks(url, upstream_body, request_headers, timeout),
            media_type="text/event-stream",
        )
    return await _proxy_buffered(url, upstream_body, request_headers, timeout)


async def _resolve_from_body(body: dict[str, Any]) -> GatewayTarget:
    from orchestrator.gateway.selectors import resolve_gateway_target

    model = body.get("model")
    if not isinstance(model, str):
        raise GatewayRouteError(
            status_code=400,
            code="invalid_model",
            message="The model field must be a string.",
        )
    return await sync_to_async(resolve_gateway_target, thread_sensitive=False)(model)
