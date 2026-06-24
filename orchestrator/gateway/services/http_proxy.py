"""Shared HTTP proxy utilities for gateway upstream forwarding."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator

import httpx
from asgiref.sync import sync_to_async
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from orchestrator.gateway.router import GatewayRouteError, GatewayTarget

CONTENT_TYPE_JSON = "application/json"
MSG_UPSTREAM_TIMEOUT = "Upstream request timed out."
MSG_BAD_GATEWAY = "Could not reach the inference backend."

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


def httpx_client_timeout() -> httpx.Timeout:
    """Match httpx read deadline to NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS."""
    seconds = proxy_timeout_seconds()
    return httpx.Timeout(seconds)


def forward_request_headers(headers: Any) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in FORWARD_REQUEST_HEADERS:
            forwarded[key] = value
    if "content-type" not in {name.lower() for name in forwarded}:
        forwarded["content-type"] = CONTENT_TYPE_JSON
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


def upstream_url_for_path(target: GatewayTarget, path: str) -> str:
    return f"{target.base_url}{path}"


def validate_target_launch_mode(
    target: GatewayTarget,
    allowed_modes: frozenset[str],
    endpoint_label: str,
) -> None:
    if target.launch_mode in allowed_modes:
        return
    raise GatewayRouteError(
        status_code=400,
        code="unsupported_endpoint",
        message=(
            f"Alias '{target.alias}' ({target.launch_mode}) does not support "
            f"{endpoint_label}."
        ),
    )


async def resolve_target_from_model(model: object) -> GatewayTarget:
    from orchestrator.lifecycle_services import ensure_instance_ready, touch_instance_last_used_at

    if not isinstance(model, str) or not model.strip():
        raise GatewayRouteError(
            status_code=400,
            code="invalid_model",
            message="The model field is required.",
        )
    target = await sync_to_async(ensure_instance_ready, thread_sensitive=False)(model)
    await sync_to_async(
        touch_instance_last_used_at,
        thread_sensitive=False,
    )(target.instance_id)
    return target


async def resolve_target_from_body(body: dict[str, Any]) -> GatewayTarget:
    return await resolve_target_from_model(body.get("model"))


def gateway_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": code, "code": code}},
    )


def read_upstream_error(response: httpx.Response) -> JSONResponse:
    content_type = response.headers.get("content-type", "")
    if CONTENT_TYPE_JSON in content_type:
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = {"error": {"message": response.text, "type": "upstream_error"}}
    else:
        payload = {"error": {"message": response.text, "type": "upstream_error"}}
    return JSONResponse(status_code=response.status_code, content=payload)


async def proxy_binary_post(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> Response:
    """Forward a JSON POST and stream the upstream binary body without buffering."""
    client = httpx.AsyncClient(timeout=httpx_client_timeout())
    try:
        async with asyncio.timeout(proxy_timeout_seconds()):
            request = client.build_request("POST", url, json=body, headers=headers)
            response = await client.send(request, stream=True)
    except (TimeoutError, httpx.TimeoutException):
        await client.aclose()
        return gateway_error(504, "gateway_timeout", MSG_UPSTREAM_TIMEOUT)
    except httpx.HTTPError:
        await client.aclose()
        return gateway_error(502, "bad_gateway", MSG_BAD_GATEWAY)

    if response.status_code >= 400:
        error_body = await response.aread()
        await response.aclose()
        await client.aclose()
        content_type = response.headers.get("content-type", "")
        if CONTENT_TYPE_JSON in content_type:
            try:
                payload = json.loads(error_body)
            except json.JSONDecodeError:
                payload = {"error": {"message": error_body.decode(), "type": "upstream_error"}}
            return JSONResponse(status_code=response.status_code, content=payload)
        return JSONResponse(
            status_code=response.status_code,
            content={"error": {"message": error_body.decode(), "type": "upstream_error"}},
        )

    media_type = response.headers.get("content-type", "application/octet-stream")
    passthrough = passthrough_response_headers(response.headers)

    async def iter_chunks() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        iter_chunks(),
        status_code=response.status_code,
        media_type=media_type,
        headers=passthrough,
    )


async def proxy_json_post(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> Response:
    try:
        async with asyncio.timeout(proxy_timeout_seconds()):
            async with httpx.AsyncClient(timeout=httpx_client_timeout()) as client:
                response = await client.post(url, json=body, headers=headers)
    except (TimeoutError, httpx.TimeoutException):
        return gateway_error(504, "gateway_timeout", MSG_UPSTREAM_TIMEOUT)
    except httpx.HTTPError:
        return gateway_error(502, "bad_gateway", MSG_BAD_GATEWAY)

    if response.status_code >= 400:
        return read_upstream_error(response)

    content_type = response.headers.get("content-type", "")
    if CONTENT_TYPE_JSON in content_type:
        return JSONResponse(
            status_code=response.status_code,
            content=response.json(),
            headers=passthrough_response_headers(response.headers),
        )
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=passthrough_response_headers(response.headers),
        media_type=content_type or None,
    )


async def stream_upstream_chunks(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> AsyncIterator[bytes]:
    client = httpx.AsyncClient(timeout=httpx_client_timeout())
    try:
        async with asyncio.timeout(proxy_timeout_seconds()):
            request = client.build_request("POST", url, json=body, headers=headers)
            response = await client.send(request, stream=True)
    except (TimeoutError, httpx.TimeoutException):
        await client.aclose()
        yield _encode_sse_error(MSG_UPSTREAM_TIMEOUT)
        return
    except httpx.HTTPError:
        await client.aclose()
        yield _encode_sse_error(MSG_BAD_GATEWAY)
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
