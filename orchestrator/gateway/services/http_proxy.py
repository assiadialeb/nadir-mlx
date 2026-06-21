"""Shared HTTP proxy utilities for gateway upstream forwarding."""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

import httpx
from asgiref.sync import sync_to_async
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from orchestrator.gateway.router import GatewayRouteError, GatewayTarget

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
    from orchestrator.gateway.selectors import resolve_gateway_target

    if not isinstance(model, str) or not model.strip():
        raise GatewayRouteError(
            status_code=400,
            code="invalid_model",
            message="The model field is required.",
        )
    return await sync_to_async(resolve_gateway_target, thread_sensitive=False)(model)


async def resolve_target_from_body(body: dict[str, Any]) -> GatewayTarget:
    return await resolve_target_from_model(body.get("model"))


def gateway_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": code, "code": code}},
    )


async def read_upstream_error(response: httpx.Response) -> JSONResponse:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = {"error": {"message": response.text, "type": "upstream_error"}}
    else:
        payload = {"error": {"message": response.text, "type": "upstream_error"}}
    return JSONResponse(status_code=response.status_code, content=payload)


async def proxy_json_post(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> Response:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body, headers=headers)
    except httpx.TimeoutException:
        return gateway_error(504, "gateway_timeout", "Upstream request timed out.")
    except httpx.HTTPError:
        return gateway_error(502, "bad_gateway", "Could not reach the inference backend.")

    if response.status_code >= 400:
        return await read_upstream_error(response)

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
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
