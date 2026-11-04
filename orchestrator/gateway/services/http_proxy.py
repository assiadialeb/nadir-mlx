"""Shared HTTP proxy utilities for gateway upstream forwarding."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator

import httpx
from asgiref.sync import sync_to_async
from django.conf import settings
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from orchestrator.env_utils import env_float

from orchestrator.gateway.http_client import get_http_client
from orchestrator.gateway.response_normalize import (
    normalize_client_chat_payload,
    restore_response_model,
)
from orchestrator.gateway.router import GatewayRouteError, GatewayTarget
from orchestrator.gateway.upstream_concurrency import (
    MSG_QUEUE_TIMEOUT,
    UpstreamQueueTimeoutError,
    upstream_concurrency_slot,
)

CONTENT_TYPE_JSON = "application/json"
MSG_UPSTREAM_TIMEOUT = "Upstream request timed out."
MSG_BAD_GATEWAY = "Could not reach the inference backend."
_DEFAULT_ACTIVITY_HEARTBEAT_SECONDS = 15.0

logger = logging.getLogger(__name__)

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
    return env_float(
        "NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS",
        float(settings.NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS),
    )


def httpx_client_timeout() -> httpx.Timeout:
    """Match httpx read deadline to NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS."""
    seconds = proxy_timeout_seconds()
    return httpx.Timeout(seconds)


def forward_request_headers(headers: Any) -> dict[str, str]:
    from orchestrator.env_utils import env_str

    forwarded: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in FORWARD_REQUEST_HEADERS:
            forwarded[key] = value
    if "content-type" not in {name.lower() for name in forwarded}:
        forwarded["content-type"] = CONTENT_TYPE_JSON
    # Inject worker auth when configured (image/stt/embedding/tts Nadir servers).
    inference_key = env_str("NADIR_INFERENCE_API_KEY", "")
    if inference_key:
        forwarded["authorization"] = f"Bearer {inference_key}"
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


def rewrite_sse_line_model(line: bytes, client_model_alias: str) -> bytes:
    """Normalize JSON inside a single SSE ``data:`` line for OpenAI clients."""
    if not line.startswith(b"data:"):
        return line
    raw = line[5:].strip()
    if not raw or raw == b"[DONE]":
        return line
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return line
    restored = normalize_client_chat_payload(payload, client_model_alias)
    if restored is payload:
        return line
    encoded = json.dumps(restored, separators=(",", ":")).encode()
    return b"data: " + encoded


def is_usage_only_sse_line(line: bytes) -> bool:
    """Return True for mlx-lm usage-only chunks (``choices: []`` + ``usage``).

    OpenAI SDKs that index ``chunk.choices[0]`` crash on these lines when the
    client did not opt into ``stream_options.include_usage`` (regression C1).
    """
    if not line.startswith(b"data:"):
        return False
    raw = line[5:].strip()
    if not raw or raw == b"[DONE]":
        return False
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    choices = payload.get("choices")
    usage = payload.get("usage")
    return isinstance(choices, list) and len(choices) == 0 and isinstance(usage, dict)


async def iter_sse_with_restored_model(
    chunks: AsyncIterator[bytes],
    client_model_alias: str,
) -> AsyncIterator[bytes]:
    """Transform an upstream SSE byte stream so ``model`` matches the gateway alias."""
    from orchestrator.gateway.request_timing import get_active_collector

    collector = get_active_collector()
    strip_usage_only = bool(
        collector is not None and getattr(collector, "strip_usage_only", False)
    )
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            restored = rewrite_sse_line_model(line, client_model_alias)
            if collector is not None:
                collector.observe_sse_line(restored)
            if strip_usage_only and is_usage_only_sse_line(restored):
                continue
            yield restored + b"\n"
    if buffer:
        restored = rewrite_sse_line_model(buffer, client_model_alias)
        if collector is not None:
            collector.observe_sse_line(restored)
        if not (strip_usage_only and is_usage_only_sse_line(restored)):
            yield restored


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
    import time

    from orchestrator.gateway.request_timing import get_active_collector
    from orchestrator.lifecycle_services import ensure_instance_ready

    if not isinstance(model, str) or not model.strip():
        raise GatewayRouteError(
            status_code=400,
            code="invalid_model",
            message="The model field is required.",
        )
    collector = get_active_collector()
    wake_started = time.perf_counter()
    target = await sync_to_async(ensure_instance_ready, thread_sensitive=False)(model)
    if collector is not None:
        # cold_start may already be set inside lifecycle wake path.
        if collector.wake_ms is None:
            collector.mark_wake(
                (time.perf_counter() - wake_started) * 1000.0,
                cold_start=False,
            )
        else:
            # Keep cold_start flag; refresh duration to include full ensure wait.
            collector.mark_wake(
                (time.perf_counter() - wake_started) * 1000.0,
                cold_start=collector.cold_start,
            )
        collector.bind_target(alias=target.alias, instance_id=target.instance_id)
    await touch_target_last_used_at(target)
    return target


async def resolve_target_from_body(body: dict[str, Any]) -> GatewayTarget:
    return await resolve_target_from_model(body.get("model"))


async def touch_target_last_used_at(target: GatewayTarget) -> None:
    """Refresh LRU / idle-offload activity (best-effort; never fails the request)."""
    from orchestrator.lifecycle_services import touch_instance_last_used_at

    try:
        await sync_to_async(
            touch_instance_last_used_at,
            thread_sensitive=False,
        )(target.instance_id)
    except Exception:
        logger.warning(
            "Failed to refresh last_used_at for instance %s",
            target.instance_id,
            exc_info=True,
        )


def activity_heartbeat_seconds() -> float:
    """Interval for mid-request ``last_used_at`` refresh during long calls."""
    return max(1.0, env_float(
        "NADIR_GATEWAY_ACTIVITY_HEARTBEAT_SECONDS",
        _DEFAULT_ACTIVITY_HEARTBEAT_SECONDS,
    ))


async def _activity_heartbeat_loop(target: GatewayTarget) -> None:
    """Periodically refresh activity until cancelled."""
    interval = activity_heartbeat_seconds()
    while True:
        await asyncio.sleep(interval)
        await touch_target_last_used_at(target)


@asynccontextmanager
async def _activity_heartbeat(target: GatewayTarget):
    """Run a background activity heartbeat for the duration of a proxy call."""
    heartbeat = asyncio.create_task(
        _activity_heartbeat_loop(target),
        name=f"nadir-activity-heartbeat-{target.instance_id}",
    )
    try:
        yield
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat


def gateway_error(status_code: int, code: str, message: str) -> JSONResponse:
    from orchestrator.gateway.errors import openai_error_envelope

    return JSONResponse(
        status_code=status_code,
        content=openai_error_envelope(message, code),
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
    target: GatewayTarget,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> Response:
    """Forward a JSON POST and stream the upstream binary body without buffering."""
    try:
        async with upstream_concurrency_slot(target):
            return await _proxy_binary_post_impl(url, body, headers)
    except UpstreamQueueTimeoutError as exc:
        return gateway_error(503, "upstream_queue_timeout", exc.message)


async def _proxy_binary_post_impl(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> Response:
    client = get_http_client()
    try:
        async with asyncio.timeout(proxy_timeout_seconds()):
            request = client.build_request("POST", url, json=body, headers=headers)
            response = await client.send(request, stream=True)
    except (TimeoutError, httpx.TimeoutException):
        return gateway_error(504, "gateway_timeout", MSG_UPSTREAM_TIMEOUT)
    except httpx.HTTPError:
        return gateway_error(502, "bad_gateway", MSG_BAD_GATEWAY)

    if response.status_code >= 400:
        error_body = await response.aread()
        await response.aclose()
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

    return StreamingResponse(
        iter_chunks(),
        status_code=response.status_code,
        media_type=media_type,
        headers=passthrough,
    )


async def proxy_json_post(
    target: GatewayTarget,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> Response:
    from orchestrator.gateway.request_timing import get_active_collector

    collector = get_active_collector()
    async with _activity_heartbeat(target):
        try:
            async with upstream_concurrency_slot(target):
                if collector is not None:
                    collector.mark_upstream_open()
                response = await _proxy_json_post_impl(url, body, headers, target.alias)
        except UpstreamQueueTimeoutError as exc:
            if collector is not None:
                collector.set_error("upstream_queue_timeout")
            return gateway_error(503, "upstream_queue_timeout", exc.message)
        if response.status_code < 400:
            await touch_target_last_used_at(target)
        elif collector is not None:
            collector.set_error("upstream_error")
        return response


async def _proxy_json_post_impl(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    client_model_alias: str,
) -> Response:
    from orchestrator.gateway.request_timing import get_active_collector

    collector = get_active_collector()
    client = get_http_client()
    try:
        async with asyncio.timeout(proxy_timeout_seconds()):
            response = await client.post(url, json=body, headers=headers)
    except (TimeoutError, httpx.TimeoutException):
        if collector is not None:
            collector.set_error("gateway_timeout")
        return gateway_error(504, "gateway_timeout", MSG_UPSTREAM_TIMEOUT)
    except httpx.HTTPError:
        if collector is not None:
            collector.set_error("bad_gateway")
        return gateway_error(502, "bad_gateway", MSG_BAD_GATEWAY)

    if response.status_code >= 400:
        return read_upstream_error(response)

    content_type = response.headers.get("content-type", "")
    if CONTENT_TYPE_JSON in content_type:
        payload = normalize_client_chat_payload(response.json(), client_model_alias)
        if collector is not None:
            collector.mark_json_complete(payload)
        return JSONResponse(
            status_code=response.status_code,
            content=payload,
            headers=passthrough_response_headers(response.headers),
        )
    if collector is not None:
        collector.mark_json_complete(None)
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=passthrough_response_headers(response.headers),
        media_type=content_type or None,
    )


async def read_upstream_stream_error(response: httpx.Response) -> Response:
    """Consume a failed streamed upstream response and return a JSON/HTTP error."""
    error_body = await response.aread()
    await response.aclose()
    content_type = response.headers.get("content-type", "")
    if CONTENT_TYPE_JSON in content_type:
        try:
            payload = json.loads(error_body)
        except json.JSONDecodeError:
            payload = {
                "error": {
                    "message": error_body.decode(errors="replace"),
                    "type": "upstream_error",
                    "code": "upstream_error",
                }
            }
        return JSONResponse(status_code=response.status_code, content=payload)
    return JSONResponse(
        status_code=response.status_code,
        content={
            "error": {
                "message": error_body.decode(errors="replace"),
                "type": "upstream_error",
                "code": "upstream_error",
            }
        },
    )


async def proxy_sse_post(
    target: GatewayTarget,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> Response:
    """Open upstream SSE before responding; propagate non-2xx as real HTTP errors."""
    from orchestrator.gateway.request_timing import get_active_collector

    collector = get_active_collector()
    state: dict[str, Any] = {}
    early_error = await _open_upstream_sse_session(target, url, body, headers, state)
    if early_error is not None:
        return early_error
    if collector is not None:
        collector.mark_upstream_open()

    async def iter_chunks() -> AsyncIterator[bytes]:
        from orchestrator.gateway.request_timing import reset_request_timing

        response: httpx.Response = state["response"]
        async with _activity_heartbeat(target):
            try:
                async for chunk in iter_sse_with_restored_model(
                    response.aiter_bytes(),
                    target.alias,
                ):
                    yield chunk
            except (httpx.HTTPError, TimeoutError):
                if collector is not None:
                    collector.set_error("bad_gateway")
                yield _encode_sse_error(MSG_BAD_GATEWAY)
            finally:
                await _close_upstream_sse_session(state)
                await touch_target_last_used_at(target)
                reset_request_timing(collector)

    return StreamingResponse(
        iter_chunks(),
        status_code=200,
        media_type="text/event-stream",
    )


async def _open_upstream_sse_session(
    target: GatewayTarget,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    state: dict[str, Any],
) -> Response | None:
    """Acquire queue slot and open upstream stream. Return error Response or None."""
    slot = upstream_concurrency_slot(target)
    try:
        await slot.__aenter__()
    except UpstreamQueueTimeoutError as exc:
        return gateway_error(503, "upstream_queue_timeout", exc.message)
    state["slot"] = slot

    client = get_http_client()
    try:
        async with asyncio.timeout(proxy_timeout_seconds()):
            request = client.build_request("POST", url, json=body, headers=headers)
            response = await client.send(request, stream=True)
    except (TimeoutError, httpx.TimeoutException):
        await _close_upstream_sse_session(state)
        return gateway_error(504, "gateway_timeout", MSG_UPSTREAM_TIMEOUT)
    except httpx.HTTPError:
        await _close_upstream_sse_session(state)
        return gateway_error(502, "bad_gateway", MSG_BAD_GATEWAY)

    state["response"] = response
    if response.status_code < 400:
        return None

    error_response = await read_upstream_stream_error(response)
    state.pop("response", None)
    await _close_upstream_sse_session(state)
    return error_response


async def _close_upstream_sse_session(state: dict[str, Any]) -> None:
    response = state.pop("response", None)
    if response is not None:
        await response.aclose()
    slot = state.pop("slot", None)
    if slot is not None:
        await slot.__aexit__(None, None, None)


def _encode_sse_error(message: str) -> bytes:
    payload = json.dumps({"error": {"message": message, "type": "gateway_error"}})
    return f"data: {payload}\n\n".encode()
