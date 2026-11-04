"""Shared helpers for gateway route handlers."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from orchestrator.gateway.errors import openai_error_envelope
from orchestrator.gateway.router import GatewayRouteError


def route_error_response(exc: GatewayRouteError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_openai_error())


def _content_type_is_json(request: Request) -> bool:
    raw = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    # Absent Content-Type is allowed (CLI clients); reject other media types
    # so browsers cannot issue "simple" cross-site POSTs without a preflight.
    return raw in {"", "application/json", "text/json"}


async def parse_json_body(request: Request) -> dict[str, Any] | JSONResponse:
    if not _content_type_is_json(request):
        return JSONResponse(
            status_code=415,
            content=openai_error_envelope(
                "Content-Type must be application/json.",
                "unsupported_media_type",
            ),
        )
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content=openai_error_envelope(
                "Request body must be valid JSON.",
                "invalid_request",
            ),
        )
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content=openai_error_envelope(
                "Request body must be a JSON object.",
                "invalid_request",
            ),
        )
    return body
