"""Anthropic Messages API façade routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from orchestrator.gateway.auth import verify_gateway_api_key
from orchestrator.gateway.protocol.anthropic_translate import (
    anthropic_error_envelope,
    anthropic_error_type_for_code,
    estimate_anthropic_token_count,
)
from orchestrator.gateway.router import GatewayRouteError
from orchestrator.gateway.services.anthropic_proxy import proxy_anthropic_messages

router = APIRouter(dependencies=[Depends(verify_gateway_api_key)])


def _anthropic_route_error(exc: GatewayRouteError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=anthropic_error_envelope(
            exc.message,
            anthropic_error_type_for_code(exc.code),
        ),
    )


async def _parse_anthropic_json_body(request: Request) -> dict[str, Any] | JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content=anthropic_error_envelope("Request body must be valid JSON."),
        )
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content=anthropic_error_envelope("Request body must be a JSON object."),
        )
    return body


@router.post("/v1/messages")
async def create_message(request: Request):
    body = await _parse_anthropic_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return await proxy_anthropic_messages(body, request.headers)
    except GatewayRouteError as exc:
        return _anthropic_route_error(exc)


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    body = await _parse_anthropic_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return JSONResponse(content=estimate_anthropic_token_count(body))
    except GatewayRouteError as exc:
        return _anthropic_route_error(exc)
