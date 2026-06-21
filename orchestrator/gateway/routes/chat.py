"""Chat and completion proxy routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from orchestrator.gateway.router import GatewayRouteError
from orchestrator.gateway.services.chat_proxy import (
    proxy_chat_completions,
    proxy_text_completions,
)

router = APIRouter()


def _route_error_response(exc: GatewayRouteError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_openai_error())


async def _parse_json_body(request: Request) -> dict[str, Any] | JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Request body must be valid JSON.",
                    "type": "invalid_request",
                }
            },
        )
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Request body must be a JSON object.",
                    "type": "invalid_request",
                }
            },
        )
    return body


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await _parse_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return await proxy_chat_completions(body, request.headers)
    except GatewayRouteError as exc:
        return _route_error_response(exc)


@router.post("/v1/completions")
async def text_completions(request: Request):
    body = await _parse_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return await proxy_text_completions(body, request.headers)
    except GatewayRouteError as exc:
        return _route_error_response(exc)
