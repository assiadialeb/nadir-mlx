"""Chat and completion proxy routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from orchestrator.gateway.router import GatewayRouteError
from orchestrator.gateway.routes.common import parse_json_body, route_error_response
from orchestrator.gateway.services.chat_proxy import (
    proxy_chat_completions,
    proxy_text_completions,
)

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await parse_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return await proxy_chat_completions(body, request.headers)
    except GatewayRouteError as exc:
        return route_error_response(exc)


@router.post("/v1/completions")
async def text_completions(request: Request):
    body = await parse_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return await proxy_text_completions(body, request.headers)
    except GatewayRouteError as exc:
        return route_error_response(exc)
