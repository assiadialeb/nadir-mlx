"""Ollama-compatible API façade routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from orchestrator.gateway.auth import verify_gateway_api_key
from orchestrator.gateway.router import GatewayRouteError
from orchestrator.gateway.routes.common import parse_json_body, route_error_response
from orchestrator.gateway.services.ollama_proxy import (
    proxy_ollama_chat,
    proxy_ollama_embed,
    proxy_ollama_tags,
)

router = APIRouter(dependencies=[Depends(verify_gateway_api_key)])


@router.get("/api/tags")
async def ollama_tags():
    return await proxy_ollama_tags()


@router.post("/api/chat")
async def ollama_chat(request: Request):
    body = await parse_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return await proxy_ollama_chat(body, request.headers)
    except GatewayRouteError as exc:
        return route_error_response(exc)


@router.post("/api/embed")
@router.post("/api/embeddings")
async def ollama_embed(request: Request):
    body = await parse_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return await proxy_ollama_embed(body, request.headers)
    except GatewayRouteError as exc:
        return route_error_response(exc)
