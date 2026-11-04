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
    proxy_ollama_generate,
    proxy_ollama_ps,
    proxy_ollama_root,
    proxy_ollama_show,
    proxy_ollama_tags,
    proxy_ollama_version,
)

router = APIRouter(dependencies=[Depends(verify_gateway_api_key)])


@router.get("/")
async def ollama_root():
    return await proxy_ollama_root()


@router.get("/api/version")
async def ollama_version():
    return await proxy_ollama_version()


@router.get("/api/tags")
async def ollama_tags():
    return await proxy_ollama_tags()


@router.get("/api/ps")
async def ollama_ps():
    return await proxy_ollama_ps()


@router.post("/api/show")
async def ollama_show(request: Request):
    body = await parse_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return await proxy_ollama_show(body)
    except GatewayRouteError as exc:
        return route_error_response(exc)


@router.post("/api/chat")
async def ollama_chat(request: Request):
    body = await parse_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return await proxy_ollama_chat(body, request.headers)
    except GatewayRouteError as exc:
        return route_error_response(exc)


@router.post("/api/generate")
async def ollama_generate(request: Request):
    body = await parse_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return await proxy_ollama_generate(body, request.headers)
    except GatewayRouteError as exc:
        return route_error_response(exc)


@router.post("/api/embed")
async def ollama_embed(request: Request):
    body = await parse_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return await proxy_ollama_embed(body, request.headers, legacy=False)
    except GatewayRouteError as exc:
        return route_error_response(exc)


@router.post("/api/embeddings")
async def ollama_embeddings_legacy(request: Request):
    body = await parse_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return await proxy_ollama_embed(body, request.headers, legacy=True)
    except GatewayRouteError as exc:
        return route_error_response(exc)
