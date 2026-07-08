"""Multi-mode OpenAI-compatible proxy routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from orchestrator.gateway.auth import verify_gateway_api_key

from orchestrator.gateway.router import GatewayRouteError
from orchestrator.gateway.routes.common import parse_json_body, route_error_response
from orchestrator.gateway.services.mode_proxy import (
    proxy_audio_speech,
    proxy_audio_transcriptions,
    proxy_audio_transcriptions_stream,
    proxy_audio_translations,
    proxy_embeddings,
    proxy_image_edits,
    proxy_image_generations,
    proxy_image_variations,
    proxy_rerank,
)

router = APIRouter(dependencies=[Depends(verify_gateway_api_key)])


async def _handle_json_route(request: Request, handler):
    body = await parse_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return await handler(body, request.headers)
    except GatewayRouteError as exc:
        return route_error_response(exc)


@router.post("/v1/embeddings")
async def embeddings(request: Request):
    return await _handle_json_route(request, proxy_embeddings)


@router.post("/v1/rerank")
async def rerank(request: Request):
    return await _handle_json_route(request, proxy_rerank)


@router.post("/v1/images/generations")
async def image_generations(request: Request):
    return await _handle_json_route(request, proxy_image_generations)


@router.post("/v1/images/edits")
async def image_edits(request: Request):
    try:
        return await proxy_image_edits(request)
    except GatewayRouteError as exc:
        return route_error_response(exc)


@router.post("/v1/images/variations")
async def image_variations(request: Request):
    try:
        return await proxy_image_variations(request)
    except GatewayRouteError as exc:
        return route_error_response(exc)


@router.post("/v1/audio/speech")
async def audio_speech(request: Request):
    return await _handle_json_route(request, proxy_audio_speech)


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request):
    try:
        return await proxy_audio_transcriptions(request)
    except GatewayRouteError as exc:
        return route_error_response(exc)


@router.post("/v1/audio/transcriptions/stream")
async def audio_transcriptions_stream(request: Request):
    try:
        return await proxy_audio_transcriptions_stream(request)
    except GatewayRouteError as exc:
        return route_error_response(exc)


@router.post("/v1/audio/translations")
async def audio_translations(request: Request):
    try:
        return await proxy_audio_translations(request)
    except GatewayRouteError as exc:
        return route_error_response(exc)
