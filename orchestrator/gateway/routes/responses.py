"""OpenAI Responses API façade routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from orchestrator.gateway.auth import verify_gateway_api_key
from orchestrator.gateway.router import GatewayRouteError
from orchestrator.gateway.routes.common import parse_json_body, route_error_response
from orchestrator.gateway.services.responses_proxy import proxy_responses_create

router = APIRouter(dependencies=[Depends(verify_gateway_api_key)])


@router.post("/v1/responses")
async def create_response(request: Request):
    body = await parse_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return await proxy_responses_create(body, request.headers)
    except GatewayRouteError as exc:
        return route_error_response(exc)
