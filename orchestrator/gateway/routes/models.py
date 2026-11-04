"""Gateway model discovery routes."""

from __future__ import annotations

from asgiref.sync import sync_to_async
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from orchestrator.gateway.router import GatewayRouteError
from orchestrator.gateway.selectors import get_gateway_model, list_running_gateway_models

router = APIRouter()


@router.get("/v1/models")
async def list_models() -> JSONResponse:
    payload = await sync_to_async(list_running_gateway_models, thread_sensitive=False)()
    return JSONResponse(content=payload)


@router.get("/v1/models/{model_id:path}")
async def retrieve_model(model_id: str) -> JSONResponse:
    try:
        payload = await sync_to_async(get_gateway_model, thread_sensitive=False)(model_id)
    except GatewayRouteError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.to_openai_error())
    return JSONResponse(content=payload)
