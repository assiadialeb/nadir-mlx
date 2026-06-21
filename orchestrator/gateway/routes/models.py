"""Gateway model discovery routes."""

from __future__ import annotations

from asgiref.sync import sync_to_async
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from orchestrator.gateway.selectors import list_running_gateway_models

router = APIRouter()


@router.get("/v1/models")
async def list_models() -> JSONResponse:
    payload = await sync_to_async(list_running_gateway_models, thread_sensitive=False)()
    return JSONResponse(content=payload)
