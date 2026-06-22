"""FastAPI application for the Nadir Gateway worker."""

from __future__ import annotations

from fastapi import Depends, FastAPI

from orchestrator.gateway.auth import verify_gateway_api_key

from orchestrator.gateway.routes.chat import router as chat_router
from orchestrator.gateway.routes.image_files import router as image_files_router
from orchestrator.gateway.routes.models import router as models_router
from orchestrator.gateway.routes.modes import router as modes_router


def create_app() -> FastAPI:
    """Build the gateway FastAPI application."""
    app = FastAPI(
        title="Nadir Gateway",
        description="OpenAI-compatible proxy for local MLX inference instances.",
        version="0.1.0",
    )

    @app.get("/health")
    def gateway_health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "nadir-gateway",
            "docs": "/docs",
            "models": "/v1/models",
        }

    app.include_router(models_router, dependencies=[Depends(verify_gateway_api_key)])
    app.include_router(image_files_router, dependencies=[Depends(verify_gateway_api_key)])
    app.include_router(chat_router, dependencies=[Depends(verify_gateway_api_key)])
    app.include_router(modes_router, dependencies=[Depends(verify_gateway_api_key)])
    return app
