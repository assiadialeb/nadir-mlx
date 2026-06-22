"""FastAPI application for the Nadir Gateway worker."""

from __future__ import annotations

from fastapi import FastAPI

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

    app.include_router(models_router)
    app.include_router(image_files_router)
    app.include_router(chat_router)
    app.include_router(modes_router)
    return app
