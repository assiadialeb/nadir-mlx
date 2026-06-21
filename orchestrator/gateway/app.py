"""FastAPI application for the Nadir Gateway worker."""

from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Build the gateway FastAPI application."""
    app = FastAPI(
        title="Nadir Gateway",
        description="OpenAI-compatible proxy for local MLX inference instances.",
        version="0.1.0",
    )

    @app.get("/health")
    def gateway_health() -> dict[str, str]:
        return {"status": "ok", "service": "nadir-gateway"}

    return app
