"""FastAPI application for the Nadir Gateway worker."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from django.conf import settings

from orchestrator.gateway.auth import verify_gateway_api_key
from orchestrator.gateway.errors import openai_error_envelope
from orchestrator.gateway.http_client import gateway_lifespan
from orchestrator.gateway.middleware import install_gateway_middleware, parse_cors_origins
from orchestrator.gateway.router import GatewayRouteError
from orchestrator.gateway.routes.anthropic import router as anthropic_router
from orchestrator.gateway.routes.chat import router as chat_router
from orchestrator.gateway.routes.image_files import router as image_files_router
from orchestrator.gateway.routes.models import router as models_router
from orchestrator.gateway.routes.modes import router as modes_router
from orchestrator.gateway.routes.ollama import public_router as ollama_public_router
from orchestrator.gateway.routes.ollama import router as ollama_router
from orchestrator.gateway.routes.responses import router as responses_router


def _validation_param(errors: list[dict]) -> str | None:
    if not errors:
        return None
    loc = errors[0].get("loc") or ()
    parts = [str(part) for part in loc if part not in {"body", "query", "path", "header"}]
    return ".".join(parts) if parts else None


def register_exception_handlers(app: FastAPI) -> None:
    """Map framework exceptions to OpenAI-compatible JSON errors."""

    @app.exception_handler(GatewayRouteError)
    async def handle_gateway_route_error(
        request: Request,
        exc: GatewayRouteError,
    ) -> JSONResponse:
        if request.url.path.startswith("/v1/messages"):
            from orchestrator.gateway.protocol.anthropic_translate import (
                anthropic_error_envelope,
                anthropic_error_type_for_code,
            )

            return JSONResponse(
                status_code=exc.status_code,
                content=anthropic_error_envelope(
                    exc.message,
                    anthropic_error_type_for_code(exc.code),
                ),
            )
        return JSONResponse(status_code=exc.status_code, content=exc.to_openai_error())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        errors = exc.errors()
        first = errors[0] if errors else {}
        message = str(first.get("msg") or "Invalid request.")
        return JSONResponse(
            status_code=400,
            content=openai_error_envelope(
                message,
                "invalid_request",
                param=_validation_param(errors),
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        _request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        if exc.status_code == 401:
            code = "invalid_api_key"
        elif exc.status_code == 404:
            code = "invalid_request"
        elif exc.status_code >= 500:
            code = "gateway_error"
        else:
            code = "invalid_request"
        detail = exc.detail
        message = detail if isinstance(detail, str) else "Request failed."
        return JSONResponse(
            status_code=exc.status_code,
            content=openai_error_envelope(message, code),
        )

    # FastAPI's HTTPException subclasses Starlette's; keep an explicit alias handler.
    app.add_exception_handler(HTTPException, handle_http_exception)


def create_app() -> FastAPI:
    """Build the gateway FastAPI application."""
    app = FastAPI(
        title="Nadir Gateway",
        description="OpenAI-compatible proxy for local MLX inference instances.",
        version="0.1.0",
        lifespan=gateway_lifespan,
    )
    install_gateway_middleware(
        app,
        cors_origins=parse_cors_origins(getattr(settings, "NADIR_GATEWAY_CORS_ORIGINS", "")),
    )
    register_exception_handlers(app)

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
    app.include_router(responses_router, dependencies=[Depends(verify_gateway_api_key)])
    app.include_router(anthropic_router, dependencies=[Depends(verify_gateway_api_key)])
    app.include_router(ollama_public_router)
    app.include_router(ollama_router, dependencies=[Depends(verify_gateway_api_key)])
    return app
