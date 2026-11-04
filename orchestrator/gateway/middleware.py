"""ASGI middleware helpers for the Nadir Gateway."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "x-request-id"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Propagate or generate ``X-Request-Id`` on every response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = (request.headers.get(REQUEST_ID_HEADER) or "").strip()
        request_id = incoming or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


def parse_cors_origins(raw: str) -> list[str]:
    """Split a comma-separated origins string into a clean list."""
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def install_gateway_middleware(
    app: FastAPI,
    cors_origins: Sequence[str] | None = None,
) -> None:
    """Attach request-id (always) and optional CORS middleware."""
    app.add_middleware(RequestIdMiddleware)
    origins = list(cors_origins or [])
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[REQUEST_ID_HEADER],
        )
