"""ASGI middleware helpers for the Nadir Gateway."""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Sequence
from urllib.parse import urlparse

from django.conf import settings
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import Response

from orchestrator.env_utils import env_float, env_str
from orchestrator.gateway.errors import openai_error_envelope
from orchestrator.security_utils import is_loopback_bind_host

REQUEST_ID_HEADER = "x-request-id"
_DEFAULT_MAX_BODY_MB = 50.0


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Propagate or generate ``X-Request-Id`` on every response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = (request.headers.get(REQUEST_ID_HEADER) or "").strip()
        request_id = incoming or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class OriginGuardMiddleware(BaseHTTPMiddleware):
    """Reject browser requests whose Origin is not explicitly allowed.

    CLI/SDK clients typically omit Origin; those requests pass through.
    """

    def __init__(self, app, allowed_origins: Sequence[str] | None = None) -> None:
        super().__init__(app)
        self.allowed_origins = frozenset(allowed_origins or [])

    async def dispatch(self, request: Request, call_next) -> Response:
        origin = (request.headers.get("origin") or "").strip()
        if not origin:
            return await call_next(request)
        if origin in self.allowed_origins:
            return await call_next(request)
        return JSONResponse(
            status_code=403,
            content=openai_error_envelope(
                "Origin not allowed.",
                "forbidden",
            ),
        )


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject oversized request bodies early via Content-Length."""

    def __init__(self, app, max_body_bytes: int) -> None:
        super().__init__(app)
        self.max_body_bytes = max(1, int(max_body_bytes))

    async def dispatch(self, request: Request, call_next) -> Response:
        raw = request.headers.get("content-length")
        if raw:
            try:
                length = int(raw)
            except ValueError:
                length = -1
            if length > self.max_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content=openai_error_envelope(
                        "Request body too large.",
                        "payload_too_large",
                    ),
                )
        return await call_next(request)


def parse_cors_origins(raw: str) -> list[str]:
    """Split a comma-separated origins string into a clean list."""
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def cors_allow_credentials(origins: Sequence[str]) -> bool:
    """Whether CORS may set ``Access-Control-Allow-Credentials``.

    Always ``False`` for this gateway: clients authenticate with bearer /
    ``X-API-Key`` headers, not cookies. This also prevents the forbidden
    ``credentials=True`` + ``Allow-Origin: *`` combination.
    """
    _ = origins
    return False


def gateway_allowed_hosts() -> list[str]:
    """Hosts accepted by TrustedHostMiddleware (anti DNS rebinding)."""
    hosts: list[str] = ["localhost", "127.0.0.1", "[::1]"]
    # Django/Starlette TestClient uses Host: testserver — never in production.
    if (
        "pytest" in sys.modules
        or os.environ.get("PYTEST_CURRENT_TEST")
        or env_str("NADIR_TESTING", "").lower() in {"1", "true", "yes", "on"}
    ):
        hosts.append("testserver")
    extra = env_str(
        "NADIR_GATEWAY_ALLOWED_HOSTS",
        getattr(settings, "NADIR_GATEWAY_ALLOWED_HOSTS", ""),
    )
    hosts.extend(parse_cors_origins(extra))
    public = getattr(settings, "NADIR_GATEWAY_PUBLIC_BASE_URL", "") or ""
    hostname = urlparse(public).hostname if public else None
    if hostname:
        hosts.append(hostname)
    bind = env_str("NADIR_GATEWAY_HOST", getattr(settings, "NADIR_GATEWAY_HOST", "127.0.0.1"))
    if bind and bind not in {"0.0.0.0", "::"}:
        hosts.append(bind.strip("[]"))
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(hosts))


def gateway_max_body_bytes() -> int:
    mb = max(1.0, env_float("NADIR_GATEWAY_MAX_BODY_MB", _DEFAULT_MAX_BODY_MB))
    return int(mb * 1024 * 1024)


def expose_gateway_docs() -> bool:
    """Expose /docs and OpenAPI only on loopback binds."""
    bind = env_str("NADIR_GATEWAY_HOST", getattr(settings, "NADIR_GATEWAY_HOST", "127.0.0.1"))
    return is_loopback_bind_host(bind)


def install_gateway_middleware(
    app: FastAPI,
    cors_origins: Sequence[str] | None = None,
) -> None:
    """Attach security and observability middleware (order: last added = outermost)."""
    origins = list(cors_origins or [])
    # Wildcard cannot mix with explicit origins under credentialed CORS.
    if "*" in origins:
        origins = ["*"]

    # Last added runs first on the request path. CORS must be outermost so
    # preflight OPTIONS can short-circuit before TrustedHost / body limits.
    app.add_middleware(RequestIdMiddleware)
    if origins != ["*"]:
        app.add_middleware(OriginGuardMiddleware, allowed_origins=origins)
    app.add_middleware(MaxBodySizeMiddleware, max_body_bytes=gateway_max_body_bytes())
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=gateway_allowed_hosts())
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=cors_allow_credentials(origins),
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[REQUEST_ID_HEADER],
        )
