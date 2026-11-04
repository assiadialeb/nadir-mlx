"""Start the Nadir Streamable HTTP MCP server (instance control for IDEs)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

import uvicorn
from django.core.management.base import BaseCommand


def _build_asgi_app() -> Callable[..., Any]:
    """Build an ASGI app that only serves the MCP endpoint with lifespan."""
    from orchestrator.mcp.server import (
        get_mcp_http_application,
        mcp_http_mount_path,
        mcp_session_lifespan,
    )

    @asynccontextmanager
    async def _lifespan(_app: Any) -> AsyncIterator[None]:
        async with mcp_session_lifespan():
            yield

    class _LifespanMiddleware:
        def __init__(self, app: Callable[..., Any]) -> None:
            self.app = app

        async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
            if scope["type"] != "lifespan":
                await self.app(scope, receive, send)
                return
            async with _lifespan(None):
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        await send({"type": "lifespan.shutdown.complete"})
                        return

    async def _route(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            return
        path = scope.get("path", "")
        prefix = mcp_http_mount_path()
        if path == prefix or path.startswith(f"{prefix}/"):
            await get_mcp_http_application()(scope, receive, send)
            return
        body = b'{"error":"Not Found","hint":"MCP is mounted at /mcp"}'
        await send(
            {
                "type": "http.response.start",
                "status": 404,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return _LifespanMiddleware(_route)


class Command(BaseCommand):
    help = "Start the Nadir MCP Streamable HTTP server (list/start/stop instances)."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--host",
            default=os.environ.get("NADIR_MCP_HOST", "127.0.0.1"),
            help="Bind address (default NADIR_MCP_HOST or 127.0.0.1).",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=int(os.environ.get("NADIR_MCP_PORT", "11381")),
            help="Bind port (default NADIR_MCP_PORT or 11381).",
        )

    def handle(self, *args, **options) -> None:
        host = str(options["host"]).strip() or "127.0.0.1"
        port = int(options["port"])
        from orchestrator.mcp.auth_middleware import configured_mcp_api_key
        from orchestrator.security_utils import is_loopback_bind_host

        if not is_loopback_bind_host(host) and not configured_mcp_api_key():
            raise SystemExit(
                "Refusing non-loopback MCP bind without NADIR_MCP_API_KEY "
                "(or NADIR_GATEWAY_API_KEY)."
            )

        app = _build_asgi_app()
        self.stdout.write(
            self.style.SUCCESS(f"Starting Nadir MCP on http://{host}:{port}/mcp …")
        )
        uvicorn.run(app, host=host, port=port, log_level="info")
