"""
ASGI config for mlx_orchestrator (Django + optional integrated MCP HTTP endpoint).

Use with::

    uvicorn mlx_orchestrator.asgi:application --host 127.0.0.1 --port 8000

``python manage.py runserver`` remains WSGI; prefer ``manage.py run_mcp`` for a
dedicated MCP port, or uvicorn for a combined control plane + MCP mount.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mlx_orchestrator.settings")

django_application = get_asgi_application()


@asynccontextmanager
async def _mcp_lifespan() -> AsyncIterator[None]:
    """Start MCP session manager only when Django ASGI mounts ``/mcp``."""
    from orchestrator.mcp.server import mcp_django_mount_enabled, mcp_session_lifespan

    if not mcp_django_mount_enabled():
        yield
        return
    async with mcp_session_lifespan():
        yield


class _LifespanMiddleware:
    """Run MCP lifespan hooks on ASGI lifespan events."""

    def __init__(self, app: Callable[..., Any]) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "lifespan":
            await self.app(scope, receive, send)
            return

        async with _mcp_lifespan():
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return


async def _route_application(scope: dict[str, Any], receive: Any, send: Any) -> None:
    """Route MCP Streamable HTTP traffic before the Django ASGI stack."""
    if scope.get("type") == "http":
        from orchestrator.mcp.server import (
            get_mcp_http_application,
            mcp_django_mount_enabled,
            mcp_http_mount_path,
        )

        if mcp_django_mount_enabled():
            path = scope.get("path", "")
            prefix = mcp_http_mount_path()
            if path == prefix or path.startswith(f"{prefix}/"):
                await get_mcp_http_application()(scope, receive, send)
                return
    await django_application(scope, receive, send)


application = _LifespanMiddleware(_route_application)
