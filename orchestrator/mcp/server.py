"""FastMCP server factory for Nadir (Streamable HTTP / SSE-compatible clients)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.server.transport_security import TransportSecuritySettings

from orchestrator.mcp.auth_middleware import NadirMcpAuthMiddleware
from orchestrator.mcp.tools import register_tools

MCP_HTTP_PATH = os.environ.get("NADIR_MCP_PATH", "/mcp").strip() or "/mcp"


def _mcp_transport_security() -> TransportSecuritySettings:
    """DNS-rebinding protection for non-loopback MCP binds."""
    raw_hosts = os.environ.get("NADIR_MCP_ALLOWED_HOSTS", "").strip()
    if not raw_hosts:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)

    allowed_hosts: list[str] = []
    for item in raw_hosts.split(","):
        host = item.strip()
        if not host:
            continue
        allowed_hosts.append(host if ":" in host else f"{host}:*")
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
    )


mcp = FastMCP(
    "nadir-mlx",
    stateless_http=True,
    streamable_http_path=MCP_HTTP_PATH,
    transport_security=_mcp_transport_security(),
)
register_tools(mcp)

_mcp_http_application: Any | None = None
_session_manager_ready = False


def mcp_http_enabled() -> bool:
    """Return True when the integrated HTTP MCP endpoint should be mounted."""
    return os.environ.get("NADIR_MCP_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def mcp_http_mount_path() -> str:
    """Return the URL mount path for the MCP ASGI sub-application."""
    normalized = MCP_HTTP_PATH if MCP_HTTP_PATH.startswith("/") else f"/{MCP_HTTP_PATH}"
    return normalized.rstrip("/") or "/mcp"


def ensure_mcp_session_manager() -> None:
    """Initialize the Streamable HTTP session manager (idempotent)."""
    global _session_manager_ready
    if _session_manager_ready:
        return
    mcp.streamable_http_app()
    _session_manager_ready = True


@asynccontextmanager
async def mcp_session_lifespan() -> AsyncIterator[None]:
    """Keep the MCP session manager task group alive for the ASGI process."""
    if not mcp_http_enabled():
        yield
        return
    ensure_mcp_session_manager()
    async with mcp.session_manager.run():
        yield


def get_mcp_http_application() -> Any:
    """Return the authenticated Streamable HTTP ASGI handler."""
    global _mcp_http_application
    if _mcp_http_application is None:
        ensure_mcp_session_manager()
        handler = StreamableHTTPASGIApp(mcp.session_manager)
        _mcp_http_application = NadirMcpAuthMiddleware(handler)
    return _mcp_http_application
