"""Shared httpx client lifecycle for the Nadir Gateway."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from django.conf import settings
from fastapi import FastAPI

from orchestrator.env_utils import env_float

_shared_client: httpx.AsyncClient | None = None


def _client_timeout() -> httpx.Timeout:
    seconds = env_float(
        "NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS",
        float(settings.NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS),
    )
    return httpx.Timeout(seconds)


def get_http_client() -> httpx.AsyncClient:
    """Return the process-wide client, creating one on demand for tests."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(timeout=_client_timeout())
    return _shared_client


async def startup_http_client() -> None:
    """Create the shared upstream HTTP client."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(timeout=_client_timeout())


async def shutdown_http_client() -> None:
    """Close the shared upstream HTTP client."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
    _shared_client = None


@asynccontextmanager
async def gateway_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan hook that owns the shared httpx client."""
    await startup_http_client()
    try:
        yield
    finally:
        await shutdown_http_client()
