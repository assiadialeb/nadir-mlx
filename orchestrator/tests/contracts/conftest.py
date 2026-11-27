"""Shared fixtures for OpenAPI contract tests."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from django.db.backends.signals import connection_created
from fastapi.testclient import TestClient

from orchestrator.gateway.app import create_app
from orchestrator.gateway.route_cache import clear_gateway_route_cache
from orchestrator.tests.contracts.validators import load_curated_spec


@pytest.fixture
def curated_spec() -> dict[str, Any]:
    return load_curated_spec()


def _close_raw_connection(wrapper: Any) -> None:
    """Close a Django connection opened by another thread (MLX-137).

    ``DatabaseWrapper.close()`` is a no-op on an in-memory SQLite test database
    and refuses connections from another thread, so close the DB-API connection
    directly. Django opens SQLite connections with ``check_same_thread=False``.
    """
    raw = wrapper.connection
    if raw is None:
        return
    raw.close()
    wrapper.connection = None


@contextmanager
def gateway_test_client(opened: list[Any] | None = None) -> Iterator[TestClient]:
    """Gateway test client that leaves no database connection behind (MLX-137).

    The gateway reads the database through ``sync_to_async(thread_sensitive=False)``,
    so each executor thread opens its own Django connection that nothing closes.
    On the shared in-memory SQLite database such a connection can still hold a
    table lock when ``django_db(transaction=True)`` flushes at teardown.

    ``opened`` collects the connections created outside the calling thread, so a
    test can check that they are closed on exit.
    """
    clear_gateway_route_cache()
    caller_thread = threading.get_ident()
    tracked: list[Any] = opened if opened is not None else []

    def _track(sender: Any, connection: Any, **_kwargs: Any) -> None:
        if threading.get_ident() != caller_thread:
            tracked.append(connection)

    connection_created.connect(_track, weak=False)
    try:
        # Not entered as a context manager on purpose: the lifespan would build
        # the shared httpx client before the tests patch ``httpx.AsyncClient``.
        yield TestClient(create_app())
    finally:
        connection_created.disconnect(_track)
        for wrapper in tracked:
            _close_raw_connection(wrapper)


@pytest.fixture
def gateway_client() -> Iterator[TestClient]:
    with gateway_test_client() as client:
        yield client
