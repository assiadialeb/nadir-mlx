"""Shared helpers for gateway contract tests."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from jsonschema.exceptions import ValidationError

from orchestrator.tests.contracts.validators import validate_response_body


def mock_buffered_upstream(
    mock_client_cls: MagicMock,
    *,
    status_code: int = 200,
    json_body: dict[str, Any] | None = None,
    content: bytes | None = None,
    content_type: str = "application/json",
) -> AsyncMock:
    """Attach a mocked httpx.AsyncClient returning a buffered upstream response."""
    upstream = MagicMock()
    upstream.status_code = status_code
    upstream.headers = httpx.Headers({"content-type": content_type})
    if json_body is not None:
        upstream.json.return_value = json_body
    if content is not None:
        upstream.content = content
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=upstream)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client
    return mock_client


def assert_contract_json_response(
    spec: dict[str, Any],
    *,
    path: str,
    method: str,
    status_code: int,
    response: Any,
) -> dict[str, Any]:
    """Assert HTTP status and validate JSON body against the curated OpenAPI contract."""
    assert response.status_code == status_code
    body = response.json()
    try:
        validate_response_body(
            spec,
            path=path,
            method=method,
            status_code=status_code,
            body=body,
        )
    except ValidationError as exc:
        pytest.fail(str(exc))
    return body


def parse_sse_json_payloads(content: bytes) -> list[dict[str, Any]]:
    """Extract JSON objects from SSE `data:` lines (excluding `[DONE]`)."""
    payloads: list[dict[str, Any]] = []
    for raw_line in content.splitlines():
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        payloads.append(json.loads(data))
    return payloads
