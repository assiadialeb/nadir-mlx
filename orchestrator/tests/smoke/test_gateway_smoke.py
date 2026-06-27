"""Live gateway smoke tests (MLX-47).

Run against a running Nadir gateway on the host (not CI by default):

    export NADIR_SMOKE_GATEWAY_URL=http://127.0.0.1:11380
    export NADIR_SMOKE_MODEL_ALIAS=<running-text-alias>  # optional for chat smoke
    pytest -m smoke orchestrator/tests/smoke -q
"""

from __future__ import annotations

import os

import httpx
import pytest


def _smoke_base_url() -> str | None:
    raw = os.environ.get("NADIR_SMOKE_GATEWAY_URL", "").strip().rstrip("/")
    return raw or None


@pytest.mark.smoke
def test_smoke_health_endpoint() -> None:
    base_url = _smoke_base_url()
    if not base_url:
        pytest.skip("Set NADIR_SMOKE_GATEWAY_URL to run smoke tests.")

    response = httpx.get(f"{base_url}/health", timeout=10.0)
    response.raise_for_status()
    payload = response.json()
    assert payload.get("status") == "ok"
    assert payload.get("service") == "nadir-gateway"


@pytest.mark.smoke
def test_smoke_list_models() -> None:
    base_url = _smoke_base_url()
    if not base_url:
        pytest.skip("Set NADIR_SMOKE_GATEWAY_URL to run smoke tests.")

    response = httpx.get(f"{base_url}/v1/models", timeout=30.0)
    response.raise_for_status()
    payload = response.json()
    assert payload.get("object") == "list"
    assert isinstance(payload.get("data"), list)


@pytest.mark.smoke
def test_smoke_chat_completion_non_stream() -> None:
    base_url = _smoke_base_url()
    model_alias = os.environ.get("NADIR_SMOKE_MODEL_ALIAS", "").strip()
    if not base_url or not model_alias:
        pytest.skip("Set NADIR_SMOKE_GATEWAY_URL and NADIR_SMOKE_MODEL_ALIAS.")

    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_alias,
            "messages": [{"role": "user", "content": "Reply with the word ok."}],
            "max_tokens": 16,
        },
        timeout=httpx.Timeout(300.0, connect=10.0),
    )
    response.raise_for_status()
    payload = response.json()
    assert payload.get("object") == "chat.completion"
    assert payload.get("choices")
