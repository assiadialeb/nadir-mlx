"""Live gateway smoke tests (MLX-47, MLX-60, MLX-63).

Run against a running Nadir gateway on the host (not CI by default):

    export NADIR_SMOKE_GATEWAY_URL=http://127.0.0.1:11380
    export NADIR_SMOKE_MODEL_ALIAS=<running-text-alias>       # chat smoke
    export NADIR_SMOKE_ON_DEMAND_ALIAS=<on-demand-text-alias> # wake smoke (MLX-60)
    export NADIR_SMOKE_EMBED_ALIAS=<embedding-alias>          # embeddings smoke (MLX-63)
    export NADIR_SMOKE_RERANK_ALIAS=<reranker-alias>        # rerank smoke (MLX-63)
    export NADIR_SMOKE_MTP_ALIAS=<multimodal-mtp-alias>     # MTP generation smoke (MLX-70)
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


@pytest.mark.smoke
def test_smoke_chat_wakes_on_demand_instance() -> None:
    """MLX-60: first chat request should wake a STOPPED on_demand alias."""
    base_url = _smoke_base_url()
    model_alias = os.environ.get("NADIR_SMOKE_ON_DEMAND_ALIAS", "").strip()
    if not base_url or not model_alias:
        pytest.skip("Set NADIR_SMOKE_GATEWAY_URL and NADIR_SMOKE_ON_DEMAND_ALIAS.")

    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_alias,
            "messages": [{"role": "user", "content": "Reply with the word ok."}],
            "max_tokens": 16,
        },
        timeout=httpx.Timeout(600.0, connect=10.0),
    )
    response.raise_for_status()
    payload = response.json()
    assert payload.get("object") == "chat.completion"
    assert payload.get("choices")


@pytest.mark.smoke
def test_smoke_embeddings_non_stream() -> None:
    """MLX-63: live embeddings against a configured EMBEDDING alias."""
    base_url = _smoke_base_url()
    model_alias = os.environ.get("NADIR_SMOKE_EMBED_ALIAS", "").strip()
    if not base_url or not model_alias:
        pytest.skip("Set NADIR_SMOKE_GATEWAY_URL and NADIR_SMOKE_EMBED_ALIAS.")

    response = httpx.post(
        f"{base_url}/v1/embeddings",
        json={"model": model_alias, "input": "smoke test embedding"},
        timeout=httpx.Timeout(120.0, connect=10.0),
    )
    response.raise_for_status()
    payload = response.json()
    assert payload.get("object") == "list"
    assert isinstance(payload.get("data"), list)
    assert payload["data"]


@pytest.mark.smoke
def test_smoke_rerank_non_stream() -> None:
    """MLX-63: live rerank against a configured RERANKER alias."""
    base_url = _smoke_base_url()
    model_alias = os.environ.get("NADIR_SMOKE_RERANK_ALIAS", "").strip()
    if not base_url or not model_alias:
        pytest.skip("Set NADIR_SMOKE_GATEWAY_URL and NADIR_SMOKE_RERANK_ALIAS.")

    response = httpx.post(
        f"{base_url}/v1/rerank",
        json={
            "model": model_alias,
            "query": "python programming",
            "documents": ["Python is a language.", "Java runs on JVM."],
        },
        timeout=httpx.Timeout(120.0, connect=10.0),
    )
    response.raise_for_status()
    payload = response.json()
    assert isinstance(payload.get("results"), list)
    assert payload["results"]


@pytest.mark.smoke
def test_smoke_multimodal_mtp_generation() -> None:
    """MLX-70: live chat on a MULTIMODAL alias configured with MTP draft."""
    base_url = _smoke_base_url()
    model_alias = os.environ.get("NADIR_SMOKE_MTP_ALIAS", "").strip()
    if not base_url or not model_alias:
        pytest.skip("Set NADIR_SMOKE_GATEWAY_URL and NADIR_SMOKE_MTP_ALIAS.")

    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_alias,
            "messages": [{"role": "user", "content": "Reply with the word ok."}],
            "max_tokens": 16,
        },
        timeout=httpx.Timeout(600.0, connect=10.0),
    )
    response.raise_for_status()
    payload = response.json()
    assert payload.get("object") == "chat.completion"
    assert payload.get("choices")
