"""Tests for thinking / reasoning → mlx chat_template_kwargs (MLX-124)."""

from __future__ import annotations

from orchestrator.gateway.chat_extensions import prepare_chat_upstream_body
from orchestrator.gateway.thinking_control import (
    kwargs_from_anthropic,
    kwargs_from_ollama,
    kwargs_from_openai_chat,
    kwargs_from_reasoning_effort,
    kwargs_from_responses,
)


def test_reasoning_effort_disables_for_none_and_minimal() -> None:
    assert kwargs_from_reasoning_effort("none") == {"enable_thinking": False}
    assert kwargs_from_reasoning_effort("minimal") == {"enable_thinking": False}


def test_reasoning_effort_enables_with_budget() -> None:
    assert kwargs_from_reasoning_effort("low") == {
        "enable_thinking": True,
        "thinking_budget": 1024,
    }
    assert kwargs_from_reasoning_effort("high")["enable_thinking"] is True


def test_openai_chat_maps_reasoning_effort() -> None:
    assert kwargs_from_openai_chat({"reasoning_effort": "none"}) == {
        "enable_thinking": False
    }
    assert kwargs_from_openai_chat({}) == {}


def test_responses_maps_reasoning_effort_object() -> None:
    assert kwargs_from_responses({"reasoning": {"effort": "minimal"}}) == {
        "enable_thinking": False
    }
    assert kwargs_from_responses({"reasoning_effort": "medium"})["enable_thinking"] is True


def test_anthropic_absent_or_disabled_turns_thinking_off() -> None:
    assert kwargs_from_anthropic({}) == {"enable_thinking": False}
    assert kwargs_from_anthropic({"thinking": {"type": "disabled"}}) == {
        "enable_thinking": False
    }


def test_anthropic_enabled_with_budget() -> None:
    assert kwargs_from_anthropic(
        {"thinking": {"type": "enabled", "budget_tokens": 2048}}
    ) == {"enable_thinking": True, "thinking_budget": 2048}


def test_ollama_think_explicit_only() -> None:
    assert kwargs_from_ollama({}) == {}
    assert kwargs_from_ollama({"think": False}) == {"enable_thinking": False}
    assert kwargs_from_ollama({"think": True}) == {"enable_thinking": True}


def test_prepare_chat_upstream_merges_kwargs_and_strips_effort() -> None:
    body = {
        "model": "client-alias",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "none",
        "chat_template_kwargs": {"foo": 1},
    }
    upstream = prepare_chat_upstream_body(
        body,
        "upstream-model",
        chat_template_kwargs={"enable_thinking": False},
    )
    assert upstream["model"] == "upstream-model"
    assert "reasoning_effort" not in upstream
    assert upstream["chat_template_kwargs"] == {"foo": 1, "enable_thinking": False}
