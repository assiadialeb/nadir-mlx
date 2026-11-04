"""Map client thinking / reasoning controls to mlx-lm ``chat_template_kwargs``."""

from __future__ import annotations

from typing import Any

# Heuristic budgets for templates that honor ``thinking_budget`` (e.g. some Qwen builds).
_EFFORT_THINKING_BUDGET: dict[str, int] = {
    "low": 1024,
    "medium": 8192,
    "high": 24576,
    "xhigh": 49152,
}
_DISABLE_EFFORTS = frozenset(
    {"", "none", "minimal", "disable", "disabled", "off", "false", "0"}
)
_ENABLE_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def merge_chat_template_kwargs(
    body: dict[str, Any],
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a copy of ``body`` with ``chat_template_kwargs`` merged (extra wins)."""
    if not extra:
        return body
    merged = dict(body)
    existing = merged.get("chat_template_kwargs")
    base = dict(existing) if isinstance(existing, dict) else {}
    base.update(extra)
    merged["chat_template_kwargs"] = base
    return merged


def kwargs_from_reasoning_effort(effort: Any) -> dict[str, Any]:
    """Map OpenAI-style ``reasoning_effort`` to mlx chat template kwargs."""
    if effort is None:
        return {}
    if isinstance(effort, bool):
        return {"enable_thinking": effort}
    if not isinstance(effort, str):
        return {}
    normalized = effort.strip().lower()
    if normalized in _DISABLE_EFFORTS:
        return {"enable_thinking": False}
    if normalized in _ENABLE_EFFORTS:
        result: dict[str, Any] = {"enable_thinking": True}
        budget = _EFFORT_THINKING_BUDGET.get(normalized)
        if budget is not None:
            result["thinking_budget"] = budget
        return result
    return {}


def kwargs_from_openai_chat(body: dict[str, Any]) -> dict[str, Any]:
    """OpenAI chat completions: ``reasoning_effort`` (top-level)."""
    if "reasoning_effort" not in body:
        return {}
    return kwargs_from_reasoning_effort(body.get("reasoning_effort"))


def kwargs_from_responses(body: dict[str, Any]) -> dict[str, Any]:
    """OpenAI Responses: ``reasoning.effort`` or top-level ``reasoning_effort``."""
    if "reasoning_effort" in body:
        return kwargs_from_reasoning_effort(body.get("reasoning_effort"))
    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict):
        return {}
    if "effort" not in reasoning:
        return {}
    return kwargs_from_reasoning_effort(reasoning.get("effort"))


def kwargs_from_anthropic(body: dict[str, Any]) -> dict[str, Any]:
    """Anthropic Messages: ``thinking.type`` enabled/disabled/absent."""
    thinking = body.get("thinking")
    if thinking is None:
        return {"enable_thinking": False}
    if not isinstance(thinking, dict):
        return {"enable_thinking": False}
    thinking_type = str(thinking.get("type") or "").strip().lower()
    if thinking_type == "enabled":
        result: dict[str, Any] = {"enable_thinking": True}
        budget = thinking.get("budget_tokens")
        if isinstance(budget, int) and budget > 0:
            result["thinking_budget"] = budget
        return result
    # disabled / absent type / unknown → off
    return {"enable_thinking": False}


def kwargs_from_ollama(body: dict[str, Any]) -> dict[str, Any]:
    """Ollama: ``think`` boolean (only when explicitly set)."""
    if "think" not in body:
        return {}
    think = body.get("think")
    if think is True:
        return {"enable_thinking": True}
    if think is False:
        return {"enable_thinking": False}
    return {}
