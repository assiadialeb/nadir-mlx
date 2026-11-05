#!/usr/bin/env python3
"""Manual replay: Claude Code-shaped Anthropic turns to validate prefix KV cache.

Sends three ``POST /v1/messages`` requests with a stable system prompt + tools
and a growing tool-use history. Prints ``usage.cache_read_input_tokens`` when
the gateway maps mlx-lm ``prompt_tokens_details.cached_tokens`` (MLX-134).

Environment:
  NADIR_GATEWAY_BASE_URL  default http://127.0.0.1:11380
  NADIR_GATEWAY_API_KEY   optional Bearer / X-API-Key when gateway requires auth
  NADIR_GATEWAY_MODEL     gateway alias (required)

Example:
  export NADIR_GATEWAY_MODEL=my-text-alias
  python scripts/gateway_anthropic_cache_replay.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import httpx

_SYSTEM_PROMPT = (
    "You are a coding agent on Apple Silicon. Follow project conventions. "
    "Use tools when needed. Keep responses concise. "
    + ("Context block. " * 120)
)

_TOOLS: list[dict[str, Any]] = [
    {
        "name": f"tool_{index}",
        "description": f"Tool number {index} for filesystem or search operations.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
    for index in range(10)
]


def _gateway_config() -> tuple[str, str | None, str]:
    base = os.environ.get("NADIR_GATEWAY_BASE_URL", "http://127.0.0.1:11380").rstrip("/")
    api_key = os.environ.get("NADIR_GATEWAY_API_KEY") or None
    model = os.environ.get("NADIR_GATEWAY_MODEL", "").strip()
    if not model:
        print("Set NADIR_GATEWAY_MODEL to a gateway alias.", file=sys.stderr)
        sys.exit(1)
    return base, api_key, model


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _turn_messages(turn: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Inspect the repo and summarize structure."},
    ]
    for step in range(1, turn):
        tool_id = f"toolu_replay_{step}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": "tool_0",
                            "input": {"path": f"/tmp/step-{step}"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": json.dumps({"lines": 42 + step, "ok": True}),
                        }
                    ],
                },
            ]
        )
        messages.append(
            {"role": "user", "content": f"Continue analysis (turn {step + 1})."},
        )
    return messages


def _post_message(
    client: httpx.Client,
    *,
    url: str,
    model: str,
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], float]:
    body = {
        "model": model,
        "max_tokens": 64,
        "system": _SYSTEM_PROMPT,
        "tools": _TOOLS,
        "messages": messages,
    }
    started = time.perf_counter()
    response = client.post(url, json=body)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Expected JSON object from gateway")
    return payload, elapsed


def _print_turn_summary(turn: int, payload: dict[str, Any], elapsed: float) -> None:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    cache_read = usage.get("cache_read_input_tokens", 0)
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    prompt_total = int(input_tokens) + int(cache_read or 0)
    rate = (100.0 * float(cache_read) / prompt_total) if prompt_total else 0.0
    print(
        f"Turn {turn}: prompt≈{prompt_total} cached={cache_read} "
        f"({rate:.1f}%) input={input_tokens} out={output_tokens} "
        f"elapsed={elapsed:.1f}s"
    )


def main() -> None:
    base, api_key, model = _gateway_config()
    url = f"{base}/v1/messages"
    headers = _headers(api_key)
    with httpx.Client(headers=headers, timeout=600.0) as client:
        for turn in range(1, 4):
            messages = _turn_messages(turn)
            payload, elapsed = _post_message(
                client,
                url=url,
                model=model,
                messages=messages,
            )
            _print_turn_summary(turn, payload, elapsed)


if __name__ == "__main__":
    main()
