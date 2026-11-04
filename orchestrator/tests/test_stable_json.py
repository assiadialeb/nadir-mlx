"""Tests for byte-stable façade JSON (MLX-125 prompt-cache prefixes)."""

from __future__ import annotations

import json

from django.test import SimpleTestCase

from orchestrator.gateway.chat_extensions import prepare_chat_upstream_body
from orchestrator.gateway.protocol.anthropic_translate import anthropic_request_to_chat
from orchestrator.gateway.protocol.ollama_translate import ollama_chat_to_openai
from orchestrator.gateway.protocol.responses_translate import responses_request_to_chat
from orchestrator.gateway.protocol.stable_json import (
    canonicalize_json_value,
    stable_json_dumps,
    stabilize_chat_tools,
)


def _tools_fingerprint(chat: dict) -> str:
    return stable_json_dumps(chat.get("tools") or [])


class StableJsonHelperTests(SimpleTestCase):
    def test_canonicalize_sorts_object_keys(self) -> None:
        value = {"z": 1, "a": {"y": 2, "b": 3}}
        self.assertEqual(
            list(canonicalize_json_value(value).keys()),
            ["a", "z"],
        )
        self.assertEqual(
            list(canonicalize_json_value(value)["a"].keys()),
            ["b", "y"],
        )

    def test_stabilize_chat_tools_preserves_order(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "b", "parameters": {"z": 1, "a": 2}}},
            {"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}},
        ]
        out = stabilize_chat_tools(tools)
        self.assertEqual(out[0]["function"]["name"], "b")
        self.assertEqual(out[1]["function"]["name"], "a")
        # Schema property order must stay client-defined (not re-sorted).
        self.assertEqual(list(out[0]["function"]["parameters"].keys()), ["z", "a"])


class FacadePrefixStabilityTests(SimpleTestCase):
    """Two identical turns must produce identical tools prefixes after translation."""

    def test_anthropic_two_turns_identical_tools_prefix(self) -> None:
        body = {
            "model": "demo",
            "max_tokens": 32,
            "system": "You are helpful.",
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "messages": [{"role": "user", "content": "Paris?"}],
        }
        first = anthropic_request_to_chat(body)
        second = anthropic_request_to_chat(
            {
                **body,
                "messages": [
                    {"role": "user", "content": "Paris?"},
                    {"role": "assistant", "content": "Checking."},
                    {"role": "user", "content": "And Lyon?"},
                ],
            }
        )
        self.assertEqual(_tools_fingerprint(first), _tools_fingerprint(second))
        self.assertEqual(first["messages"][0], second["messages"][0])

    def test_anthropic_preserves_client_schema_key_order(self) -> None:
        schema = {
            "required": ["city"],
            "properties": {"city": {"type": "string"}},
            "type": "object",
        }
        chat = anthropic_request_to_chat(
            {
                "model": "demo",
                "max_tokens": 8,
                "tools": [{"name": "t", "input_schema": schema}],
                "messages": [{"role": "user", "content": "a"}],
            }
        )
        params = chat["tools"][0]["function"]["parameters"]
        self.assertEqual(list(params.keys()), ["required", "properties", "type"])

    def test_openai_prepare_preserves_tools_across_turns(self) -> None:
        body = {
            "model": "demo",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    },
                }
            ],
        }
        first = prepare_chat_upstream_body(body, "upstream")
        second = prepare_chat_upstream_body(body, "upstream")
        self.assertEqual(_tools_fingerprint(first), _tools_fingerprint(second))
        self.assertEqual(
            list(first["tools"][0]["function"]["parameters"].keys()),
            ["type", "properties"],
        )

    def test_responses_and_ollama_tools_stable(self) -> None:
        responses_body = {
            "model": "demo",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
                }
            ],
        }
        ollama_body = {
            "model": "demo",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"properties": {"id": {"type": "string"}}, "type": "object"},
                    },
                }
            ],
        }
        chat_r = responses_request_to_chat(responses_body)
        chat_o = ollama_chat_to_openai(ollama_body)
        self.assertEqual(
            chat_r["tools"][0]["function"]["name"],
            chat_o["tools"][0]["function"]["name"],
        )
        self.assertEqual(
            json.dumps(chat_r["tools"][0]["function"]["parameters"], sort_keys=True),
            json.dumps(chat_o["tools"][0]["function"]["parameters"], sort_keys=True),
        )
