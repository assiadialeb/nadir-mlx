"""Tests for Anthropic Messages translation helpers."""

from django.test import SimpleTestCase

from orchestrator.gateway.protocol.anthropic_translate import (
    anthropic_error_envelope,
    anthropic_error_type_for_code,
    anthropic_request_to_chat,
    chat_completion_to_anthropic,
    chat_sse_delta_text,
    chat_sse_finish_reason,
    estimate_anthropic_token_count,
    openai_finish_to_anthropic_stop,
)
from orchestrator.gateway.router import GatewayRouteError


class AnthropicTranslateTests(SimpleTestCase):
    def test_anthropic_request_to_chat(self) -> None:
        chat = anthropic_request_to_chat(
            {
                "model": "demo",
                "max_tokens": 64,
                "system": "Be kind",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.4,
                "top_p": 0.9,
                "stop_sequences": ["END"],
            }
        )
        self.assertEqual(
            chat["messages"],
            [
                {"role": "system", "content": "Be kind"},
                {"role": "user", "content": "Hi"},
            ],
        )
        self.assertEqual(chat["max_tokens"], 64)
        self.assertEqual(chat["temperature"], 0.4)
        self.assertEqual(chat["top_p"], 0.9)
        self.assertEqual(chat["stop"], ["END"])

    def test_system_blocks_and_content_blocks(self) -> None:
        chat = anthropic_request_to_chat(
            {
                "model": "demo",
                "max_tokens": 10,
                "system": [{"type": "text", "text": "Sys"}, "extra"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Look"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": "abc",
                                },
                            },
                        ],
                    },
                    "skip",
                ],
                "stream": True,
            }
        )
        self.assertEqual(chat["messages"][0]["content"], "Sysextra")
        content = chat["messages"][1]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertIn("image/jpeg", content[1]["image_url"]["url"])
        self.assertTrue(chat["stream"])

    def test_tools_and_tool_choice_round_trip(self) -> None:
        chat = anthropic_request_to_chat(
            {
                "model": "demo",
                "max_tokens": 32,
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Weather lookup",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
                "tool_choice": "any",
                "messages": [
                    {"role": "user", "content": "Paris?"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "need tool"},
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "get_weather",
                                "input": {"city": "Paris"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "sunny",
                            }
                        ],
                    },
                ],
            }
        )
        self.assertEqual(chat["tool_choice"], "required")
        self.assertEqual(chat["tools"][0]["function"]["name"], "get_weather")
        assistant = chat["messages"][1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["reasoning_content"], "need tool")
        self.assertEqual(assistant["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(assistant["tool_calls"][0]["function"]["arguments"], '{"city":"Paris"}')
        tool_msg = chat["messages"][2]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], "toolu_1")
        self.assertEqual(tool_msg["content"], "sunny")

    def test_tool_choice_named_tool(self) -> None:
        chat = anthropic_request_to_chat(
            {
                "model": "demo",
                "max_tokens": 8,
                "tool_choice": {"type": "tool", "name": "lookup"},
                "messages": [{"role": "user", "content": "x"}],
            }
        )
        self.assertEqual(
            chat["tool_choice"],
            {"type": "function", "function": {"name": "lookup"}},
        )

    def test_missing_model_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            anthropic_request_to_chat({"max_tokens": 1, "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(ctx.exception.code, "invalid_model")

    def test_missing_max_tokens_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            anthropic_request_to_chat({"model": "demo", "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(ctx.exception.param, "max_tokens")

    def test_empty_messages_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            anthropic_request_to_chat({"model": "demo", "max_tokens": 1, "messages": []})
        self.assertEqual(ctx.exception.param, "messages")

    def test_chat_to_anthropic_response(self) -> None:
        payload = chat_completion_to_anthropic(
            {
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
            client_model_alias="demo",
        )
        self.assertEqual(payload["type"], "message")
        self.assertEqual(payload["content"][0]["text"], "Hello")
        self.assertEqual(payload["model"], "demo")
        self.assertEqual(payload["stop_reason"], "end_turn")

    def test_chat_to_anthropic_thinking_and_tool_use(self) -> None:
        payload = chat_completion_to_anthropic(
            {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "plan",
                            "content": "done",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"q":"a"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            },
            client_model_alias="demo",
        )
        self.assertEqual(payload["stop_reason"], "tool_use")
        self.assertEqual(payload["content"][0], {"type": "thinking", "thinking": "plan"})
        self.assertEqual(payload["content"][1], {"type": "text", "text": "done"})
        self.assertEqual(payload["content"][2]["type"], "tool_use")
        self.assertEqual(payload["content"][2]["name"], "lookup")
        self.assertEqual(payload["content"][2]["input"], {"q": "a"})

    def test_chat_to_anthropic_max_tokens_stop(self) -> None:
        payload = chat_completion_to_anthropic(
            {"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]},
            client_model_alias="demo",
        )
        self.assertEqual(payload["stop_reason"], "max_tokens")

    def test_estimate_anthropic_token_count(self) -> None:
        result = estimate_anthropic_token_count(
            {
                "model": "demo",
                "messages": [{"role": "user", "content": "abcd" * 10}],
            }
        )
        self.assertEqual(result["model"], "demo")
        self.assertGreater(result["input_tokens"], 0)

    def test_anthropic_error_helpers(self) -> None:
        envelope = anthropic_error_envelope("boom", "authentication_error")
        self.assertEqual(envelope["type"], "error")
        self.assertEqual(envelope["error"]["type"], "authentication_error")
        self.assertEqual(anthropic_error_type_for_code("invalid_api_key"), "authentication_error")
        self.assertEqual(anthropic_error_type_for_code("model_not_found"), "not_found_error")

    def test_chat_sse_delta_and_finish_helpers(self) -> None:
        line = b'data: {"choices":[{"delta":{"content":"ab"},"finish_reason":null}]}'
        self.assertEqual(chat_sse_delta_text(line), "ab")
        self.assertIsNone(chat_sse_finish_reason(line))
        done = b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}'
        self.assertIsNone(chat_sse_delta_text(done))
        self.assertEqual(chat_sse_finish_reason(done), "length")
        self.assertEqual(openai_finish_to_anthropic_stop("length"), "max_tokens")
        self.assertEqual(openai_finish_to_anthropic_stop("tool_calls"), "tool_use")
        self.assertEqual(openai_finish_to_anthropic_stop("stop"), "end_turn")
