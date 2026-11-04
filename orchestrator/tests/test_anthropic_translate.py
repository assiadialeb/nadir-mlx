"""Tests for Anthropic Messages translation helpers."""

from django.test import SimpleTestCase

from orchestrator.gateway.protocol.anthropic_translate import (
    anthropic_request_to_chat,
    chat_completion_to_anthropic,
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

    def test_chat_to_anthropic_max_tokens_stop(self) -> None:
        payload = chat_completion_to_anthropic(
            {"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]},
            client_model_alias="demo",
        )
        self.assertEqual(payload["stop_reason"], "max_tokens")
