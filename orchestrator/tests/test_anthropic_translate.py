"""Tests for Anthropic Messages translation helpers."""

from django.test import SimpleTestCase

from orchestrator.gateway.protocol.anthropic_translate import (
    anthropic_request_to_chat,
    chat_completion_to_anthropic,
)


class AnthropicTranslateTests(SimpleTestCase):
    def test_anthropic_request_to_chat(self) -> None:
        chat = anthropic_request_to_chat(
            {
                "model": "demo",
                "max_tokens": 64,
                "system": "Be kind",
                "messages": [{"role": "user", "content": "Hi"}],
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
