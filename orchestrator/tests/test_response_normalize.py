"""Tests for gateway response normalization helpers."""

from django.test import SimpleTestCase

from orchestrator.gateway.response_normalize import (
    dual_write_reasoning_fields,
    normalize_client_chat_payload,
    restore_response_model,
)


class ResponseNormalizeTests(SimpleTestCase):
    def test_restore_response_model_replaces_upstream_value(self) -> None:
        payload = restore_response_model({"model": "default_model", "id": "1"}, "demo")
        self.assertEqual(payload["model"], "demo")
        self.assertEqual(payload["id"], "1")

    def test_dual_write_reasoning_on_message(self) -> None:
        payload = {
            "choices": [
                {"message": {"role": "assistant", "content": "hi", "reasoning": "think"}}
            ]
        }
        normalized = dual_write_reasoning_fields(payload)
        message = normalized["choices"][0]["message"]
        self.assertEqual(message["reasoning"], "think")
        self.assertEqual(message["reasoning_content"], "think")

    def test_dual_write_reasoning_on_delta(self) -> None:
        payload = {"choices": [{"delta": {"reasoning": "step"}}]}
        normalized = dual_write_reasoning_fields(payload)
        self.assertEqual(normalized["choices"][0]["delta"]["reasoning_content"], "step")

    def test_normalize_client_chat_payload_combines_model_and_reasoning(self) -> None:
        payload = {
            "model": "default_model",
            "choices": [{"message": {"reasoning": "x", "content": "y"}}],
        }
        normalized = normalize_client_chat_payload(payload, "alias-chat")
        self.assertEqual(normalized["model"], "alias-chat")
        self.assertEqual(
            normalized["choices"][0]["message"]["reasoning_content"],
            "x",
        )
