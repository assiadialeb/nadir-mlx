"""Tests for Responses ↔ chat completions translation."""

from django.test import SimpleTestCase

from orchestrator.gateway.protocol.responses_translate import (
    chat_completion_to_response,
    responses_request_to_chat,
)
from orchestrator.gateway.router import GatewayRouteError


class ResponsesTranslateTests(SimpleTestCase):
    def test_string_input_maps_to_user_message(self) -> None:
        chat = responses_request_to_chat(
            {
                "model": "demo",
                "instructions": "Be brief",
                "input": "Hello",
                "max_output_tokens": 32,
            }
        )
        self.assertEqual(chat["model"], "demo")
        self.assertEqual(chat["max_tokens"], 32)
        self.assertEqual(
            chat["messages"],
            [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "Hello"},
            ],
        )

    def test_unsupported_field_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            responses_request_to_chat(
                {
                    "model": "demo",
                    "input": "Hi",
                    "previous_response_id": "resp_1",
                }
            )
        self.assertEqual(ctx.exception.code, "invalid_request")
        self.assertEqual(ctx.exception.param, "previous_response_id")

    def test_chat_completion_to_response_shape(self) -> None:
        payload = chat_completion_to_response(
            {
                "id": "chatcmpl-abc",
                "created": 100,
                "choices": [
                    {"message": {"role": "assistant", "content": "Hi there"}},
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
            client_model_alias="demo",
        )
        self.assertEqual(payload["object"], "response")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["model"], "demo")
        self.assertEqual(payload["output"][0]["content"][0]["text"], "Hi there")
        self.assertEqual(payload["usage"]["total_tokens"], 5)
