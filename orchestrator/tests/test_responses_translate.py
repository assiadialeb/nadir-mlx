"""Tests for Responses ↔ chat completions translation."""

from django.test import SimpleTestCase

from orchestrator.gateway.protocol.responses_translate import (
    build_response_stream_events,
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

    def test_missing_model_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            responses_request_to_chat({"input": "Hi"})
        self.assertEqual(ctx.exception.code, "invalid_model")

    def test_missing_input_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            responses_request_to_chat({"model": "demo"})
        self.assertEqual(ctx.exception.code, "invalid_request")
        self.assertEqual(ctx.exception.param, "input")

    def test_invalid_input_type_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            responses_request_to_chat({"model": "demo", "input": 42})
        self.assertEqual(ctx.exception.param, "input")

    def test_empty_message_array_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            responses_request_to_chat({"model": "demo", "input": []})
        self.assertEqual(ctx.exception.param, "input")

    def test_message_array_and_json_schema_format(self) -> None:
        chat = responses_request_to_chat(
            {
                "model": "demo",
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
                    {"type": "message", "role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "Again"},
                    "skip-me",
                ],
                "text": {"format": {"type": "json_object"}},
                "temperature": 0.1,
                "stream": True,
            }
        )
        self.assertEqual(chat["response_format"], {"type": "json_object"})
        self.assertEqual(chat["temperature"], 0.1)
        self.assertTrue(chat["stream"])
        self.assertEqual(chat["messages"][0]["content"], "Hi")
        self.assertEqual(chat["messages"][1]["role"], "assistant")

    def test_multimodal_content_keeps_image_parts(self) -> None:
        chat = responses_request_to_chat(
            {
                "model": "demo",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe"},
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,abc",
                            },
                        ],
                    }
                ],
            }
        )
        content = chat["messages"][0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "input_image")

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

    def test_chat_completion_empty_choices_returns_empty_text(self) -> None:
        payload = chat_completion_to_response(
            {"choices": [], "created": 1},
            client_model_alias="demo",
        )
        self.assertEqual(payload["output"][0]["content"][0]["text"], "")

    def test_chat_completion_legacy_text_choice(self) -> None:
        payload = chat_completion_to_response(
            {"choices": [{"text": "legacy"}], "created": 1},
            client_model_alias="demo",
        )
        self.assertEqual(payload["output"][0]["content"][0]["text"], "legacy")

    def test_build_response_stream_events(self) -> None:
        meta = build_response_stream_events(
            client_model_alias="demo",
            response_id="resp_1",
        )
        self.assertEqual(meta["response_id"], "resp_1")
        self.assertEqual(meta["model"], "demo")
        self.assertTrue(meta["message_id"].startswith("msg_"))
