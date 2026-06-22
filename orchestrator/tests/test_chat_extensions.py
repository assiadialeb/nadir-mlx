"""Tests for OpenAI chat extension qualification helpers."""

from django.test import SimpleTestCase

from orchestrator.gateway.chat_extensions import (
    count_vision_images,
    has_tool_definitions,
    has_vision_content,
    prepare_chat_upstream_body,
    structured_output_type,
)


class ChatExtensionTests(SimpleTestCase):
    def test_prepare_chat_upstream_body_rewrites_model_only(self) -> None:
        body = {
            "model": "alias",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "response_format": {"type": "json_object"},
        }
        upstream = prepare_chat_upstream_body(body, "default_model")
        self.assertEqual(upstream["model"], "default_model")
        self.assertEqual(body["model"], "alias")
        self.assertEqual(len(upstream["tools"]), 1)

    def test_has_tool_definitions_detects_non_empty_tools(self) -> None:
        self.assertFalse(has_tool_definitions({}))
        self.assertFalse(has_tool_definitions({"tools": []}))
        self.assertTrue(
            has_tool_definitions(
                {"tools": [{"type": "function", "function": {"name": "ping"}}]}
            )
        )

    def test_structured_output_type_reads_json_modes(self) -> None:
        self.assertIsNone(structured_output_type({}))
        self.assertEqual(
            structured_output_type({"response_format": {"type": "json_object"}}),
            "json_object",
        )
        self.assertEqual(
            structured_output_type(
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "answer", "schema": {}},
                    }
                }
            ),
            "json_schema",
        )

    def test_has_vision_content_detects_image_url_blocks(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            }
        ]
        self.assertTrue(has_vision_content(messages))
        self.assertEqual(count_vision_images(messages), 1)

    def test_has_vision_content_false_for_text_only(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        self.assertFalse(has_vision_content(messages))
