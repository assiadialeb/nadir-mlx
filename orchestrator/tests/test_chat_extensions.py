"""Tests for OpenAI chat extension qualification helpers."""

from django.test import SimpleTestCase

from orchestrator.gateway.chat_extensions import (
    has_tool_definitions,
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
