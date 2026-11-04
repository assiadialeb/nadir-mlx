"""Unit tests for shared chat SSE helpers."""

from django.test import SimpleTestCase

from orchestrator.gateway.protocol.chat_sse import (
    ToolCallAccumulator,
    chat_sse_delta_text,
    chat_sse_finish_reason,
    parse_chat_sse_line,
    tool_calls_as_ollama,
)


class ChatSseHelperTests(SimpleTestCase):
    def test_parse_done_and_noise(self) -> None:
        self.assertTrue(parse_chat_sse_line(b"data: [DONE]").done)
        self.assertIsNone(parse_chat_sse_line(b": comment"))
        self.assertIsNone(parse_chat_sse_line(b"data: "))
        self.assertIsNone(parse_chat_sse_line(b"data: {bad"))
        self.assertIsNone(parse_chat_sse_line(b"data: []"))

    def test_parse_reasoning_variants_and_usage_only(self) -> None:
        event = parse_chat_sse_line(
            b'data: {"choices":[{"delta":{"reasoning_content":"a"}}]}'
        )
        self.assertEqual(event.reasoning, "a")
        event = parse_chat_sse_line(b'data: {"choices":[{"delta":{"reasoning":"b"}}]}')
        self.assertEqual(event.reasoning, "b")
        usage = parse_chat_sse_line(b'data: {"choices":[],"usage":{"prompt_tokens":1}}')
        self.assertEqual(usage.usage["prompt_tokens"], 1)

    def test_helpers_and_tool_call_conversion(self) -> None:
        self.assertEqual(
            chat_sse_delta_text(b'data: {"choices":[{"delta":{"content":"hi"}}]}'),
            "hi",
        )
        self.assertIsNone(chat_sse_delta_text(b"data: [DONE]"))
        self.assertEqual(
            chat_sse_finish_reason(
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
            ),
            "stop",
        )
        self.assertIsNone(chat_sse_finish_reason(b": noop"))
        converted = tool_calls_as_ollama(
            [
                {"function": {"name": "x", "arguments": '{"a":1}'}},
                {"function": {"name": "y", "arguments": "{bad"}},
                {"function": {"name": "z", "arguments": ["nope"]}},
                {"function": {"name": ""}},
                "skip",
            ]
        )
        self.assertEqual(converted[0]["function"]["arguments"], {"a": 1})
        self.assertEqual(converted[1]["function"]["arguments"], {})
        self.assertEqual(converted[2]["function"]["arguments"], {})
        self.assertEqual(len(converted), 3)

    def test_tool_call_accumulator(self) -> None:
        acc = ToolCallAccumulator()
        acc.ingest(
            [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "lookup", "arguments": "{"},
                }
            ]
        )
        acc.ingest([{"index": 0, "function": {"arguments": "}"}}])
        calls = acc.completed_calls()
        self.assertTrue(acc.has_calls())
        self.assertEqual(calls[0]["index"], 0)
        self.assertEqual(calls[0]["function"]["arguments"], "{}")
