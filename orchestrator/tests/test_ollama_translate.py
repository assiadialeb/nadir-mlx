"""Tests for Ollama protocol translation helpers."""

from django.test import SimpleTestCase

from orchestrator.gateway.protocol.ollama_translate import (
    chat_sse_delta_text,
    models_list_to_ollama_ps,
    models_list_to_ollama_tags,
    ollama_chat_stream_chunk,
    ollama_chat_to_openai,
    ollama_embed_to_openai,
    ollama_generate_to_openai,
    ollama_show_stub,
    openai_chat_to_ollama,
    openai_chat_to_ollama_generate,
    openai_embeddings_to_ollama,
)
from orchestrator.gateway.router import GatewayRouteError


class OllamaTranslateTests(SimpleTestCase):
    def test_ollama_chat_to_openai(self) -> None:
        chat = ollama_chat_to_openai(
            {
                "model": "demo",
                "messages": [{"role": "user", "content": "Hi"}],
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "num_predict": 16,
                    "stop": ["END"],
                    "seed": 7,
                    "num_ctx": 4096,
                },
                "format": "json",
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            }
        )
        self.assertEqual(chat["temperature"], 0.2)
        self.assertEqual(chat["top_p"], 0.8)
        self.assertEqual(chat["max_tokens"], 16)
        self.assertEqual(chat["stop"], ["END"])
        self.assertEqual(chat["seed"], 7)
        self.assertEqual(chat["response_format"]["type"], "json_object")
        self.assertEqual(chat["tools"][0]["function"]["name"], "lookup")
        self.assertTrue(chat["stream"])
        self.assertNotIn("num_ctx", chat)

    def test_ollama_capabilities_and_show_stub(self) -> None:
        from orchestrator.gateway.protocol.ollama_translate import (
            ollama_capabilities_for_mode,
        )

        self.assertEqual(ollama_capabilities_for_mode("EMBEDDING"), ["embedding"])
        self.assertEqual(
            ollama_capabilities_for_mode("MULTIMODAL", tool_parser_type="json"),
            ["completion", "vision", "tools"],
        )
        stub = ollama_show_stub("demo", launch_mode="TEXT", tool_parser_type="")
        self.assertEqual(stub["capabilities"], ["completion"])

    def test_ollama_tool_message_fallbacks_and_thinking(self) -> None:
        chat = ollama_chat_to_openai(
            {
                "model": "demo",
                "stream": False,
                "messages": [
                    {
                        "role": "assistant",
                        "thinking": "hmm",
                        "tool_calls": [
                            {"function": {"name": "x", "arguments": {"a": 1}}},
                            "skip",
                            {"function": {"name": ""}},
                        ],
                    },
                    {"role": "tool", "content": {"ok": True}},
                    {"role": "tool", "tool_name": "unknown", "content": "n"},
                ],
            }
        )
        self.assertEqual(chat["messages"][0]["reasoning_content"], "hmm")
        self.assertEqual(chat["messages"][1]["tool_call_id"], "call_0")
        self.assertEqual(chat["messages"][1]["content"], '{"ok": true}')
        self.assertEqual(chat["messages"][2]["tool_call_id"], "call_unknown")

    def test_ollama_chat_preserves_tool_call_history(self) -> None:
        chat = ollama_chat_to_openai(
            {
                "model": "demo",
                "stream": False,
                "messages": [
                    {"role": "user", "content": "Paris?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {"city": "Paris"},
                                }
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_name": "get_weather",
                        "content": "sunny",
                    },
                ],
            }
        )
        assistant = chat["messages"][1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(
            assistant["tool_calls"][0]["function"]["arguments"],
            '{"city":"Paris"}',
        )
        tool = chat["messages"][2]
        self.assertEqual(tool["role"], "tool")
        self.assertEqual(tool["tool_call_id"], assistant["tool_calls"][0]["id"])
        self.assertEqual(tool["content"], "sunny")

    def test_ollama_chat_images_and_format_schema(self) -> None:
        chat = ollama_chat_to_openai(
            {
                "model": "demo",
                "format": {"type": "object", "properties": {"a": {"type": "string"}}},
                "messages": [
                    {
                        "role": "user",
                        "content": "describe",
                        "images": ["abc123"],
                    }
                ],
            }
        )
        content = chat["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertIn("base64,abc123", content[1]["image_url"]["url"])
        self.assertEqual(chat["response_format"]["type"], "json_schema")

    def test_ollama_chat_missing_model_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            ollama_chat_to_openai({"messages": [{"role": "user", "content": "Hi"}]})
        self.assertEqual(ctx.exception.code, "invalid_model")

    def test_ollama_chat_missing_messages_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            ollama_chat_to_openai({"model": "demo", "messages": []})
        self.assertEqual(ctx.exception.param, "messages")

    def test_ollama_generate_to_openai(self) -> None:
        chat = ollama_generate_to_openai(
            {
                "model": "demo",
                "prompt": "Hello",
                "system": "Be brief",
                "options": {"temperature": 0.1},
            }
        )
        self.assertEqual(
            chat["messages"],
            [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "Hello"},
            ],
        )
        self.assertEqual(chat["temperature"], 0.1)

    def test_ollama_generate_missing_prompt_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            ollama_generate_to_openai({"model": "demo"})
        self.assertEqual(ctx.exception.param, "prompt")

    def test_openai_to_ollama_chat(self) -> None:
        payload = openai_chat_to_ollama(
            {
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
            client_model_alias="demo",
        )
        self.assertTrue(payload["done"])
        self.assertEqual(payload["message"]["content"], "Hello")
        self.assertEqual(payload["done_reason"], "stop")
        self.assertEqual(payload["eval_count"], 3)
        self.assertEqual(payload["prompt_eval_count"], 2)

    def test_openai_to_ollama_chat_includes_thinking_when_requested(self) -> None:
        with_thinking = openai_chat_to_ollama(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Hi",
                            "reasoning_content": "plan",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            client_model_alias="demo",
            include_thinking=True,
        )
        self.assertEqual(with_thinking["message"]["thinking"], "plan")
        without = openai_chat_to_ollama(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Hi",
                            "reasoning_content": "plan",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            client_model_alias="demo",
        )
        self.assertNotIn("thinking", without["message"])

    def test_openai_to_ollama_generate(self) -> None:
        payload = openai_chat_to_ollama_generate(
            {
                "choices": [{"message": {"content": "Hi"}, "finish_reason": "length"}],
                "usage": {"completion_tokens": 9},
            },
            client_model_alias="demo",
        )
        self.assertEqual(payload["response"], "Hi")
        self.assertEqual(payload["done_reason"], "length")
        self.assertEqual(payload["eval_count"], 9)

    def test_openai_to_ollama_chat_empty_choices(self) -> None:
        payload = openai_chat_to_ollama({"choices": []}, client_model_alias="demo")
        self.assertEqual(payload["message"]["content"], "")

    def test_embed_from_input(self) -> None:
        body = ollama_embed_to_openai({"model": "demo", "input": ["a", "b"]})
        self.assertEqual(body["input"], ["a", "b"])

    def test_embed_from_prompt(self) -> None:
        body = ollama_embed_to_openai({"model": "demo", "prompt": "hello"})
        self.assertEqual(body["input"], "hello")

    def test_embed_missing_input_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            ollama_embed_to_openai({"model": "demo"})
        self.assertEqual(ctx.exception.param, "input")

    def test_embed_missing_model_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            ollama_embed_to_openai({"input": "x"})
        self.assertEqual(ctx.exception.code, "invalid_model")

    def test_embeddings_to_ollama(self) -> None:
        payload = openai_embeddings_to_ollama(
            {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3]}]},
            client_model_alias="demo",
        )
        self.assertEqual(payload["embeddings"], [[0.1, 0.2], [0.3]])

    def test_embeddings_to_ollama_legacy_singular(self) -> None:
        payload = openai_embeddings_to_ollama(
            {"data": [{"embedding": [1.0, 2.0]}]},
            client_model_alias="demo",
            legacy=True,
        )
        self.assertEqual(payload["embedding"], [1.0, 2.0])
        self.assertNotIn("embeddings", payload)

    def test_embeddings_to_ollama_single(self) -> None:
        payload = openai_embeddings_to_ollama(
            {"data": [{"embedding": [1.0]}]},
            client_model_alias="demo",
        )
        self.assertEqual(payload["embeddings"], [[1.0]])

    def test_tags_and_ps_from_models_list(self) -> None:
        models_payload = {
            "object": "list",
            "data": [{"id": "a"}, {"id": "b"}, "skip", {"id": 1}],
        }
        tags = models_list_to_ollama_tags(models_payload)
        self.assertEqual([row["name"] for row in tags["models"]], ["a", "b"])
        ps = models_list_to_ollama_ps(models_payload)
        self.assertEqual([row["name"] for row in ps["models"]], ["a", "b"])

    def test_show_stub(self) -> None:
        shown = ollama_show_stub("demo")
        self.assertEqual(shown["name"], "demo")
        self.assertIn("completion", shown["capabilities"])

    def test_stream_chunk_and_sse_delta(self) -> None:
        line = ollama_chat_stream_chunk(
            client_model_alias="demo",
            content="Hi",
            done=False,
        )
        self.assertTrue(line.endswith(b"\n"))
        self.assertIn(b'"done":false', line)
        done = ollama_chat_stream_chunk(
            client_model_alias="demo",
            content="",
            done=True,
            done_reason="stop",
        )
        self.assertIn(b'"done_reason":"stop"', done)
        delta = chat_sse_delta_text(b'data: {"choices":[{"delta":{"content":"ab"}}]}')
        self.assertEqual(delta, "ab")
        self.assertIsNone(chat_sse_delta_text(b"data: [DONE]"))
