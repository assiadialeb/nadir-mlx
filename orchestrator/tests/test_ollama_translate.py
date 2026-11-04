"""Tests for Ollama protocol translation helpers."""

from django.test import SimpleTestCase

from orchestrator.gateway.protocol.ollama_translate import (
    chat_sse_delta_text,
    models_list_to_ollama_tags,
    ollama_chat_stream_chunk,
    ollama_chat_to_openai,
    ollama_embed_to_openai,
    openai_chat_to_ollama,
    openai_embeddings_to_ollama,
)
from orchestrator.gateway.router import GatewayRouteError


class OllamaTranslateTests(SimpleTestCase):
    def test_ollama_chat_to_openai(self) -> None:
        chat = ollama_chat_to_openai(
            {
                "model": "demo",
                "messages": [{"role": "user", "content": "Hi"}],
                "options": {"temperature": 0.2, "top_p": 0.8, "num_predict": 16},
            }
        )
        self.assertEqual(chat["temperature"], 0.2)
        self.assertEqual(chat["top_p"], 0.8)
        self.assertEqual(chat["max_tokens"], 16)
        self.assertFalse(chat["stream"])

    def test_ollama_chat_missing_model_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            ollama_chat_to_openai({"messages": [{"role": "user", "content": "Hi"}]})
        self.assertEqual(ctx.exception.code, "invalid_model")

    def test_ollama_chat_missing_messages_raises(self) -> None:
        with self.assertRaises(GatewayRouteError) as ctx:
            ollama_chat_to_openai({"model": "demo", "messages": []})
        self.assertEqual(ctx.exception.param, "messages")

    def test_openai_to_ollama_chat(self) -> None:
        payload = openai_chat_to_ollama(
            {"choices": [{"message": {"content": "Hello"}}]},
            client_model_alias="demo",
        )
        self.assertTrue(payload["done"])
        self.assertEqual(payload["message"]["content"], "Hello")

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

    def test_embeddings_to_ollama_single(self) -> None:
        payload = openai_embeddings_to_ollama(
            {"data": [{"embedding": [1.0]}]},
            client_model_alias="demo",
        )
        self.assertEqual(payload["embeddings"], [[1.0]])

    def test_tags_from_models_list(self) -> None:
        tags = models_list_to_ollama_tags(
            {"object": "list", "data": [{"id": "a"}, {"id": "b"}, "skip", {"id": 1}]}
        )
        self.assertEqual([row["name"] for row in tags["models"]], ["a", "b"])

    def test_stream_chunk_and_sse_delta(self) -> None:
        line = ollama_chat_stream_chunk(
            client_model_alias="demo",
            content="Hi",
            done=False,
        )
        self.assertTrue(line.endswith(b"\n"))
        self.assertIn(b'"done":false', line)
        delta = chat_sse_delta_text(
            b'data: {"choices":[{"delta":{"content":"ab"}}]}'
        )
        self.assertEqual(delta, "ab")
        self.assertIsNone(chat_sse_delta_text(b"data: [DONE]"))
