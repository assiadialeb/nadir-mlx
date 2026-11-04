"""Tests for Ollama protocol translation helpers."""

from django.test import SimpleTestCase

from orchestrator.gateway.protocol.ollama_translate import (
    models_list_to_ollama_tags,
    ollama_chat_to_openai,
    openai_chat_to_ollama,
)


class OllamaTranslateTests(SimpleTestCase):
    def test_ollama_chat_to_openai(self) -> None:
        chat = ollama_chat_to_openai(
            {
                "model": "demo",
                "messages": [{"role": "user", "content": "Hi"}],
                "options": {"temperature": 0.2, "num_predict": 16},
            }
        )
        self.assertEqual(chat["temperature"], 0.2)
        self.assertEqual(chat["max_tokens"], 16)

    def test_openai_to_ollama_chat(self) -> None:
        payload = openai_chat_to_ollama(
            {"choices": [{"message": {"content": "Hello"}}]},
            client_model_alias="demo",
        )
        self.assertTrue(payload["done"])
        self.assertEqual(payload["message"]["content"], "Hello")

    def test_tags_from_models_list(self) -> None:
        tags = models_list_to_ollama_tags(
            {"object": "list", "data": [{"id": "a"}, {"id": "b"}]}
        )
        self.assertEqual([row["name"] for row in tags["models"]], ["a", "b"])
