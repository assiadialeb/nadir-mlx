"""Tests for benchmark service helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import TestCase

from orchestrator.benchmark_service import resolve_benchmark_model_id
from orchestrator.models import InferenceInstance


class ResolveBenchmarkModelIdTests(TestCase):
    def _instance(self, launch_mode: str, model_name: str = "example-model") -> InferenceInstance:
        return InferenceInstance(
            model_name=model_name,
            port=11475,
            launch_mode=launch_mode,
            status="RUNNING",
        )

    def test_user_model_id_takes_priority(self) -> None:
        instance = self._instance("MULTIMODAL")
        resolved = resolve_benchmark_model_id("localhost", 11475, instance, "custom-model")
        self.assertEqual(resolved, "custom-model")

    def test_multimodal_instance_uses_default_model_without_http_probe(self) -> None:
        instance = self._instance("MULTIMODAL", "Qwen3.6-35B-A3B-4bit")
        with patch("orchestrator.benchmark_service.httpx.get") as mock_get:
            resolved = resolve_benchmark_model_id("localhost", 11475, instance, "")
        mock_get.assert_not_called()
        self.assertEqual(resolved, "default_model")

    def test_text_instance_uses_default_model_without_http_probe(self) -> None:
        instance = self._instance("TEXT", "gemma-4-e2b-it-4bit")
        with patch("orchestrator.benchmark_service.httpx.get") as mock_get:
            resolved = resolve_benchmark_model_id("localhost", 11475, instance, "")
        mock_get.assert_not_called()
        self.assertEqual(resolved, "default_model")

    def test_external_endpoint_uses_v1_models(self) -> None:
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": [{"id": "llama3"}]}
        with patch("orchestrator.benchmark_service.httpx.get", return_value=response):
            resolved = resolve_benchmark_model_id("localhost", 11434, None, "")
        self.assertEqual(resolved, "llama3")

    def test_multimodal_ignores_misleading_v1_models_list(self) -> None:
        """Regression: mlx_vlm may expose sentence-transformers in /v1/models."""
        instance = self._instance("MULTIMODAL", "Qwen3.6-35B-A3B-4bit")
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": [{"id": "sentence-transformers/all-MiniLM-L6-v2"}],
        }
        with patch("orchestrator.benchmark_service.httpx.get", return_value=response) as mock_get:
            resolved = resolve_benchmark_model_id("localhost", 11475, instance, "")
        mock_get.assert_not_called()
        self.assertEqual(resolved, "default_model")
