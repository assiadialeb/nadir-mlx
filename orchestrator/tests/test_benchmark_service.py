"""Tests for benchmark service helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from orchestrator.benchmark_service import (
    BENCHMARK_MAX_REQUESTS_PER_SCENARIO,
    _resolve_target,
    parse_benchmark_form,
    resolve_benchmark_model_id,
)
from orchestrator.models import InferenceInstance


class ParseBenchmarkFormTests(TestCase):
    def test_num_requests_rejects_values_above_cap(self) -> None:
        with self.assertRaises(ValueError):
            parse_benchmark_form(
                {
                    "target_type": "INSTANCE",
                    "instance_id": "1",
                    "num_requests": str(BENCHMARK_MAX_REQUESTS_PER_SCENARIO + 1),
                    "concurrency": "1",
                    "categories": ["medium"],
                }
            )

    @override_settings(DEBUG=False, NADIR_BENCHMARK_ENDPOINT_ENABLED=False)
    def test_endpoint_target_rejected_when_disabled(self) -> None:
        with self.assertRaises(ValueError):
            parse_benchmark_form(
                {
                    "target_type": "ENDPOINT",
                    "endpoint_host": "localhost",
                    "endpoint_port": "11434",
                    "num_requests": "5",
                    "concurrency": "1",
                    "categories": ["medium"],
                }
            )

    @override_settings(
        DEBUG=False,
        NADIR_BENCHMARK_ENDPOINT_ENABLED=True,
    )
    def test_endpoint_target_rejects_private_host_in_prod(self) -> None:
        with self.assertRaises(ValueError):
            parse_benchmark_form(
                {
                    "target_type": "ENDPOINT",
                    "endpoint_host": "10.0.0.5",
                    "endpoint_port": "11434",
                    "num_requests": "5",
                    "concurrency": "1",
                    "categories": ["medium"],
                }
            )

    @override_settings(
        DEBUG=False,
        NADIR_BENCHMARK_ENDPOINT_ENABLED=True,
    )
    def test_endpoint_target_accepts_localhost_in_prod_when_enabled(self) -> None:
        parsed = parse_benchmark_form(
            {
                "target_type": "ENDPOINT",
                "endpoint_host": "localhost",
                "endpoint_port": "11434",
                "num_requests": "5",
                "concurrency": "1",
                "categories": ["medium"],
            }
        )
        self.assertEqual(parsed["target_type"], "ENDPOINT")
        self.assertEqual(parsed["host"], "localhost")

    def test_parse_benchmark_form_accepts_quality_kind(self) -> None:
        parsed = parse_benchmark_form(
            {
                "target_type": "INSTANCE",
                "instance_id": "1",
                "benchmark_kind": "QUALITY",
                "quality_preset": "industry_lite",
                "num_requests": "5",
                "concurrency": "1",
                "categories": ["medium"],
            }
        )
        self.assertEqual(parsed["benchmark_kind"], "QUALITY")
        self.assertEqual(parsed["params"]["quality_preset"], "industry_lite")

    def test_parse_benchmark_form_rejects_invalid_kind(self) -> None:
        with self.assertRaises(ValueError):
            parse_benchmark_form(
                {
                    "target_type": "INSTANCE",
                    "instance_id": "1",
                    "benchmark_kind": "INVALID",
                    "num_requests": "5",
                    "concurrency": "1",
                    "categories": ["medium"],
                }
            )


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
        resolved = resolve_benchmark_model_id("localhost", 11380, instance, "custom-model")
        self.assertEqual(resolved, "custom-model")

    def test_multimodal_instance_uses_gateway_alias_without_http_probe(self) -> None:
        instance = self._instance("MULTIMODAL", "Qwen3.6-35B-A3B-4bit")
        with patch("orchestrator.benchmark_service.httpx.get") as mock_get:
            resolved = resolve_benchmark_model_id("localhost", 11380, instance, "")
        mock_get.assert_not_called()
        self.assertEqual(resolved, "Qwen3.6-35B-A3B-4bit")

    def test_text_instance_uses_gateway_alias_without_http_probe(self) -> None:
        instance = self._instance("TEXT", "gemma-4-e2b-it-4bit")
        with patch("orchestrator.benchmark_service.httpx.get") as mock_get:
            resolved = resolve_benchmark_model_id("localhost", 11380, instance, "")
        mock_get.assert_not_called()
        self.assertEqual(resolved, "gemma-4-e2b-it-4bit")

    def test_instance_alias_from_server_config_model_id(self) -> None:
        instance = self._instance("TEXT", "folder-name")
        instance.server_config = {"model_id": "my-alias"}
        with patch("orchestrator.benchmark_service.httpx.get") as mock_get:
            resolved = resolve_benchmark_model_id("localhost", 11380, instance, "")
        mock_get.assert_not_called()
        self.assertEqual(resolved, "my-alias")

    def test_external_endpoint_uses_v1_models(self) -> None:
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": [{"id": "llama3"}]}
        with patch("orchestrator.benchmark_service.httpx.get", return_value=response):
            resolved = resolve_benchmark_model_id("localhost", 11434, None, "")
        self.assertEqual(resolved, "llama3")


class ResolveBenchmarkTargetTests(TestCase):
    def setUp(self) -> None:
        self.instance = InferenceInstance.objects.create(
            model_name="Qwen3.6-35B-A3B-4bit",
            port=11475,
            launch_mode="MULTIMODAL",
            status="RUNNING",
            pid=999,
        )

    @override_settings(NADIR_GATEWAY_HOST="127.0.0.1", NADIR_GATEWAY_PORT=11380)
    def test_instance_target_routes_through_gateway(self) -> None:
        host, port, instance = _resolve_target("INSTANCE", self.instance.id, None, None)
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 11380)
        self.assertEqual(instance.id, self.instance.id)

    def test_endpoint_target_uses_custom_host_and_port(self) -> None:
        with override_settings(DEBUG=True, NADIR_BENCHMARK_ENDPOINT_ENABLED=True):
            host, port, instance = _resolve_target("ENDPOINT", None, "localhost", 11434)
        self.assertEqual(host, "localhost")
        self.assertEqual(port, 11434)
        self.assertIsNone(instance)

    @override_settings(
        DEBUG=False,
        NADIR_BENCHMARK_ENDPOINT_ENABLED=True,
    )
    def test_endpoint_target_rejects_metadata_ip(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_target("ENDPOINT", None, "169.254.169.254", 11434)
