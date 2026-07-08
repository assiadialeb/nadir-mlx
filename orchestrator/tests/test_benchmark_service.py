"""Tests for benchmark service helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.test import TestCase, override_settings

from orchestrator.benchmark_service import (
    BENCHMARK_MAX_REQUESTS_PER_SCENARIO,
    _build_command,
    _resolve_target,
    delete_benchmark_runs_for_model,
    execute_perf_benchmark,
    parse_benchmark_form,
    resolve_benchmark_model_id,
    start_benchmark,
)
from orchestrator.models import BenchmarkRun, InferenceInstance


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


@override_settings(LOGS_DIR="/tmp/mlx-bench-exec-tests")
class ExecutePerfBenchmarkTests(TestCase):
    def setUp(self) -> None:
        self.logs_dir = Path(settings.LOGS_DIR)
        (self.logs_dir / "benchmarks").mkdir(parents=True, exist_ok=True)
        self.instance = InferenceInstance.objects.create(
            model_name="gemma-test",
            port=11446,
            launch_mode="TEXT",
            status="RUNNING",
            pid=1234,
        )

    def test_build_command_includes_model_and_categories(self) -> None:
        run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            model_id="gemma-chat",
            params={
                "host": "127.0.0.1",
                "port": 11446,
                "num_requests": 10,
                "concurrency": [1, 4],
                "categories": ["medium", "short"],
            },
            status="PENDING",
        )
        command = _build_command(run, Path("/tmp/out.json"))
        self.assertIn("--model", command)
        self.assertIn("gemma-chat", command)
        self.assertEqual(command.count("--categories"), 2)

    @patch("orchestrator.benchmark_service.subprocess.run")
    def test_execute_perf_benchmark_persists_completed_results(
        self,
        mock_run: MagicMock,
    ) -> None:
        run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params={
                "host": "127.0.0.1",
                "port": 11446,
                "num_requests": 5,
                "concurrency": [1],
                "categories": ["medium"],
            },
            status="PENDING",
        )
        output_path = self.logs_dir / "benchmarks" / f"bench_{run.id}.json"
        output_path.write_text(json.dumps({"results": [{"summary": {"scenario": "medium_conc1"}}]}), encoding="utf-8")
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        execute_perf_benchmark(run)

        run.refresh_from_db()
        self.assertEqual(run.status, "COMPLETED")
        self.assertIn("results", run.results)

    @patch("orchestrator.benchmark_service.subprocess.run")
    def test_execute_perf_benchmark_marks_failed_on_nonzero_exit(
        self,
        mock_run: MagicMock,
    ) -> None:
        run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params={"host": "127.0.0.1", "port": 11446, "concurrency": [1], "categories": ["medium"]},
            status="PENDING",
        )
        mock_run.return_value = MagicMock(returncode=1, stderr="bench crashed", stdout="")

        execute_perf_benchmark(run)

        run.refresh_from_db()
        self.assertEqual(run.status, "FAILED")
        self.assertIn("bench crashed", run.error_message)

    def test_delete_benchmark_runs_for_model_removes_matching_rows(self) -> None:
        run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params={},
            status="COMPLETED",
        )
        deleted = delete_benchmark_runs_for_model(self.instance.model_name)
        self.assertEqual(deleted, 1)
        self.assertFalse(BenchmarkRun.objects.filter(id=run.id).exists())

    @patch(
        "orchestrator.benchmark_service.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="bench", timeout=1),
    )
    def test_execute_perf_benchmark_marks_failed_on_timeout(self, _mock_run: MagicMock) -> None:
        run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params={"host": "127.0.0.1", "port": 11446, "concurrency": [1], "categories": ["medium"]},
            status="PENDING",
        )
        execute_perf_benchmark(run)
        run.refresh_from_db()
        self.assertEqual(run.status, "FAILED")
        self.assertIn("timed out", run.error_message.lower())

    @patch("orchestrator.benchmark_service._start_benchmark_thread")
    @override_settings(NADIR_GATEWAY_HOST="127.0.0.1", NADIR_GATEWAY_PORT=11380)
    def test_start_benchmark_creates_pending_run(self, mock_thread: MagicMock) -> None:
        run = start_benchmark(
            "INSTANCE",
            self.instance.id,
            None,
            None,
            "",
            {"categories": ["medium"], "concurrency": [1], "num_requests": 5},
            benchmark_kind="PERF",
        )
        self.assertEqual(run.status, "PENDING")
        self.assertEqual(run.benchmark_kind, "PERF")
        mock_thread.assert_called_once_with(run)

    @patch("orchestrator.benchmark_service._start_benchmark_thread")
    @override_settings(NADIR_GATEWAY_HOST="127.0.0.1", NADIR_GATEWAY_PORT=11380)
    def test_start_benchmark_stores_draft_profile_from_instance(
        self,
        mock_thread: MagicMock,
    ) -> None:
        self.instance.server_config = {
            "model_id": "gemma-chat",
            "advanced": {"draft_kind": "mtp", "draft_model": "assistant"},
        }
        self.instance.save(update_fields=["server_config"])
        run = start_benchmark(
            "INSTANCE",
            self.instance.id,
            None,
            None,
            "",
            {"categories": ["medium"], "concurrency": [1], "num_requests": 5},
            benchmark_kind="PERF",
        )
        self.assertEqual(run.params["draft_profile"]["draft_kind"], "mtp")
