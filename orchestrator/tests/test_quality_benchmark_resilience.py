"""Tests for partial quality benchmark failures."""

from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase

from orchestrator.models import BenchmarkRun, InferenceInstance
from orchestrator.quality_benchmark_service import execute_quality_benchmark


class QualityBenchmarkResilienceTests(TestCase):
    def setUp(self) -> None:
        self.instance = InferenceInstance.objects.create(
            model_name="gemma-test",
            port=11446,
            launch_mode="TEXT",
            status="RUNNING",
            pid=42,
        )
        self.run = BenchmarkRun.objects.create(
            benchmark_kind="QUALITY",
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11380",
            model_id="gemma-test",
            params={
                "host": "127.0.0.1",
                "port": 11380,
                "quality_preset": "industry_lite",
            },
            status="PENDING",
        )

    @patch("orchestrator.quality_benchmark_service.run_platform_suites")
    @patch("orchestrator.quality_benchmark_service.run_lm_eval")
    def test_industry_failure_still_completes_platform_phase(
        self,
        mock_lm_eval: object,
        mock_platform: object,
    ) -> None:
        mock_lm_eval.side_effect = RuntimeError("AssertionError: messages")
        mock_platform.return_value = {
            "suites": {
                "text_platform": {
                    "pass_rate": 70.0,
                    "passed": 7,
                    "total": 10,
                    "cases": [],
                }
            }
        }

        execute_quality_benchmark(self.run)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "COMPLETED")
        self.assertEqual(self.run.error_message, "")
        self.assertIn("warnings", self.run.results)
        self.assertIn("text_platform_pass_rate", self.run.results["metrics"])
        self.assertTrue(self.run.results["industry"]["failed"])
