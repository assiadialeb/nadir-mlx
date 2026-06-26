"""Tests for quality benchmark orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import TestCase

from orchestrator.models import BenchmarkRun, InferenceInstance
from orchestrator.quality_benchmark_service import (
    build_quality_results,
    execute_quality_benchmark,
    summarize_industry_metrics,
)


class QualityBenchmarkServiceTests(TestCase):
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

    def test_summarize_industry_metrics_converts_to_percent(self) -> None:
        metrics = summarize_industry_metrics(
            {
                "tasks": {
                    "gsm8k": {"exact_match": 0.74},
                    "mmlu": {"acc": 0.61},
                }
            }
        )
        self.assertEqual(metrics["gsm8k_exact_match"], 74.0)
        self.assertEqual(metrics["mmlu_acc"], 61.0)

    def test_build_quality_results_merges_metrics(self) -> None:
        industry = {"tasks": {"gsm8k": {"exact_match": 0.5}}, "skipped": False}
        platform = {"suites": {"text_platform": {"pass_rate": 90.0, "passed": 9, "total": 10}}}
        payload = build_quality_results(industry, platform, preset="industry_lite")
        self.assertIn("gsm8k_exact_match", payload["metrics"])
        self.assertIn("text_platform_pass_rate", payload["metrics"])

    @patch("orchestrator.quality_benchmark_service.run_lm_eval")
    def test_execute_quality_benchmark_persists_results(
        self,
        mock_lm_eval: MagicMock,
    ) -> None:
        mock_lm_eval.return_value = {
            "skipped": True,
            "reason": "not installed",
            "tasks": {},
        }
        with patch(
            "orchestrator.quality_benchmark_service.run_platform_suites",
            return_value={
                "suites": {
                    "text_platform": {
                        "pass_rate": 80.0,
                        "passed": 8,
                        "total": 10,
                        "cases": [],
                    }
                }
            },
        ):
            execute_quality_benchmark(self.run)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "COMPLETED")
        self.assertEqual(self.run.results["benchmark_kind"], "QUALITY")
        self.assertEqual(self.run.results["metrics"]["text_platform_pass_rate"], 80.0)
