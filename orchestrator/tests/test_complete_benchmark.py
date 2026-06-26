"""Tests for COMPLETE benchmark orchestration."""

from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase

from orchestrator.benchmark_service import _complete_thread
from orchestrator.models import BenchmarkRun, InferenceInstance


class CompleteBenchmarkTests(TestCase):
    def setUp(self) -> None:
        self.instance = InferenceInstance.objects.create(
            model_name="gemma-test",
            port=11446,
            launch_mode="TEXT",
            status="RUNNING",
            pid=42,
        )
        self.parent = BenchmarkRun.objects.create(
            benchmark_kind="COMPLETE",
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11380",
            model_id="gemma-test",
            params={
                "host": "127.0.0.1",
                "port": 11380,
                "concurrency": [1],
                "categories": ["medium"],
                "num_requests": 2,
                "quality_preset": "industry_lite",
            },
            status="PENDING",
        )

    @patch("orchestrator.benchmark_service.execute_perf_benchmark")
    def test_complete_thread_skips_quality_when_perf_fails(self, mock_perf: object) -> None:
        def fail_perf(run: BenchmarkRun) -> None:
            run.status = "FAILED"
            run.error_message = "perf boom"
            run.save(update_fields=["status", "error_message"])

        mock_perf.side_effect = fail_perf
        _complete_thread(self.parent.id)
        self.parent.refresh_from_db()
        self.assertEqual(self.parent.status, "FAILED")
        self.assertEqual(self.parent.child_runs.count(), 1)

    @patch("orchestrator.benchmark_service.execute_quality_benchmark")
    @patch("orchestrator.benchmark_service.execute_perf_benchmark")
    def test_complete_thread_succeeds_when_both_phases_ok(
        self,
        mock_perf: object,
        mock_quality: object,
    ) -> None:
        def ok_perf(run: BenchmarkRun) -> None:
            run.status = "COMPLETED"
            run.results = {
                "results": [
                    {
                        "summary": {
                            "scenario": "medium_conc1",
                            "aggregate_tps": 42,
                            "ttft_p50_ms": 100,
                        }
                    }
                ]
            }
            run.save(update_fields=["status", "results"])

        def ok_quality(run: BenchmarkRun) -> None:
            run.status = "COMPLETED"
            run.results = {
                "metrics": {"text_platform_pass_rate": 90.0},
            }
            run.save(update_fields=["status", "results"])

        mock_perf.side_effect = ok_perf
        mock_quality.side_effect = ok_quality

        _complete_thread(self.parent.id)
        self.parent.refresh_from_db()
        self.assertEqual(self.parent.status, "COMPLETED")
        self.assertEqual(self.parent.child_runs.count(), 2)
        self.assertIn("quality_summary", self.parent.results)
