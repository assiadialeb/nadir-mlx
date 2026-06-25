"""Unit tests for orchestrator.vendor.llmbench."""

from __future__ import annotations

from django.test import SimpleTestCase

from orchestrator.vendor.llmbench import (
    BenchmarkResult,
    RequestResult,
    _delta_text,
)


class DeltaTextTests(SimpleTestCase):
    def test_collects_content_and_reasoning_fields(self) -> None:
        delta = {
            "content": "Hello",
            "reasoning": " thinking",
            "reasoning_content": " more",
        }
        self.assertEqual(_delta_text(delta), "Hello thinking more")

    def test_ignores_non_string_values(self) -> None:
        self.assertEqual(_delta_text({"content": None}), "")


class BenchmarkSummaryTests(SimpleTestCase):
    def _result(
        self,
        completion_tokens: int,
        total_ms: float,
        *,
        success: bool = True,
    ) -> RequestResult:
        return RequestResult(
            prompt_tokens=10,
            completion_tokens=completion_tokens,
            ttft_ms=50.0,
            total_ms=total_ms,
            success=success,
        )

    def test_aggregate_tps_uses_wall_clock_not_max_latency(self) -> None:
        """Regression: conc=4 must not divide by slowest request only."""
        results = [
            self._result(95, 2300.0),
            self._result(95, 2400.0),
            self._result(95, 2500.0),
            self._result(95, 2600.0),
        ]
        bench = BenchmarkResult(
            scenario="medium_conc4",
            concurrency=4,
            num_requests=4,
            prompt_category="medium",
            results=results,
        )
        summary = bench.summary(wall_sec=11.42)
        # 380 tokens / 11.42 s ≈ 33.3 tok/s (not ~146 from max latency alone)
        self.assertAlmostEqual(summary["aggregate_tps"], 33.3, places=1)
        self.assertEqual(summary["total_tokens_out"], 380)

    def test_zero_token_requests_excluded_from_successes(self) -> None:
        results = [
            self._result(50, 2000.0),
            RequestResult(
                prompt_tokens=10,
                completion_tokens=0,
                ttft_ms=0.0,
                total_ms=500.0,
                success=False,
                error="No completion tokens reported by the server.",
            ),
        ]
        bench = BenchmarkResult(
            scenario="medium_conc1",
            concurrency=1,
            num_requests=2,
            prompt_category="medium",
            results=results,
        )
        summary = bench.summary(wall_sec=4.0)
        self.assertEqual(summary["success_rate"], "1/2")
        self.assertEqual(summary["failed_requests"], 1)
