"""Golden benchmark snapshot regression tests (MLX-74)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from django.test import SimpleTestCase

from orchestrator.benchmark_golden import (
    assert_within_golden_snapshot,
    compare_headline_metrics,
    extract_headline_metrics,
    load_golden_snapshot,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "benchmark_golden"
    / "sample_llmbench_output.json"
)


def _sample_results(scenario: str = "medium_conc4") -> dict:
    return {
        "results": [
            {
                "summary": {
                    "scenario": scenario,
                    "concurrency": 4,
                    "success_rate": "100%",
                    "ttft_p50_ms": 120.5,
                    "latency_p50_ms": 800.0,
                    "latency_p95_ms": 950.0,
                    "aggregate_tps": 180.0,
                },
            },
        ],
    }


@pytest.mark.golden
class BenchmarkGoldenComparatorTests(SimpleTestCase):
    def test_extract_headline_metrics_reads_scenario_summary(self) -> None:
        metrics = extract_headline_metrics(_sample_results(), scenario="medium_conc4")
        self.assertEqual(metrics["aggregate_tps"], 180.0)
        self.assertEqual(metrics["ttft_p50_ms"], 120.5)

    def test_compare_headline_metrics_accepts_values_within_tolerance(self) -> None:
        errors = compare_headline_metrics(
            {"aggregate_tps": 190.0, "ttft_p50_ms": 125.0},
            {"aggregate_tps": 180.0, "ttft_p50_ms": 120.5},
            tolerance_percent=15,
        )
        self.assertEqual(errors, [])

    def test_compare_headline_metrics_flags_large_drift(self) -> None:
        errors = compare_headline_metrics(
            {"aggregate_tps": 120.0},
            {"aggregate_tps": 180.0},
            tolerance_percent=15,
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("aggregate_tps", errors[0])

    def test_assert_within_golden_snapshot_passes_for_reference_output(self) -> None:
        snapshot = load_golden_snapshot("gemma4_mtp_medium_conc4")
        assert_within_golden_snapshot(_sample_results(), snapshot)

    def test_golden_fixture_file_matches_committed_snapshot(self) -> None:
        snapshot = load_golden_snapshot("gemma4_mtp_medium_conc4")
        if FIXTURE_PATH.is_file():
            payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
            assert_within_golden_snapshot(payload, snapshot)
