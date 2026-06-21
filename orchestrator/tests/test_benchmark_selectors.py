"""Tests for benchmark history, chart, and comparison selectors."""

from django.test import TestCase

from orchestrator.benchmark_selectors import (
    benchmark_preset_key,
    benchmark_run_list_row,
    build_comparison_snapshot,
    chart_series_for_runs,
    comparison_rows,
    filter_benchmark_runs,
    find_comparison_candidates,
    paginate_benchmark_runs,
    parse_benchmark_history_query,
    summary_for_scenario,
)
from orchestrator.models import BenchmarkRun, InferenceInstance


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
                    "tps_per_req_median": 45.2,
                    "aggregate_tps": 180.0,
                },
            },
        ],
    }


class BenchmarkSelectorsTests(TestCase):
    def setUp(self) -> None:
        self.instance = InferenceInstance.objects.create(
            model_name="gemma-test",
            port=11446,
            launch_mode="TEXT",
            status="RUNNING",
            pid=9999,
        )
        self.params = {
            "categories": ["medium"],
            "concurrency": [1, 4],
            "num_requests": 20,
        }

    def test_benchmark_preset_key_is_stable(self) -> None:
        key_a = benchmark_preset_key(self.params)
        key_b = benchmark_preset_key(
            {"categories": ["medium"], "concurrency": [4, 1], "num_requests": 20},
        )
        self.assertEqual(key_a, key_b)

    def test_parse_benchmark_history_query_normalizes_values(self) -> None:
        parsed = parse_benchmark_history_query(
            {
                "model": "gemma",
                "launch_mode": "text",
                "status": "completed",
                "instance_id": "3",
                "page": "2",
            },
        )
        self.assertEqual(parsed["model_name"], "gemma")
        self.assertEqual(parsed["launch_mode"], "TEXT")
        self.assertEqual(parsed["status"], "COMPLETED")
        self.assertEqual(parsed["instance_id"], 3)
        self.assertEqual(parsed["page"], 2)

    def test_filter_benchmark_runs_by_model_and_instance(self) -> None:
        run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        BenchmarkRun.objects.create(
            target_type="ENDPOINT",
            endpoint_url="http://127.0.0.1:11434/v1",
            model_id="llama3",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )

        filtered = filter_benchmark_runs(
            {"model_name": "gemma", "instance_id": self.instance.id},
        )
        self.assertEqual(list(filtered), [run])

    def test_paginate_benchmark_runs_returns_page(self) -> None:
        for index in range(3):
            BenchmarkRun.objects.create(
                target_type="ENDPOINT",
                endpoint_url=f"http://127.0.0.1:1143{index}/v1",
                params=self.params,
                status="COMPLETED",
            )
        queryset = BenchmarkRun.objects.order_by("-created_at")
        runs, page_obj = paginate_benchmark_runs(queryset, 1, per_page=2)
        self.assertEqual(len(runs), 2)
        self.assertEqual(page_obj.paginator.num_pages, 2)

    def test_benchmark_run_list_row_extracts_metrics(self) -> None:
        run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        row = benchmark_run_list_row(run)
        self.assertEqual(row["model_name"], "gemma-test")
        self.assertEqual(row["launch_mode"], "TEXT")
        self.assertEqual(row["ttft_p50_ms"], 120.5)
        self.assertEqual(row["aggregate_tps"], 180.0)

    def test_chart_series_for_runs_orders_chronologically(self) -> None:
        runs = []
        for ttft in (100.0, 80.0):
            runs.append(
                BenchmarkRun.objects.create(
                    target_type="INSTANCE",
                    instance=self.instance,
                    endpoint_url="http://127.0.0.1:11446/v1",
                    params=self.params,
                    status="COMPLETED",
                    results={
                        "results": [
                            {
                                "summary": {
                                    "scenario": "medium_conc4",
                                    "ttft_p50_ms": ttft,
                                    "latency_p50_ms": 500,
                                    "latency_p95_ms": 600,
                                    "aggregate_tps": 100,
                                },
                            },
                        ],
                    },
                ),
            )
        series = chart_series_for_runs(runs, scenario="medium_conc4")
        self.assertEqual(len(series["labels"]), 2)
        self.assertEqual(series["datasets"]["ttft_p50_ms"], [100.0, 80.0])

    def test_find_comparison_candidates_pairs_same_preset(self) -> None:
        mlx_run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        external_run = BenchmarkRun.objects.create(
            target_type="ENDPOINT",
            endpoint_url="http://127.0.0.1:11434/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        pairs = find_comparison_candidates(BenchmarkRun.objects.all())
        self.assertEqual(pairs, [(mlx_run, external_run)])

    def test_build_comparison_snapshot_contains_both_runs(self) -> None:
        mlx_run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        external_run = BenchmarkRun.objects.create(
            target_type="ENDPOINT",
            endpoint_url="http://127.0.0.1:11434/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results("medium_conc1"),
        )
        snapshot = build_comparison_snapshot(mlx_run, external_run)
        self.assertEqual(snapshot["runs"]["left"]["id"], mlx_run.id)
        self.assertEqual(snapshot["runs"]["right"]["id"], external_run.id)
        self.assertEqual(len(snapshot["scenario_alignment"]), 2)

    def test_summary_for_scenario_falls_back_to_prefix_match(self) -> None:
        run = BenchmarkRun.objects.create(
            target_type="ENDPOINT",
            endpoint_url="http://127.0.0.1:11434/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results("medium_conc1"),
        )
        summary = summary_for_scenario(run, "medium_conc4")
        self.assertEqual(summary["scenario"], "medium_conc1")

    def test_comparison_rows_aligns_scenarios(self) -> None:
        run_a = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results("medium_conc4"),
        )
        run_b = BenchmarkRun.objects.create(
            target_type="ENDPOINT",
            endpoint_url="http://127.0.0.1:11434/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results("medium_conc1"),
        )
        rows = comparison_rows(run_a, run_b)
        scenarios = {row["scenario"] for row in rows}
        self.assertEqual(scenarios, {"medium_conc1", "medium_conc4"})
