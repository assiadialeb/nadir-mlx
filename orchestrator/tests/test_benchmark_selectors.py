"""Tests for benchmark history, chart, and comparison selectors."""

from django.test import TestCase

from orchestrator.benchmark_selectors import (
    benchmark_detail_phase_message,
    benchmark_endpoint_kind,
    benchmark_history_model_query,
    benchmark_preset_key,
    benchmark_run_label,
    benchmark_run_list_row,
    build_benchmark_history_query,
    build_benchmark_status_payload,
    build_comparison_snapshot,
    chart_series_for_runs,
    comparison_pair_label,
    comparison_rows,
    detail_perf_chart_payload,
    detail_quality_chart_payload,
    detail_render_ready,
    filter_benchmark_runs,
    find_comparison_candidates,
    find_draft_ab_pairs,
    format_preset_label,
    list_filter_options,
    paginate_benchmark_runs,
    parse_benchmark_history_query,
    resolve_perf_summaries,
    resolve_quality_metrics,
    runs_for_chart_filters,
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

    def test_resolve_perf_summaries_uses_child_for_complete(self) -> None:
        parent = BenchmarkRun.objects.create(
            benchmark_kind="COMPLETE",
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results={"quality_summary": {"gsm8k_exact_match": 70.0}},
        )
        perf_child = BenchmarkRun.objects.create(
            benchmark_kind="PERF",
            parent_run=parent,
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        summaries = resolve_perf_summaries(parent, perf_child)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["aggregate_tps"], 180.0)

    def test_detail_quality_chart_payload_orders_headline_metrics(self) -> None:
        payload = detail_quality_chart_payload({
            "text_platform_pass_rate": 100.0,
            "gsm8k_exact_match": 68.0,
            "ifeval_strict_acc": 79.0,
        })
        self.assertEqual(payload["labels"][0], "IFEval")
        self.assertEqual(payload["values"][2], 100.0)

    def test_detail_render_ready_complete_uses_embedded_parent_results(self) -> None:
        parent = BenchmarkRun.objects.create(
            benchmark_kind="COMPLETE",
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results={
                "perf_summaries": [{"scenario": "medium_conc1", "aggregate_tps": 42}],
                "quality_summary": {"text_platform_pass_rate": 90.0},
                "quality_results": {"metrics": {"text_platform_pass_rate": 90.0}},
            },
        )
        self.assertTrue(detail_render_ready(parent, None, None))
        self.assertEqual(len(resolve_perf_summaries(parent, None)), 1)

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
            model_id="gemma-test",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        pairs = find_comparison_candidates(BenchmarkRun.objects.all())
        self.assertEqual(pairs, [(mlx_run, external_run)])

    def test_find_comparison_candidates_pairs_nadir_instance_and_external(self) -> None:
        nadir_run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11380/v1",
            model_id="gemma-test",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        external_run = BenchmarkRun.objects.create(
            target_type="ENDPOINT",
            endpoint_url="http://127.0.0.1:11434/v1",
            model_id="gemma-test",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        pairs = find_comparison_candidates(BenchmarkRun.objects.all())
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0][0].id, pairs[0][1].id}, {nadir_run.id, external_run.id})

    def test_benchmark_history_model_query_uses_model_id_for_endpoint(self) -> None:
        run = BenchmarkRun.objects.create(
            target_type="ENDPOINT",
            endpoint_url="http://127.0.0.1:11380/v1",
            model_id="Qwen3.6-35B-A3B-4bit",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        self.assertEqual(benchmark_history_model_query(run), "Qwen3.6-35B-A3B-4bit")

    def test_benchmark_endpoint_kind_labels_instance_as_nadir(self) -> None:
        nadir_run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11380",
            model_id="qwen",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        self.assertEqual(benchmark_endpoint_kind(nadir_run, 11380), "nadir")

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

    def test_format_preset_label_includes_request_count(self) -> None:
        label = format_preset_label(self.params, benchmark_kind="PERF")
        self.assertIn("20", label)
        self.assertIn("medium", label.lower())

    def test_list_filter_options_exposes_launch_modes(self) -> None:
        options = list_filter_options()
        self.assertIn("launch_modes", options)
        self.assertIn("TEXT", options["launch_modes"])

    def test_build_benchmark_history_query_serializes_filters(self) -> None:
        query = build_benchmark_history_query(model_name="gemma", status="COMPLETED", page=2)
        self.assertIn("model=gemma", query)
        self.assertIn("status=COMPLETED", query)
        self.assertIn("page=2", query)

    def test_build_benchmark_status_payload_for_complete_run(self) -> None:
        parent = BenchmarkRun.objects.create(
            benchmark_kind="COMPLETE",
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results={
                "perf_summaries": [{"scenario": "medium_conc4", "aggregate_tps": 50}],
                "quality_summary": {"gsm8k_exact_match": 72.0},
            },
        )
        perf_child = BenchmarkRun.objects.create(
            benchmark_kind="PERF",
            parent_run=parent,
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        quality_child = BenchmarkRun.objects.create(
            benchmark_kind="QUALITY",
            parent_run=parent,
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results={"metrics": {"gsm8k_exact_match": 72.0}},
        )
        payload = build_benchmark_status_payload(parent, perf_child, quality_child)
        self.assertTrue(payload["render_ready"])
        self.assertEqual(payload["phase"]["perf_status"], "COMPLETED")
        self.assertEqual(payload["quality_metrics"]["gsm8k_exact_match"], 72.0)

    def test_benchmark_detail_phase_message_describes_perf_phase(self) -> None:
        parent = BenchmarkRun.objects.create(
            benchmark_kind="COMPLETE",
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="RUNNING",
        )
        perf_child = BenchmarkRun.objects.create(
            benchmark_kind="PERF",
            parent_run=parent,
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="RUNNING",
        )
        message = benchmark_detail_phase_message(parent, perf_child, None)
        self.assertIn("Phase 1", message)

    def test_benchmark_detail_phase_message_describes_quality_phase(self) -> None:
        parent = BenchmarkRun.objects.create(
            benchmark_kind="COMPLETE",
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="RUNNING",
        )
        perf_child = BenchmarkRun.objects.create(
            benchmark_kind="PERF",
            parent_run=parent,
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
        )
        quality_child = BenchmarkRun.objects.create(
            benchmark_kind="QUALITY",
            parent_run=parent,
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="RUNNING",
        )
        message = benchmark_detail_phase_message(parent, perf_child, quality_child)
        self.assertIn("Phase 2", message)

    def test_detail_perf_chart_payload_returns_empty_without_summaries(self) -> None:
        self.assertEqual(detail_perf_chart_payload([]), {})

    def test_runs_for_chart_filters_limits_completed_runs(self) -> None:
        BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        runs = runs_for_chart_filters({"model_name": "gemma"}, limit=5)
        self.assertEqual(len(runs), 1)

    def test_benchmark_run_label_includes_endpoint_kind(self) -> None:
        run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11380/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        label = benchmark_run_label(run, gateway_port=11380)
        self.assertIn("gemma-test", label)

    def test_comparison_pair_label_joins_run_labels(self) -> None:
        left = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        right = BenchmarkRun.objects.create(
            target_type="ENDPOINT",
            endpoint_url="http://127.0.0.1:11434/v1",
            params=self.params,
            status="COMPLETED",
            results=_sample_results(),
        )
        label = comparison_pair_label(left, right, gateway_port=11380)
        self.assertIn("vs", label)

    def test_resolve_quality_metrics_reads_child_results(self) -> None:
        parent = BenchmarkRun.objects.create(
            benchmark_kind="COMPLETE",
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results={"quality_summary": {"ifeval_strict_acc": 80.0}},
        )
        quality_child = BenchmarkRun.objects.create(
            benchmark_kind="QUALITY",
            parent_run=parent,
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results={"metrics": {"ifeval_strict_acc": 81.0}},
        )
        metrics = resolve_quality_metrics(parent, quality_child)
        self.assertEqual(metrics["ifeval_strict_acc"], 81.0)

    def test_find_draft_ab_pairs_groups_same_instance_different_draft(self) -> None:
        no_draft = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params={
                **self.params,
                "draft_profile": {},
            },
            status="COMPLETED",
            results=_sample_results(),
        )
        mtp_run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params={
                **self.params,
                "draft_profile": {"draft_kind": "mtp"},
            },
            status="COMPLETED",
            results=_sample_results(),
        )
        pairs = find_draft_ab_pairs(BenchmarkRun.objects.all())
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0][0].id, pairs[0][1].id}, {no_draft.id, mtp_run.id})
