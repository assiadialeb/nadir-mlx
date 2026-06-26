"""Tests for benchmark history and compare views."""

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from orchestrator.models import BenchmarkRun, InferenceInstance


class BenchmarkViewsTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user(username="bench", password="bench-pass")
        self.client.login(username="bench", password="bench-pass")
        self.instance = InferenceInstance.objects.create(
            model_name="gemma-test",
            port=11446,
            launch_mode="TEXT",
            status="RUNNING",
            pid=1234,
        )
        self.params = {"categories": ["medium"], "concurrency": [1, 4], "num_requests": 20}
        self.results = {
            "results": [
                {
                    "summary": {
                        "scenario": "medium_conc4",
                        "ttft_p50_ms": 100,
                        "latency_p50_ms": 500,
                        "latency_p95_ms": 600,
                        "aggregate_tps": 120,
                    },
                },
            ],
        }

    def test_benchmark_history_view_requires_login(self) -> None:
        anonymous = Client()
        response = anonymous.get(reverse("benchmark_history"))
        self.assertEqual(response.status_code, 302)

    def test_benchmark_history_view_renders_filtered_runs(self) -> None:
        BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results=self.results,
        )
        response = self.client.get(
            reverse("benchmark_history"),
            {"model": "gemma", "launch_mode": "TEXT"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Benchmark history")
        self.assertContains(response, "gemma-test")

    @override_settings(DEBUG=False, NADIR_BENCHMARK_ENDPOINT_ENABLED=False)
    def test_benchmark_view_hides_custom_endpoint_in_production(self) -> None:
        response = self.client.get(reverse("benchmark"))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'value="ENDPOINT"')
        self.assertContains(response, "disabled in this environment")

    @override_settings(DEBUG=True, NADIR_BENCHMARK_ENDPOINT_ENABLED=True)
    def test_benchmark_view_shows_custom_endpoint_in_debug(self) -> None:
        response = self.client.get(reverse("benchmark"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'value="ENDPOINT"')

    @override_settings(DEBUG=False, NADIR_BENCHMARK_ENDPOINT_ENABLED=False)
    def test_start_benchmark_rejects_endpoint_post_in_production(self) -> None:
        response = self.client.post(
            reverse("start_benchmark"),
            {
                "target_type": "ENDPOINT",
                "endpoint_host": "localhost",
                "endpoint_port": "11434",
                "num_requests": "5",
                "concurrency": "1",
                "categories": "medium",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(BenchmarkRun.objects.count(), 0)

    def test_benchmark_compare_export_returns_json_attachment(self) -> None:
        mlx_run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results=self.results,
        )
        external_run = BenchmarkRun.objects.create(
            target_type="ENDPOINT",
            endpoint_url="http://127.0.0.1:11434/v1",
            params=self.params,
            status="COMPLETED",
            results=self.results,
        )
        response = self.client.get(
            reverse("benchmark_compare_export"),
            {"run_a": mlx_run.id, "run_b": external_run.id},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response["Content-Type"].startswith("application/json"))
        self.assertEqual(response["X-Content-Type-Options"], "nosniff")
        self.assertIn("attachment", response["Content-Disposition"])
        self.assertIn(b"scenario_alignment", response.content)

    def test_benchmark_compare_view_renders_pair(self) -> None:
        mlx_run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params=self.params,
            status="COMPLETED",
            results=self.results,
        )
        external_run = BenchmarkRun.objects.create(
            target_type="ENDPOINT",
            endpoint_url="http://127.0.0.1:11434/v1",
            params=self.params,
            status="COMPLETED",
            results=self.results,
        )
        response = self.client.get(
            reverse("benchmark_compare"),
            {"run_a": mlx_run.id, "run_b": external_run.id},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Side-by-side scenarios")
        self.assertContains(response, "Export JSON snapshot")
