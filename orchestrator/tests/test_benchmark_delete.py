"""Tests for benchmark run deletion."""

from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from orchestrator.benchmark_service import delete_benchmark_run
from orchestrator.models import BenchmarkRun, InferenceInstance


@override_settings(LOGS_DIR="/tmp/mlx-bench-delete-tests")
class BenchmarkDeleteServiceTests(TestCase):
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

    def test_delete_benchmark_run_removes_db_row_and_json(self) -> None:
        run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params={"categories": ["medium"], "concurrency": [1], "num_requests": 5},
            status="COMPLETED",
        )
        artifact = self.logs_dir / "benchmarks" / f"bench_{run.id}.json"
        artifact.write_text("{}", encoding="utf-8")

        delete_benchmark_run(run.id)

        self.assertFalse(BenchmarkRun.objects.filter(id=run.id).exists())
        self.assertFalse(artifact.exists())

    def test_delete_benchmark_run_rejects_running(self) -> None:
        run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params={},
            status="RUNNING",
        )
        with self.assertRaisesRegex(ValueError, "running"):
            delete_benchmark_run(run.id)
        self.assertTrue(BenchmarkRun.objects.filter(id=run.id).exists())


@override_settings(LOGS_DIR="/tmp/mlx-bench-delete-view-tests")
class BenchmarkDeleteViewTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user(username="bench-del", password="bench-del-pass")
        self.client.login(username="bench-del", password="bench-del-pass")
        self.logs_dir = Path(settings.LOGS_DIR)
        (self.logs_dir / "benchmarks").mkdir(parents=True, exist_ok=True)
        self.instance = InferenceInstance.objects.create(
            model_name="gemma-test",
            port=11446,
            launch_mode="TEXT",
            status="RUNNING",
            pid=1234,
        )

    def test_delete_benchmark_view_removes_run_and_redirects(self) -> None:
        run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=self.instance,
            endpoint_url="http://127.0.0.1:11446/v1",
            params={},
            status="COMPLETED",
        )
        response = self.client.post(
            reverse("delete_benchmark", args=[run.id]),
            {"next": reverse("benchmark_history")},
        )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(BenchmarkRun.objects.filter(id=run.id).exists())

    def test_delete_benchmark_view_rejects_get(self) -> None:
        run = BenchmarkRun.objects.create(
            target_type="ENDPOINT",
            endpoint_url="http://127.0.0.1:11434/v1",
            params={},
            status="COMPLETED",
        )
        response = self.client.get(reverse("delete_benchmark", args=[run.id]))
        self.assertEqual(response.status_code, 302)
        self.assertTrue(BenchmarkRun.objects.filter(id=run.id).exists())
