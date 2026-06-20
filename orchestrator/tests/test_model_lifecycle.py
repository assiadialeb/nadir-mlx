"""Tests for installed model deletion (MLX-1)."""

from __future__ import annotations

from pathlib import Path

from django.test import TestCase, override_settings

from orchestrator.model_lifecycle import delete_installed_model
from orchestrator.models import BenchmarkRun, InferenceInstance, ModelDownload


@override_settings(MODELS_DIR="/tmp/mlx-test-models", LOGS_DIR="/tmp/mlx-test-logs")
class DeleteInstalledModelTests(TestCase):
    def setUp(self) -> None:
        self.models_dir = Path("/tmp/mlx-test-models")
        self.logs_dir = Path("/tmp/mlx-test-logs")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _create_model_dir(self, name: str) -> Path:
        model_dir = self.models_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"weights")
        return model_dir

    def test_delete_installed_model_removes_files_instances_and_benchmarks(self) -> None:
        model_name = "Kokoro-82M-6bit"
        self._create_model_dir(model_name)
        instance = InferenceInstance.objects.create(
            model_name=model_name,
            port=11444,
            status="STOPPED",
        )
        log_path = self.logs_dir / f"{model_name}_11444.log"
        log_path.write_text("log line", encoding="utf-8")

        bench_dir = self.logs_dir / "benchmarks"
        bench_dir.mkdir(parents=True, exist_ok=True)
        run = BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=instance,
            endpoint_url="http://127.0.0.1:11444/v1",
            model_id=model_name,
            status="COMPLETED",
        )
        bench_file = bench_dir / f"bench_{run.id}.json"
        bench_file.write_text("{}", encoding="utf-8")

        ModelDownload.objects.create(
            repo_id=f"mlx-community/{model_name}",
            local_path=str(self.models_dir / model_name),
            status="COMPLETED",
        )

        result = delete_installed_model(model_name)

        self.assertEqual(result.instances_removed, 1)
        self.assertEqual(result.benchmark_runs_removed, 1)
        self.assertGreaterEqual(result.log_files_removed, 1)
        self.assertEqual(result.download_records_removed, 1)
        self.assertTrue(result.model_directory_removed)
        self.assertFalse((self.models_dir / model_name).exists())
        self.assertFalse(log_path.exists())
        self.assertFalse(bench_file.exists())
        self.assertFalse(InferenceInstance.objects.filter(model_name=model_name).exists())
        self.assertFalse(BenchmarkRun.objects.filter(model_id=model_name).exists())
        self.assertFalse(ModelDownload.objects.filter(repo_id__endswith=model_name).exists())

    def test_delete_installed_model_rejects_active_download(self) -> None:
        model_name = "whisper-small-mlx"
        self._create_model_dir(model_name)
        ModelDownload.objects.create(
            repo_id=f"mlx-community/{model_name}",
            local_path=str(self.models_dir / model_name),
            status="DOWNLOADING",
        )

        with self.assertRaisesRegex(ValueError, "downloading"):
            delete_installed_model(model_name)

        self.assertTrue((self.models_dir / model_name).exists())

    def test_delete_installed_model_rejects_active_benchmark(self) -> None:
        model_name = "Qwen3-8B-4bit"
        self._create_model_dir(model_name)
        instance = InferenceInstance.objects.create(
            model_name=model_name,
            port=11434,
            status="RUNNING",
        )
        BenchmarkRun.objects.create(
            target_type="INSTANCE",
            instance=instance,
            endpoint_url="http://127.0.0.1:11434/v1",
            model_id=model_name,
            status="RUNNING",
        )

        with self.assertRaisesRegex(ValueError, "benchmark"):
            delete_installed_model(model_name)

    def test_delete_installed_model_rejects_invalid_name(self) -> None:
        with self.assertRaises(ValueError):
            delete_installed_model("../etc/passwd")

    def test_delete_installed_model_rejects_missing_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "not found"):
            delete_installed_model("missing-model")
