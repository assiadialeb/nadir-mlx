"""Installed model deletion orchestration (MLX-1)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings
from django.db.models import Q

from orchestrator.benchmark_service import delete_benchmark_runs_for_model
from orchestrator.model_utils import resolve_log_file_path, resolve_model_dir, validate_model_folder_name
from orchestrator.models import BenchmarkRun, InferenceInstance, ModelDownload
from orchestrator.server_manager import delete_instance


@dataclass(frozen=True)
class DeleteModelResult:
    folder_name: str
    instances_removed: int
    benchmark_runs_removed: int
    log_files_removed: int
    download_records_removed: int
    model_directory_removed: bool


def _is_model_downloading(folder_name: str, model_dir: Path) -> bool:
    model_dir_str = str(model_dir)
    return ModelDownload.objects.filter(status="DOWNLOADING").filter(
        Q(local_path=model_dir_str) | Q(repo_id__endswith=f"/{folder_name}"),
    ).exists()


def _has_active_benchmarks(folder_name: str) -> bool:
    return BenchmarkRun.objects.filter(
        status__in=("PENDING", "RUNNING"),
    ).filter(
        Q(instance__model_name=folder_name) | Q(model_id=folder_name),
    ).exists()


def _delete_model_download_records(folder_name: str, model_dir: Path) -> int:
    model_dir_str = str(model_dir)
    queryset = ModelDownload.objects.filter(
        Q(local_path=model_dir_str) | Q(repo_id__endswith=f"/{folder_name}"),
    )
    count = queryset.count()
    queryset.delete()
    return count


def _remove_instance_logs(folder_name: str) -> int:
    logs_root = Path(settings.LOGS_DIR).resolve()
    removed = 0
    for log_path in logs_root.glob(f"{folder_name}_*.log"):
        if log_path.is_file():
            log_path.unlink(missing_ok=True)
            removed += 1
    return removed


def delete_installed_model(folder_name: str) -> DeleteModelResult:
    """Stop related servers, purge benchmarks/logs, and remove model files."""
    name = validate_model_folder_name(folder_name)
    model_dir = resolve_model_dir(name)
    folder_exists = model_dir.is_dir()

    if _is_model_downloading(name, model_dir):
        raise ValueError("Cannot delete a model while it is downloading.")

    if _has_active_benchmarks(name):
        raise ValueError("Cannot delete a model while a benchmark run is active.")

    has_instances = InferenceInstance.objects.filter(model_name=name).exists()
    has_downloads = ModelDownload.objects.filter(
        Q(local_path=str(model_dir)) | Q(repo_id__endswith=f"/{name}"),
    ).exists()
    has_benchmarks = BenchmarkRun.objects.filter(
        Q(instance__model_name=name) | Q(model_id=name),
    ).exists()

    if not folder_exists and not has_instances and not has_downloads and not has_benchmarks:
        raise ValueError("Model not found.")

    instances = list(InferenceInstance.objects.filter(model_name=name))
    log_files_removed = 0
    for instance in instances:
        delete_instance(instance)
        log_path = resolve_log_file_path(name, instance.port)
        if log_path.is_file():
            log_path.unlink(missing_ok=True)
            log_files_removed += 1

    log_files_removed += _remove_instance_logs(name)
    benchmark_runs_removed = delete_benchmark_runs_for_model(name)
    download_records_removed = _delete_model_download_records(name, model_dir)

    model_directory_removed = False
    if folder_exists:
        shutil.rmtree(model_dir)
        model_directory_removed = True

    return DeleteModelResult(
        folder_name=name,
        instances_removed=len(instances),
        benchmark_runs_removed=benchmark_runs_removed,
        log_files_removed=log_files_removed,
        download_records_removed=download_records_removed,
        model_directory_removed=model_directory_removed,
    )
