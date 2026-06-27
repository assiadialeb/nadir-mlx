"""Run HTTP benchmarks against MLX or OpenAI-compatible endpoints via llmbench.py."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import httpx
from django.conf import settings
from django.db.models import Q
from django.utils import timezone

from .gateway_aliases import instance_gateway_alias
from .models import BenchmarkRun, InferenceInstance
from .quality_benchmark_service import execute_quality_benchmark
from .security_utils import (
    benchmark_endpoint_enabled,
    public_error_message,
    safe_path_under_root,
    safe_positive_int,
    validate_benchmark_endpoint_host,
    validate_outbound_http_host,
)

LLMBENCH_SCRIPT = Path(__file__).resolve().parent / "vendor" / "llmbench.py"
BENCHMARK_TIMEOUT_SECONDS = 3600
BENCHMARK_MAX_REQUESTS_PER_SCENARIO = 500
VALID_BENCHMARK_KINDS = frozenset({"PERF", "QUALITY", "COMPLETE"})
BENCHMARK_RUN_ID_FIELD = "benchmark run id"


def _benchmark_output_path(run_id: int) -> Path:
    safe_positive_int(run_id, field_name=BENCHMARK_RUN_ID_FIELD)
    benchmarks_dir = Path(settings.LOGS_DIR) / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    return safe_path_under_root(benchmarks_dir, f"bench_{run_id}.json")


def _quality_artifact_path(run_id: int) -> Path:
    safe_positive_int(run_id, field_name=BENCHMARK_RUN_ID_FIELD)
    benchmarks_dir = Path(settings.LOGS_DIR) / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    return safe_path_under_root(benchmarks_dir, f"bench_{run_id}_quality.json")


def _quality_output_dir(run_id: int) -> Path:
    safe_positive_int(run_id, field_name=BENCHMARK_RUN_ID_FIELD)
    benchmarks_dir = Path(settings.LOGS_DIR) / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    return safe_path_under_root(benchmarks_dir, f"quality_{run_id}")


def _remove_benchmark_artifacts(run: BenchmarkRun) -> None:
    _benchmark_output_path(run.id).unlink(missing_ok=True)
    _quality_artifact_path(run.id).unlink(missing_ok=True)
    quality_dir = _quality_output_dir(run.id)
    if quality_dir.is_dir():
        for path in quality_dir.rglob("*"):
            if path.is_file():
                path.unlink()
        for path in sorted(quality_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        quality_dir.rmdir()


def delete_benchmark_run(run_id: int) -> None:
    """Delete a single benchmark run and its JSON artifact."""
    run = BenchmarkRun.objects.filter(id=run_id).first()
    if not run:
        raise ValueError("Benchmark run not found.")
    if run.status in ("PENDING", "RUNNING"):
        raise ValueError("Cannot delete a benchmark while it is running.")
    for child in run.child_runs.all():
        _remove_benchmark_artifacts(child)
        child.delete()
    _remove_benchmark_artifacts(run)
    run.delete()


def delete_benchmark_runs_for_model(folder_name: str) -> int:
    """Delete benchmark DB rows and JSON artifacts linked to a model folder."""
    runs = BenchmarkRun.objects.filter(
        Q(instance__model_name=folder_name) | Q(model_id=folder_name),
        parent_run__isnull=True,
    )
    run_ids = list(runs.values_list("id", flat=True))
    for run_id in run_ids:
        delete_benchmark_run(run_id)
    return len(run_ids)


def _build_command(run: BenchmarkRun, output_path: Path) -> list[str]:
    params = run.params or {}
    cmd = [
        sys.executable,
        str(LLMBENCH_SCRIPT),
        "--host",
        str(params["host"]),
        "--port",
        str(params["port"]),
        "--num-requests",
        str(params.get("num_requests", 5)),
        "--temperature",
        str(params.get("temperature", 0.0)),
        "--output",
        str(output_path),
    ]

    for level in params.get("concurrency", [1, 4]):
        cmd.extend(["--concurrency", str(level)])

    for category in params.get("categories", ["medium"]):
        cmd.extend(["--categories", category])

    if run.model_id:
        cmd.extend(["--model", run.model_id])

    return cmd


def _mark_failed(run: BenchmarkRun, message: str) -> None:
    run.status = "FAILED"
    run.error_message = message[:2000]
    run.completed_at = timezone.now()
    run.save(update_fields=["status", "error_message", "completed_at"])


def execute_perf_benchmark(run: BenchmarkRun) -> None:
    """Run llmbench synchronously and persist results on the run."""
    output_path = _benchmark_output_path(run.id)

    try:
        run.status = "RUNNING"
        run.save(update_fields=["status"])

        result = subprocess.run(
            _build_command(run, output_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=BENCHMARK_TIMEOUT_SECONDS,
        )

        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "Benchmark failed").strip()
            _mark_failed(run, stderr)
            return

        if not output_path.exists():
            _mark_failed(run, "Benchmark finished but output file is missing.")
            return

        with open(output_path, encoding="utf-8") as handle:
            run.results = json.load(handle)

        run.status = "COMPLETED"
        run.error_message = ""
        run.completed_at = timezone.now()
        run.save(update_fields=["results", "status", "error_message", "completed_at"])
    except subprocess.TimeoutExpired:
        run.refresh_from_db()
        _mark_failed(run, "Benchmark timed out after 1 hour.")
    except Exception as exc:
        run.refresh_from_db()
        _mark_failed(run, public_error_message(exc, fallback="Benchmark failed."))


def _benchmark_thread(run_id: int) -> None:
    run = BenchmarkRun.objects.get(id=run_id)
    execute_perf_benchmark(run)


def _quality_thread(run_id: int) -> None:
    run = BenchmarkRun.objects.get(id=run_id)
    execute_quality_benchmark(run)


def _perf_headline(run: BenchmarkRun) -> dict[str, Any]:
    summary = run.summaries[0] if run.summaries else {}
    return {
        "ttft_p50_ms": summary.get("ttft_p50_ms"),
        "aggregate_tps": summary.get("aggregate_tps"),
        "scenario": summary.get("scenario"),
    }


def _create_child_run(parent: BenchmarkRun, benchmark_kind: str) -> BenchmarkRun:
    return BenchmarkRun.objects.create(
        parent_run=parent,
        benchmark_kind=benchmark_kind,
        target_type=parent.target_type,
        instance=parent.instance,
        endpoint_url=parent.endpoint_url,
        model_id=parent.model_id,
        params=parent.params,
        status="PENDING",
    )


def _complete_thread(parent_id: int) -> None:
    parent = BenchmarkRun.objects.get(id=parent_id)
    parent.status = "RUNNING"
    parent.save(update_fields=["status"])

    perf_run = _create_child_run(parent, "PERF")
    execute_perf_benchmark(perf_run)
    perf_run.refresh_from_db()
    if perf_run.status != "COMPLETED":
        _mark_failed(parent, f"Performance phase failed: {perf_run.error_message}")
        return

    quality_run = _create_child_run(parent, "QUALITY")
    execute_quality_benchmark(quality_run)
    quality_run.refresh_from_db()
    if quality_run.status != "COMPLETED":
        _mark_failed(parent, f"Quality phase failed: {quality_run.error_message}")
        return

    parent.results = {
        "benchmark_kind": "COMPLETE",
        "perf_run_id": perf_run.id,
        "quality_run_id": quality_run.id,
        "perf_summary": _perf_headline(perf_run),
        "perf_summaries": perf_run.summaries,
        "quality_summary": quality_run.results.get("metrics", {}) if quality_run.results else {},
        "quality_results": quality_run.results or {},
    }
    parent.status = "COMPLETED"
    parent.error_message = ""
    parent.completed_at = timezone.now()
    parent.save(update_fields=["results", "status", "error_message", "completed_at"])


def _gateway_benchmark_endpoint() -> tuple[str, int]:
    """Host/port for INSTANCE benchmarks (always via the Nadir gateway)."""
    raw_host = str(settings.NADIR_GATEWAY_HOST or "127.0.0.1").strip()
    connect_host = "127.0.0.1" if raw_host == "0.0.0.0" else raw_host
    safe_host = validate_outbound_http_host(connect_host)
    return safe_host, int(settings.NADIR_GATEWAY_PORT)


def _resolve_target(
    target_type: str,
    instance_id: int | None,
    host: str | None,
    port: int | None,
) -> tuple[str, int, InferenceInstance | None]:
    if target_type == "INSTANCE":
        if instance_id is None:
            raise ValueError("Select a running MLX instance.")
        instance = InferenceInstance.objects.get(id=instance_id)
        if instance.status != "RUNNING":
            raise ValueError("The selected instance must be RUNNING.")
        if instance.launch_mode not in ("TEXT", "MULTIMODAL"):
            raise ValueError("Benchmark is only available for TEXT or MULTIMODAL instances.")
        gateway_host, gateway_port = _gateway_benchmark_endpoint()
        return gateway_host, gateway_port, instance

    if not host or not host.strip():
        raise ValueError("Host is required for a custom endpoint.")
    if port is None:
        raise ValueError("Port is required for a custom endpoint.")

    safe_host = validate_benchmark_endpoint_host(host.strip())
    if port < 1 or port > 65535:
        raise ValueError("Port must be between 1 and 65535.")
    return safe_host, port, None


def _parse_concurrency(raw_value: str) -> list[int]:
    levels = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not levels:
        raise ValueError("Provide at least one concurrency level.")
    if any(level < 1 for level in levels):
        raise ValueError("Concurrency levels must be positive integers.")
    return levels


def _default_model_id_for_instance(instance: InferenceInstance) -> str:
    """Return the gateway alias used as the OpenAI model id for this instance."""
    return instance_gateway_alias(instance)


def resolve_benchmark_model_id(
    host: str,
    port: int,
    instance: InferenceInstance | None,
    user_model_id: str,
) -> str:
    """Resolve the model ID passed to llmbench.

  INSTANCE targets are benchmarked through the Nadir gateway; use the gateway
  alias (``server_config.model_id`` or folder name). ENDPOINT targets probe
  ``/v1/models`` when no model id is provided.
    """
    cleaned = user_model_id.strip()
    if cleaned:
        return cleaned

    if instance is not None and instance.launch_mode in ("TEXT", "MULTIMODAL"):
        return _default_model_id_for_instance(instance)

    safe_host = validate_outbound_http_host(host)
    base_url = f"http://{safe_host}:{port}"
    try:
        response = httpx.get(f"{base_url}/v1/models", timeout=10)
        response.raise_for_status()
        models = response.json().get("data", [])
        if models:
            return str(models[0]["id"])
    except Exception:
        pass

    if instance is not None:
        return _default_model_id_for_instance(instance)

    raise ValueError(
        "Could not auto-detect model ID from /v1/models. "
        "Enter the model name manually (e.g. llama3 for Ollama)."
    )


def _start_benchmark_thread(run: BenchmarkRun) -> None:
    if run.benchmark_kind == "QUALITY":
        target = _quality_thread
    elif run.benchmark_kind == "COMPLETE":
        target = _complete_thread
    else:
        target = _benchmark_thread

    thread = threading.Thread(target=target, args=(run.id,), daemon=True)
    thread.start()


def start_benchmark(
    target_type: str,
    instance_id: int | None,
    host: str | None,
    port: int | None,
    model_id: str,
    params: dict[str, Any],
    *,
    benchmark_kind: str = "PERF",
) -> BenchmarkRun:
    if benchmark_kind not in VALID_BENCHMARK_KINDS:
        raise ValueError("Invalid benchmark kind.")

    resolved_host, resolved_port, instance = _resolve_target(
        target_type,
        instance_id,
        host,
        port,
    )

    full_params = {
        **params,
        "host": resolved_host,
        "port": resolved_port,
    }
    endpoint_url = f"http://{resolved_host}:{resolved_port}"
    resolved_model_id = resolve_benchmark_model_id(
        resolved_host,
        resolved_port,
        instance,
        model_id,
    )

    run = BenchmarkRun.objects.create(
        benchmark_kind=benchmark_kind,
        target_type=target_type,
        instance=instance,
        endpoint_url=endpoint_url,
        model_id=resolved_model_id,
        params=full_params,
        status="PENDING",
    )

    _start_benchmark_thread(run)
    return run


def _validate_endpoint_target(target_type: str, host: str) -> None:
    if target_type != "ENDPOINT":
        return
    if not benchmark_endpoint_enabled():
        raise ValueError(
            "Custom endpoint benchmarks are disabled. Use a running MLX instance."
        )
    validate_benchmark_endpoint_host(host)


def parse_benchmark_form(data: dict[str, str]) -> dict[str, Any]:
    target_type = data.get("target_type", "INSTANCE")
    if target_type not in {"INSTANCE", "ENDPOINT"}:
        raise ValueError("Invalid target type.")

    benchmark_kind = str(data.get("benchmark_kind", "PERF")).strip().upper()
    if benchmark_kind not in VALID_BENCHMARK_KINDS:
        raise ValueError("Invalid benchmark kind.")

    endpoint_host = data.get("endpoint_host", "localhost").strip()
    _validate_endpoint_target(target_type, endpoint_host)

    instance_id_raw = data.get("instance_id", "").strip()
    instance_id = int(instance_id_raw) if instance_id_raw else None

    port_raw = data.get("endpoint_port", "").strip()
    port = int(port_raw) if port_raw else None

    categories = data.getlist("categories") if hasattr(data, "getlist") else []
    if not categories:
        categories = ["medium"]

    num_requests = int(data.get("num_requests", "5"))
    if num_requests < 1:
        raise ValueError("Number of requests must be at least 1.")
    if num_requests > BENCHMARK_MAX_REQUESTS_PER_SCENARIO:
        raise ValueError(
            f"Number of requests cannot exceed {BENCHMARK_MAX_REQUESTS_PER_SCENARIO} per scenario."
        )

    quality_preset = str(data.get("quality_preset", "industry_lite")).strip() or "industry_lite"

    return {
        "target_type": target_type,
        "benchmark_kind": benchmark_kind,
        "instance_id": instance_id,
        "host": endpoint_host,
        "port": port,
        "model_id": data.get("model_id", "").strip(),
        "params": {
            "concurrency": _parse_concurrency(data.get("concurrency", "1,4")),
            "categories": categories,
            "num_requests": num_requests,
            "temperature": 0.0,
            "quality_preset": quality_preset,
        },
    }
