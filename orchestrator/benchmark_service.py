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

from .models import BenchmarkRun, InferenceInstance
from .security_utils import (
    public_error_message,
    safe_path_under_root,
    safe_positive_int,
    validate_outbound_http_host,
)

LLMBENCH_SCRIPT = Path(__file__).resolve().parent / "vendor" / "llmbench.py"
BENCHMARK_TIMEOUT_SECONDS = 3600


def _benchmark_output_path(run_id: int) -> Path:
    safe_positive_int(run_id, field_name="benchmark run id")
    benchmarks_dir = Path(settings.LOGS_DIR) / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    return safe_path_under_root(benchmarks_dir, f"bench_{run_id}.json")


def _remove_benchmark_artifacts(run: BenchmarkRun) -> None:
    _benchmark_output_path(run.id).unlink(missing_ok=True)


def delete_benchmark_run(run_id: int) -> None:
    """Delete a single benchmark run and its JSON artifact."""
    run = BenchmarkRun.objects.filter(id=run_id).first()
    if not run:
        raise ValueError("Benchmark run not found.")
    if run.status in ("PENDING", "RUNNING"):
        raise ValueError("Cannot delete a benchmark while it is running.")
    _remove_benchmark_artifacts(run)
    run.delete()


def delete_benchmark_runs_for_model(folder_name: str) -> int:
    """Delete benchmark DB rows and JSON artifacts linked to a model folder."""
    runs = BenchmarkRun.objects.filter(
        Q(instance__model_name=folder_name) | Q(model_id=folder_name),
    )
    removed = 0
    for run in runs:
        _remove_benchmark_artifacts(run)
        run.delete()
        removed += 1
    return removed


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


def _benchmark_thread(run_id: int) -> None:
    run = BenchmarkRun.objects.get(id=run_id)
    output_path = _benchmark_output_path(run_id)

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
        return "localhost", instance.port, instance

    if not host or not host.strip():
        raise ValueError("Host is required for a custom endpoint.")
    if port is None:
        raise ValueError("Port is required for a custom endpoint.")

    safe_host = validate_outbound_http_host(host.strip())
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
    """Return the chat model id for a local mlx_lm / mlx_vlm instance."""
    if instance.launch_mode == "EMBEDDING":
        return instance.model_name
    return "default_model"


def resolve_benchmark_model_id(
    host: str,
    port: int,
    instance: InferenceInstance | None,
    user_model_id: str,
) -> str:
    """Resolve the model ID passed to llmbench.

    Local mlx_lm / mlx_vlm instances preload one checkpoint and accept
    ``default_model`` for chat. Their ``/v1/models`` response is unreliable
    (empty for some mlx_lm folders; unrelated embedding ids for some mlx_vlm
    builds). Prefer ``default_model`` for INSTANCE targets before probing the API.
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


def start_benchmark(
    target_type: str,
    instance_id: int | None,
    host: str | None,
    port: int | None,
    model_id: str,
    params: dict[str, Any],
) -> BenchmarkRun:
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
        target_type=target_type,
        instance=instance,
        endpoint_url=endpoint_url,
        model_id=resolved_model_id,
        params=full_params,
        status="PENDING",
    )

    thread = threading.Thread(
        target=_benchmark_thread,
        args=(run.id,),
        daemon=True,
    )
    thread.start()
    return run


def parse_benchmark_form(data: dict[str, str]) -> dict[str, Any]:
    target_type = data.get("target_type", "INSTANCE")
    if target_type not in {"INSTANCE", "ENDPOINT"}:
        raise ValueError("Invalid target type.")

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

    return {
        "target_type": target_type,
        "instance_id": instance_id,
        "host": data.get("endpoint_host", "localhost").strip(),
        "port": port,
        "model_id": data.get("model_id", "").strip(),
        "params": {
            "concurrency": _parse_concurrency(data.get("concurrency", "1,4")),
            "categories": categories,
            "num_requests": num_requests,
            "temperature": 0.0,
        },
    }
