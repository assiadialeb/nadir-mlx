"""Quality benchmark orchestration (lm_eval + platform suites)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from django.utils import timezone

from orchestrator.models import BenchmarkRun
from orchestrator.security_utils import (
    build_validated_http_url,
    public_error_message,
    validated_benchmark_artifact_path,
    validated_launch_port,
    validate_benchmark_endpoint_host,
    validate_outbound_http_host,
)
from orchestrator.vendor.lm_eval_runner import run_lm_eval
from orchestrator.vendor.qualitybench import (
    run_platform_suites,
    summarize_platform_metrics,
)

logger = logging.getLogger(__name__)

BENCHMARK_QUALITY_TIMEOUT_SECONDS = 7200


def _quality_output_dir(run_id: int) -> Path:
    return validated_benchmark_artifact_path(run_id, f"quality_{run_id}")


def _quality_artifact_path(run_id: int) -> Path:
    return validated_benchmark_artifact_path(run_id, f"bench_{run_id}_quality.json")


def _mark_failed(run: BenchmarkRun, message: str) -> None:
    run.status = "FAILED"
    run.error_message = message[:2000]
    run.completed_at = timezone.now()
    run.save(update_fields=["status", "error_message", "completed_at"])


def _format_error(exc: Exception) -> str:
    """Return an operator-safe error string without stack traces or paths."""
    return public_error_message(exc, fallback="Quality benchmark failed.")


def summarize_industry_metrics(industry: dict[str, Any]) -> dict[str, float | None]:
    """Extract headline percentages from lm_eval task metrics."""
    if industry.get("skipped") or industry.get("failed"):
        return {}
    metrics: dict[str, float | None] = {}
    for task_name, task_metrics in (industry.get("tasks") or {}).items():
        if task_metrics.get("acc") is not None:
            metrics[f"{task_name}_acc"] = round(float(task_metrics["acc"]) * 100, 1)
        if task_metrics.get("exact_match") is not None:
            metrics[f"{task_name}_exact_match"] = round(
                float(task_metrics["exact_match"]) * 100,
                1,
            )
        strict_acc = task_metrics.get("prompt_level_strict_acc")
        if strict_acc is not None:
            metrics[f"{task_name}_strict_acc"] = round(float(strict_acc) * 100, 1)
    return metrics


def build_quality_results(
    industry: dict[str, Any],
    platform: dict[str, Any],
    *,
    preset: str,
) -> dict[str, Any]:
    """Merge industry and platform payloads into a stored results document."""
    industry_summary = summarize_industry_metrics(industry)
    platform_summary = summarize_platform_metrics(platform)
    return {
        "benchmark_kind": "QUALITY",
        "preset": preset,
        "industry": industry,
        "platform": platform,
        "metrics": {**industry_summary, **platform_summary},
    }


def _run_industry_phase(
    host: str,
    port: int,
    model: str,
    output_dir: Path,
    *,
    preset: str,
) -> dict[str, Any]:
    """Run lm_eval; never raise — record skip/failure in the returned dict."""
    try:
        return run_lm_eval(
            host,
            port,
            model,
            output_dir,
            preset=preset,
            timeout_seconds=BENCHMARK_QUALITY_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.exception("Industry quality benchmark failed for model %s", model)
        return {
            "skipped": True,
            "failed": True,
            "source": "lm_eval",
            "reason": _format_error(exc),
            "tasks": {},
        }


def _run_platform_phase(host: str, port: int, model: str) -> dict[str, Any]:
    """Run Nadir platform suites; never raise."""
    try:
        return run_platform_suites(host, port, model)
    except Exception as exc:
        logger.exception("Platform quality benchmark failed for model %s", model)
        return {
            "failed": True,
            "source": "qualitybench",
            "error": _format_error(exc),
            "suites": {},
        }


def _phase_warnings(industry: dict[str, Any], platform: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if industry.get("failed"):
        warnings.append(str(industry.get("reason") or "Industry benchmark failed."))
    elif industry.get("skipped"):
        reason = str(industry.get("reason") or "").strip()
        if reason:
            warnings.append(reason)
    if platform.get("failed"):
        warnings.append(str(platform.get("error") or "Platform benchmark failed."))
    return warnings


def _phase_errors(industry: dict[str, Any], platform: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if industry.get("failed") and platform.get("failed"):
        if industry.get("failed"):
            errors.append(str(industry.get("reason") or "Industry benchmark failed."))
        if platform.get("failed"):
            errors.append(str(platform.get("error") or "Platform benchmark failed."))
    return errors


def execute_quality_benchmark(run: BenchmarkRun) -> None:
    """Run quality suites synchronously and persist results on the run."""
    params = run.params or {}
    raw_host = str(params["host"])
    if run.target_type == "ENDPOINT":
        safe_host = validate_benchmark_endpoint_host(raw_host)
    else:
        safe_host = validate_outbound_http_host(raw_host)
    safe_port = validated_launch_port(int(params["port"]))
    model = run.model_id or ""
    preset = str(params.get("quality_preset", "industry_lite"))

    if not model:
        _mark_failed(run, "Model ID is required for quality benchmarks.")
        return

    try:
        run.status = "RUNNING"
        run.save(update_fields=["status"])

        output_dir = _quality_output_dir(run.id)
        industry = _run_industry_phase(safe_host, safe_port, model, output_dir, preset=preset)
        platform = _run_platform_phase(safe_host, safe_port, model)
        phase_errors = _phase_errors(industry, platform)
        phase_warnings = _phase_warnings(industry, platform)

        if phase_errors:
            _mark_failed(run, "; ".join(phase_errors)[:2000])
            return

        results = build_quality_results(industry, platform, preset=preset)
        if phase_warnings:
            results["warnings"] = phase_warnings

        artifact_path = _quality_artifact_path(run.id)
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

        run.results = results
        run.status = "COMPLETED"
        run.error_message = ""
        run.completed_at = timezone.now()
        run.save(update_fields=["results", "status", "error_message", "completed_at"])
    except Exception as exc:
        logger.exception("Quality benchmark failed for run #%s", run.id)
        run.refresh_from_db()
        _mark_failed(run, _format_error(exc))
