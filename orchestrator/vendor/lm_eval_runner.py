"""Subprocess wrapper for lm-evaluation-harness (optional dependency)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from orchestrator.security_utils import (
    build_validated_http_url,
    validated_launch_port,
    validated_subprocess_model_reference,
    validate_outbound_http_host,
)

# MMLU requires loglikelihood scoring; incompatible with local-chat-completions.
INDUSTRY_LITE_TASKS = "ifeval,gsm8k"
INDUSTRY_LITE_DEPENDENCIES = ("langdetect", "immutabledict")

QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "industry_lite": {
        "tasks": INDUSTRY_LITE_TASKS,
        "limit": 100,
        "num_concurrent": 1,
        "temperature": 0.0,
    },
}


def is_lm_eval_available() -> bool:
    """Return True when the lm_eval package is importable."""
    return importlib.util.find_spec("lm_eval") is not None


def missing_industry_dependencies() -> list[str]:
    """Return import names missing for the industry_lite preset."""
    missing: list[str] = []
    for package_name in INDUSTRY_LITE_DEPENDENCIES:
        if importlib.util.find_spec(package_name) is None:
            missing.append(package_name)
    return missing


def build_lm_eval_command(
    host: str,
    port: int,
    model: str,
    output_dir: Path,
    *,
    preset: str = "industry_lite",
) -> list[str]:
    """Build argv for ``python -m lm_eval`` against a chat-completions API."""
    config = QUALITY_PRESETS.get(preset)
    if config is None:
        raise ValueError(f"Unknown quality preset: {preset}")

    safe_host = validate_outbound_http_host(host)
    safe_port = validated_launch_port(port)
    safe_model = validated_subprocess_model_reference(model)
    base_url = build_validated_http_url(safe_host, safe_port, "/v1/chat/completions")
    model_args = (
        f"model={safe_model},base_url={base_url},"
        f"num_concurrent={config['num_concurrent']},temperature={config['temperature']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    return [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "local-chat-completions",
        "--model_args",
        model_args,
        "--tasks",
        str(config["tasks"]),
        "--batch_size",
        "1",
        "--limit",
        str(config["limit"]),
        "--apply_chat_template",
        "--output_path",
        str(output_dir),
    ]


def parse_lm_eval_output(output_dir: Path) -> dict[str, Any]:
    """Parse the newest lm_eval JSON artifact under output_dir."""
    candidates = sorted(
        output_dir.glob("**/results*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise ValueError("lm_eval finished but no results JSON was found.")

    with open(candidates[0], encoding="utf-8") as handle:
        raw = json.load(handle)

    return normalize_lm_eval_results(raw)


def normalize_lm_eval_results(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract per-task metrics into a stable Nadir envelope."""
    results_block = raw.get("results", raw)
    tasks: dict[str, dict[str, float | None]] = {}

    for task_name, metrics in results_block.items():
        if not isinstance(metrics, dict):
            continue
        tasks[task_name] = {
            "acc": _as_float(metrics.get("acc,none") or metrics.get("acc")),
            "exact_match": _as_float(
                metrics.get("exact_match,strict-match") or metrics.get("exact_match")
            ),
            "prompt_level_strict_acc": _as_float(
                metrics.get("prompt_level_strict_acc,none")
                or metrics.get("prompt_level_strict_acc")
            ),
        }

    return {
        "source": "lm_eval",
        "tasks": tasks,
        "raw_keys": sorted(results_block.keys()) if isinstance(results_block, dict) else [],
    }


def run_lm_eval(
    host: str,
    port: int,
    model: str,
    output_dir: Path,
    *,
    preset: str = "industry_lite",
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    """Run lm_eval subprocess and return normalized metrics."""
    if not is_lm_eval_available():
        return {
            "skipped": True,
            "reason": "lm_eval is not installed. pip install -r requirements-quality.txt",
        }

    missing_deps = missing_industry_dependencies()
    if missing_deps:
        packages = ", ".join(missing_deps)
        return {
            "skipped": True,
            "reason": (
                f"Missing Python packages for industry benchmarks: {packages}. "
                "pip install -r requirements-quality.txt"
            ),
        }

    command = build_lm_eval_command(host, port, model, output_dir, preset=preset)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "lm_eval failed").strip()
        raise RuntimeError(_summarize_subprocess_error(stderr))

    parsed = parse_lm_eval_output(output_dir)
    parsed["preset"] = preset
    parsed["skipped"] = False
    return parsed


def _summarize_subprocess_error(stderr: str, *, max_length: int = 500) -> str:
    """Extract the most useful line from lm_eval stderr."""
    for line in reversed(stderr.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if any(
            marker in stripped
            for marker in (
                "Error",
                "Exception",
                "ModuleNotFoundError",
                "NotImplementedError",
                "AssertionError",
            )
        ):
            return stripped[:max_length]
    return stderr[:max_length]


def _as_float(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None
