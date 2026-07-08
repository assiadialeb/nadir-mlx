"""Golden benchmark snapshot comparison for regression guards (MLX-74)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GOLDEN_FIXTURE_DIR = Path(__file__).resolve().parent / "tests" / "fixtures" / "benchmark_golden"
DEFAULT_TOLERANCE_PERCENT = 15.0
HEADLINE_METRIC_KEYS = ("aggregate_tps", "ttft_p50_ms", "latency_p50_ms")


@dataclass(frozen=True)
class GoldenBenchmarkSnapshot:
    """Expected headline metrics for one benchmark scenario."""

    name: str
    scenario: str
    metrics: dict[str, float]
    tolerance_percent: float = DEFAULT_TOLERANCE_PERCENT


def load_golden_snapshot(name: str) -> GoldenBenchmarkSnapshot:
    """Load a committed golden JSON snapshot by stem name."""
    path = GOLDEN_FIXTURE_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Golden snapshot '{name}' must be a JSON object.")
    metrics = payload.get("metrics") or {}
    if not isinstance(metrics, dict):
        raise ValueError(f"Golden snapshot '{name}' metrics must be an object.")
    return GoldenBenchmarkSnapshot(
        name=name,
        scenario=str(payload.get("scenario") or "medium_conc4"),
        metrics={key: float(value) for key, value in metrics.items()},
        tolerance_percent=float(payload.get("tolerance_percent", DEFAULT_TOLERANCE_PERCENT)),
    )


def extract_scenario_summary(
    results_payload: dict[str, Any],
    scenario: str,
) -> dict[str, Any] | None:
    """Return the summary block for a scenario from llmbench JSON output."""
    for entry in results_payload.get("results") or []:
        summary = entry.get("summary") or {}
        if summary.get("scenario") == scenario:
            return summary
    for entry in results_payload.get("results") or []:
        summary = entry.get("summary") or {}
        scenario_name = str(summary.get("scenario") or "")
        if scenario_name.startswith(scenario.split("_")[0]):
            return summary
    return None


def extract_headline_metrics(
    results_payload: dict[str, Any],
    *,
    scenario: str,
) -> dict[str, float]:
    """Pull comparable headline metrics from a benchmark results document."""
    summary = extract_scenario_summary(results_payload, scenario)
    if not summary:
        raise ValueError(f"No summary found for scenario '{scenario}'.")

    metrics: dict[str, float] = {}
    for key in HEADLINE_METRIC_KEYS:
        raw_value = summary.get(key)
        if raw_value is None:
            continue
        metrics[key] = float(raw_value)
    if not metrics:
        raise ValueError(f"Scenario '{scenario}' has no headline metrics.")
    return metrics


def _within_tolerance(
    actual: float,
    expected: float,
    tolerance_percent: float,
) -> bool:
    if expected == 0:
        return actual == 0
    delta = abs(actual - expected) / abs(expected) * 100.0
    return delta <= tolerance_percent


def compare_headline_metrics(
    actual: dict[str, float],
    expected: dict[str, float],
    *,
    tolerance_percent: float = DEFAULT_TOLERANCE_PERCENT,
) -> list[str]:
    """Return human-readable errors when actual metrics drift beyond tolerance."""
    errors: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            errors.append(f"Missing metric '{key}'.")
            continue
        actual_value = actual[key]
        if not _within_tolerance(actual_value, expected_value, tolerance_percent):
            errors.append(
                f"{key}: actual={actual_value:.2f}, expected={expected_value:.2f} "
                f"(tolerance ±{tolerance_percent:g}%).",
            )
    return errors


def assert_within_golden_snapshot(
    results_payload: dict[str, Any],
    snapshot: GoldenBenchmarkSnapshot,
) -> None:
    """Raise AssertionError when results drift outside the golden tolerance band."""
    actual = extract_headline_metrics(results_payload, scenario=snapshot.scenario)
    errors = compare_headline_metrics(
        actual,
        snapshot.metrics,
        tolerance_percent=snapshot.tolerance_percent,
    )
    if errors:
        joined = "; ".join(errors)
        raise AssertionError(f"Golden benchmark drift for '{snapshot.name}': {joined}")
