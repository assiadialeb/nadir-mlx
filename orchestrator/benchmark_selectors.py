"""Read-only selectors for benchmark history, charts, and comparisons."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from django.core.paginator import Paginator
from django.db.models import Q, QuerySet

from orchestrator.models import BenchmarkRun, InferenceInstance

BENCHMARK_HISTORY_PER_PAGE = 25

VALID_STATUSES = frozenset({"PENDING", "RUNNING", "COMPLETED", "FAILED"})
VALID_TARGET_TYPES = frozenset({"INSTANCE", "ENDPOINT"})
VALID_LAUNCH_MODES = frozenset(
    {"TEXT", "MULTIMODAL", "EMBEDDING", "RERANKER", "IMAGE", "TTS", "STT"}
)


def benchmark_preset_key(params: dict[str, Any] | None) -> str:
    """Stable key for runs sharing the same llmbench preset."""
    raw = params or {}
    categories = sorted(str(item) for item in (raw.get("categories") or []))
    concurrency = sorted(int(level) for level in (raw.get("concurrency") or []))
    num_requests = int(raw.get("num_requests") or 5)
    return (
        f"cat:{'+'.join(categories) or 'medium'}"
        f"|conc:{','.join(str(level) for level in concurrency) or '1'}"
        f"|n:{num_requests}"
    )


def _parse_optional_enum(raw_value: object, valid_values: frozenset[str]) -> str:
    normalized = str(raw_value or "").strip().upper()
    return normalized if normalized in valid_values else ""


def _parse_positive_int(raw_value: object) -> int | None:
    cleaned = str(raw_value or "").strip()
    return int(cleaned) if cleaned.isdigit() else None


def _parse_history_page(raw_value: object) -> int:
    cleaned = str(raw_value or "1").strip()
    if cleaned.isdigit() and int(cleaned) > 0:
        return int(cleaned)
    return 1


def parse_benchmark_history_query(params: dict[str, Any]) -> dict[str, Any]:
    """Parse history filter query parameters."""
    model_name = str(params.get("model") or "").strip()
    launch_mode = _parse_optional_enum(params.get("launch_mode"), VALID_LAUNCH_MODES)
    status = _parse_optional_enum(params.get("status"), VALID_STATUSES)
    target_type = _parse_optional_enum(params.get("target_type"), VALID_TARGET_TYPES)
    instance_id = _parse_positive_int(params.get("instance_id"))
    scenario = str(params.get("scenario") or "medium_conc4").strip() or "medium_conc4"
    page = _parse_history_page(params.get("page"))

    return {
        "model_name": model_name,
        "launch_mode": launch_mode,
        "status": status,
        "target_type": target_type,
        "instance_id": instance_id,
        "scenario": scenario,
        "preset_key": str(params.get("preset") or "").strip(),
        "page": page,
    }


def filter_benchmark_runs(filters: dict[str, Any]) -> QuerySet[BenchmarkRun]:
    """Return benchmark runs matching history filters."""
    queryset = BenchmarkRun.objects.select_related("instance").order_by("-created_at")

    if filters.get("model_name"):
        needle = filters["model_name"]
        queryset = queryset.filter(
            Q(instance__model_name__icontains=needle)
            | Q(model_id__icontains=needle)
        )

    if filters.get("instance_id"):
        queryset = queryset.filter(instance_id=filters["instance_id"])

    if filters.get("launch_mode"):
        queryset = queryset.filter(instance__launch_mode=filters["launch_mode"])

    if filters.get("status"):
        queryset = queryset.filter(status=filters["status"])

    if filters.get("target_type"):
        queryset = queryset.filter(target_type=filters["target_type"])

    if filters.get("preset_key"):
        matching_ids = [
            run.id
            for run in queryset
            if benchmark_preset_key(run.params) == filters["preset_key"]
        ]
        queryset = queryset.filter(id__in=matching_ids)

    return queryset


def paginate_benchmark_runs(
    queryset: QuerySet[BenchmarkRun],
    page: int,
    *,
    per_page: int = BENCHMARK_HISTORY_PER_PAGE,
) -> tuple[list[BenchmarkRun], Any]:
    """Paginate a benchmark queryset."""
    paginator = Paginator(queryset, per_page)
    page_obj = paginator.get_page(page)
    return list(page_obj.object_list), page_obj


def summary_for_scenario(run: BenchmarkRun, scenario: str) -> dict[str, Any] | None:
    """Return the summary dict for a scenario name, with sensible fallbacks."""
    summaries = run.summaries
    if not summaries:
        return None

    for entry in summaries:
        if entry.get("scenario") == scenario:
            return entry

    for entry in summaries:
        if scenario.split("_")[0] in str(entry.get("scenario", "")):
            return entry

    return summaries[0]


def benchmark_run_list_row(run: BenchmarkRun, scenario: str = "medium_conc4") -> dict[str, Any]:
    """Serialize a run for the history table."""
    summary = summary_for_scenario(run, scenario)
    return {
        "id": run.id,
        "status": run.status,
        "target_type": run.target_type,
        "endpoint_url": run.endpoint_url,
        "model_name": run.instance.model_name if run.instance_id else run.model_id,
        "launch_mode": run.instance.launch_mode if run.instance_id else "",
        "preset_key": benchmark_preset_key(run.params),
        "preset_label": format_preset_label(run.params),
        "created_at": run.created_at,
        "completed_at": run.completed_at,
        "ttft_p50_ms": summary.get("ttft_p50_ms") if summary else None,
        "latency_p50_ms": summary.get("latency_p50_ms") if summary else None,
        "latency_p95_ms": summary.get("latency_p95_ms") if summary else None,
        "aggregate_tps": summary.get("aggregate_tps") if summary else None,
        "scenario": summary.get("scenario") if summary else scenario,
    }


def format_preset_label(params: dict[str, Any] | None) -> str:
    """Human-readable preset description."""
    raw = params or {}
    categories = ", ".join(raw.get("categories") or ["medium"])
    concurrency = ", ".join(str(level) for level in (raw.get("concurrency") or [1]))
    num_requests = raw.get("num_requests", 5)
    return f"{categories} · conc {concurrency} · {num_requests} req"


def list_distinct_preset_keys(queryset: QuerySet[BenchmarkRun]) -> list[tuple[str, str]]:
    """Return unique preset keys with labels from completed runs."""
    seen: dict[str, str] = {}
    for run in queryset.filter(status="COMPLETED"):
        key = benchmark_preset_key(run.params)
        if key not in seen:
            seen[key] = format_preset_label(run.params)
    return sorted(seen.items(), key=lambda item: item[1])


def list_filter_options() -> dict[str, Any]:
    """Dropdown values for history filters."""
    instances = InferenceInstance.objects.order_by("-created_at")
    return {
        "instances": instances,
        "launch_modes": sorted(VALID_LAUNCH_MODES),
        "statuses": sorted(VALID_STATUSES),
        "target_types": sorted(VALID_TARGET_TYPES),
    }


def build_benchmark_history_query(**kwargs: Any) -> str:
    """Build query string for benchmark history filters."""
    params: dict[str, str] = {}
    mapping = {
        "model_name": "model",
        "launch_mode": "launch_mode",
        "status": "status",
        "target_type": "target_type",
        "instance_id": "instance_id",
        "scenario": "scenario",
        "preset_key": "preset",
        "page": "page",
    }
    for source_key, param_key in mapping.items():
        value = kwargs.get(source_key)
        if value is None or value == "":
            continue
        params[param_key] = str(value)
    return urlencode(params)


def chart_series_for_runs(
    runs: list[BenchmarkRun],
    *,
    scenario: str,
) -> dict[str, Any]:
    """Build Chart.js datasets for metric evolution."""
    completed = [run for run in runs if run.status == "COMPLETED" and run.summaries]
    completed.sort(key=lambda run: run.completed_at or run.created_at)

    labels: list[str] = []
    ttft: list[float | None] = []
    latency_p50: list[float | None] = []
    latency_p95: list[float | None] = []
    aggregate_tps: list[float | None] = []

    for run in completed:
        summary = summary_for_scenario(run, scenario)
        if not summary:
            continue
        stamp = run.completed_at or run.created_at
        labels.append(stamp.strftime("%d/%m %H:%M"))
        ttft.append(_numeric_metric(summary.get("ttft_p50_ms")))
        latency_p50.append(_numeric_metric(summary.get("latency_p50_ms")))
        latency_p95.append(_numeric_metric(summary.get("latency_p95_ms")))
        aggregate_tps.append(_numeric_metric(summary.get("aggregate_tps")))

    return {
        "labels": labels,
        "scenario": scenario,
        "datasets": {
            "ttft_p50_ms": ttft,
            "latency_p50_ms": latency_p50,
            "latency_p95_ms": latency_p95,
            "aggregate_tps": aggregate_tps,
        },
        "runs": [run.id for run in completed],
    }


def _numeric_metric(raw_value: Any) -> float | None:
    if raw_value is None or raw_value == "N/A":
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def comparison_rows(run_a: BenchmarkRun, run_b: BenchmarkRun) -> list[dict[str, Any]]:
    """Align scenario summaries from two runs for side-by-side display."""
    summaries_a = {item.get("scenario"): item for item in run_a.summaries}
    summaries_b = {item.get("scenario"): item for item in run_b.summaries}
    scenarios = sorted(set(summaries_a) | set(summaries_b))

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        rows.append(
            {
                "scenario": scenario,
                "left": summaries_a.get(scenario),
                "right": summaries_b.get(scenario),
            }
        )
    return rows


def build_comparison_snapshot(run_a: BenchmarkRun, run_b: BenchmarkRun) -> dict[str, Any]:
    """Exportable JSON snapshot for an MLX vs external comparison."""
    stamp = run_b.completed_at or run_b.created_at
    return {
        "generated_at": stamp.isoformat() if stamp else "",
        "preset_key": benchmark_preset_key(run_a.params),
        "preset_label": format_preset_label(run_a.params),
        "scenario_alignment": comparison_rows(run_a, run_b),
        "runs": {
            "left": _run_snapshot(run_a),
            "right": _run_snapshot(run_b),
        },
    }


def _run_snapshot(run: BenchmarkRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "target_type": run.target_type,
        "endpoint_url": run.endpoint_url,
        "model_name": run.instance.model_name if run.instance_id else run.model_id,
        "launch_mode": run.instance.launch_mode if run.instance_id else "",
        "status": run.status,
        "params": run.params,
        "summaries": run.summaries,
    }


def find_comparison_candidates(
    queryset: QuerySet[BenchmarkRun],
    *,
    preset_key: str | None = None,
) -> list[tuple[BenchmarkRun, BenchmarkRun]]:
    """Pair INSTANCE and ENDPOINT runs that share the same preset."""
    completed = list(
        queryset.filter(status="COMPLETED").select_related("instance").order_by("-completed_at")
    )
    groups: dict[str, dict[str, list[BenchmarkRun]]] = {}
    for run in completed:
        key = benchmark_preset_key(run.params)
        if preset_key and key != preset_key:
            continue
        bucket = groups.setdefault(key, {"INSTANCE": [], "ENDPOINT": []})
        bucket[run.target_type].append(run)

    pairs: list[tuple[BenchmarkRun, BenchmarkRun]] = []
    for bucket in groups.values():
        mlx_runs = bucket.get("INSTANCE") or []
        external_runs = bucket.get("ENDPOINT") or []
        for mlx_run in mlx_runs[:3]:
            for external_run in external_runs[:3]:
                pairs.append((mlx_run, external_run))
    return pairs


def runs_for_chart_filters(filters: dict[str, Any], *, limit: int = 50) -> list[BenchmarkRun]:
    """Return recent completed runs for charting."""
    queryset = filter_benchmark_runs(filters).filter(status="COMPLETED")[:limit]
    return list(queryset)
