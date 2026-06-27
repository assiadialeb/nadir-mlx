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
VALID_BENCHMARK_KINDS = frozenset({"PERF", "QUALITY", "COMPLETE"})
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
    benchmark_kind = _parse_optional_enum(params.get("benchmark_kind"), VALID_BENCHMARK_KINDS)
    instance_id = _parse_positive_int(params.get("instance_id"))
    scenario = str(params.get("scenario") or "medium_conc4").strip() or "medium_conc4"
    page = _parse_history_page(params.get("page"))

    return {
        "model_name": model_name,
        "launch_mode": launch_mode,
        "status": status,
        "target_type": target_type,
        "benchmark_kind": benchmark_kind,
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

    if filters.get("benchmark_kind"):
        queryset = queryset.filter(benchmark_kind=filters["benchmark_kind"])

    queryset = queryset.filter(parent_run__isnull=True)

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
    quality = run.quality_metrics
    headline_quality = _headline_quality_metric(quality)
    return {
        "id": run.id,
        "status": run.status,
        "benchmark_kind": run.benchmark_kind,
        "target_type": run.target_type,
        "endpoint_url": run.endpoint_url,
        "model_name": run.instance.model_name if run.instance_id else run.model_id,
        "launch_mode": run.instance.launch_mode if run.instance_id else "",
        "preset_key": benchmark_preset_key(run.params),
        "preset_label": format_preset_label(run.params, run.benchmark_kind),
        "created_at": run.created_at,
        "completed_at": run.completed_at,
        "ttft_p50_ms": summary.get("ttft_p50_ms") if summary else None,
        "latency_p50_ms": summary.get("latency_p50_ms") if summary else None,
        "latency_p95_ms": summary.get("latency_p95_ms") if summary else None,
        "aggregate_tps": summary.get("aggregate_tps") if summary else None,
        "quality_headline": headline_quality,
        "scenario": summary.get("scenario") if summary else scenario,
    }


def _headline_quality_metric(metrics: dict[str, Any]) -> str | None:
    if not metrics:
        return None
    for key in ("ifeval_strict_acc", "gsm8k_exact_match", "mmlu_acc", "text_platform_pass_rate"):
        if metrics.get(key) is not None:
            return f"{key.replace('_', ' ')}: {metrics[key]}%"
    for key, value in metrics.items():
        if value is not None:
            return f"{key}: {value}%"
    return None


def format_preset_label(params: dict[str, Any] | None, benchmark_kind: str = "PERF") -> str:
    """Human-readable preset description."""
    if benchmark_kind == "QUALITY":
        preset = (params or {}).get("quality_preset", "industry_lite")
        return f"Quality · {preset}"
    if benchmark_kind == "COMPLETE":
        return "Complete · perf → quality"
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
            seen[key] = format_preset_label(run.params, run.benchmark_kind)
    return sorted(seen.items(), key=lambda item: item[1])


def list_filter_options() -> dict[str, Any]:
    """Dropdown values for history filters."""
    instances = InferenceInstance.objects.order_by("-created_at")
    return {
        "instances": instances,
        "launch_modes": sorted(VALID_LAUNCH_MODES),
        "statuses": sorted(VALID_STATUSES),
        "target_types": sorted(VALID_TARGET_TYPES),
        "benchmark_kinds": sorted(VALID_BENCHMARK_KINDS),
    }


def build_benchmark_history_query(**kwargs: Any) -> str:
    """Build query string for benchmark history filters."""
    params: dict[str, str] = {}
    mapping = {
        "model_name": "model",
        "launch_mode": "launch_mode",
        "status": "status",
        "target_type": "target_type",
        "benchmark_kind": "benchmark_kind",
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
        "preset_label": format_preset_label(run_a.params, run_a.benchmark_kind),
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


def benchmark_run_model_key(run: BenchmarkRun) -> str:
    """Normalized model identity for grouping runs (instance folder or API model id)."""
    if run.instance_id and run.instance:
        return run.instance.model_name.strip().lower()
    return (run.model_id or "").strip().lower()


def benchmark_endpoint_kind(run: BenchmarkRun, gateway_port: int) -> str:
    """Short target label for compare UI."""
    if run.target_type == "INSTANCE":
        return "nadir"
    url = (run.endpoint_url or "").lower()
    if f":{gateway_port}" in url:
        return "gateway"
    return "external"


def benchmark_run_label(run: BenchmarkRun, gateway_port: int = 11380) -> str:
    """Human-readable label for compare dropdowns."""
    model = run.instance.model_name if run.instance_id else (run.model_id or "—")
    kind = benchmark_endpoint_kind(run, gateway_port)
    stamp = run.completed_at or run.created_at
    time_str = stamp.strftime("%d/%m %H:%M") if stamp else ""
    return f"#{run.id} · {model} · {kind} · {time_str}"


def comparison_pair_label(
    run_a: BenchmarkRun,
    run_b: BenchmarkRun,
    gateway_port: int = 11380,
) -> str:
    """Short label for a suggested comparison pair."""
    kind_a = benchmark_endpoint_kind(run_a, gateway_port)
    kind_b = benchmark_endpoint_kind(run_b, gateway_port)
    return f"#{run_a.id} {kind_a} vs #{run_b.id} {kind_b}"


def benchmark_history_model_query(run: BenchmarkRun) -> str:
    """Model filter value for history charts linking from a run detail page."""
    if run.instance_id and run.instance:
        return run.instance.model_name
    return run.model_id or ""


def _group_runs_for_comparison(
    completed: list[BenchmarkRun],
    preset_key: str | None,
) -> dict[tuple[str, str], list[BenchmarkRun]]:
    groups: dict[tuple[str, str], list[BenchmarkRun]] = {}
    for run in completed:
        preset = benchmark_preset_key(run.params)
        if preset_key and preset != preset_key:
            continue
        model_key = benchmark_run_model_key(run)
        if not model_key:
            continue
        groups.setdefault((preset, model_key), []).append(run)
    return groups


def _append_unique_endpoint_pairs(
    runs: list[BenchmarkRun],
    seen_ids: set[tuple[int, int]],
    pairs: list[tuple[BenchmarkRun, BenchmarkRun]],
    *,
    max_pairs: int,
) -> bool:
    """Add cross-endpoint pairs from one preset/model group. Return True when max_pairs reached."""
    if len(runs) < 2:
        return False
    for index, run_a in enumerate(runs):
        for run_b in runs[index + 1 :]:
            if run_a.endpoint_url == run_b.endpoint_url:
                continue
            pair_ids = tuple(sorted((run_a.id, run_b.id)))
            if pair_ids in seen_ids:
                continue
            seen_ids.add(pair_ids)
            pairs.append((run_a, run_b))
            if len(pairs) >= max_pairs:
                return True
    return False


def find_comparison_candidates(
    queryset: QuerySet[BenchmarkRun],
    *,
    preset_key: str | None = None,
) -> list[tuple[BenchmarkRun, BenchmarkRun]]:
    """Pair completed runs that share preset + model but use different endpoints."""
    completed = list(
        queryset.filter(status="COMPLETED").select_related("instance").order_by("-completed_at")
    )
    groups = _group_runs_for_comparison(completed, preset_key)

    pairs: list[tuple[BenchmarkRun, BenchmarkRun]] = []
    seen_ids: set[tuple[int, int]] = set()
    for runs in groups.values():
        if _append_unique_endpoint_pairs(runs, seen_ids, pairs, max_pairs=12):
            return pairs
    return pairs


def runs_for_chart_filters(filters: dict[str, Any], *, limit: int = 50) -> list[BenchmarkRun]:
    """Return recent completed runs for charting."""
    queryset = filter_benchmark_runs(filters).filter(status="COMPLETED")[:limit]
    return list(queryset)


QUALITY_METRIC_LABELS: dict[str, str] = {
    "gsm8k_exact_match": "GSM8K",
    "ifeval_strict_acc": "IFEval",
    "text_platform_pass_rate": "Platform",
    "mmlu_acc": "MMLU",
}

QUALITY_CHART_KEY_ORDER = (
    "ifeval_strict_acc",
    "gsm8k_exact_match",
    "text_platform_pass_rate",
    "mmlu_acc",
)


def quality_metric_label(metric_key: str) -> str:
    """Human-readable label for a stored quality metric key."""
    return QUALITY_METRIC_LABELS.get(metric_key, metric_key.replace("_", " "))


def resolve_perf_summaries(
    run: BenchmarkRun,
    perf_child: BenchmarkRun | None = None,
) -> list[dict[str, Any]]:
    """Return scenario summaries for the detail page (child perf run for COMPLETE)."""
    if run.benchmark_kind == "COMPLETE":
        if perf_child is not None and perf_child.summaries:
            return perf_child.summaries
        embedded = (run.results or {}).get("perf_summaries") or []
        if embedded:
            return embedded
        return []
    return run.summaries


def resolve_quality_metrics(
    run: BenchmarkRun,
    quality_child: BenchmarkRun | None = None,
) -> dict[str, Any]:
    """Return quality headline metrics for the detail page."""
    if run.benchmark_kind == "COMPLETE" and quality_child is not None:
        return quality_child.quality_metrics
    return run.quality_metrics


def resolve_quality_results(
    run: BenchmarkRun,
    quality_child: BenchmarkRun | None = None,
) -> dict[str, Any] | None:
    """Return full quality payload (platform cases, industry block)."""
    if run.benchmark_kind == "COMPLETE":
        if quality_child is not None and quality_child.results:
            return quality_child.results
        embedded = (run.results or {}).get("quality_results")
        if embedded:
            return embedded
        return None
    return run.results


def detail_render_ready(
    run: BenchmarkRun,
    perf_child: BenchmarkRun | None = None,
    quality_child: BenchmarkRun | None = None,
) -> bool:
    """True when the detail page has enough data to render final results."""
    if run.status != "COMPLETED":
        return False
    if run.benchmark_kind == "PERF":
        return bool(resolve_perf_summaries(run, perf_child))
    if run.benchmark_kind == "QUALITY":
        return bool(resolve_quality_metrics(run, quality_child))
    if run.benchmark_kind == "COMPLETE":
        return bool(resolve_perf_summaries(run, perf_child)) and bool(
            resolve_quality_metrics(run, quality_child)
        )
    return True


def build_benchmark_status_payload(
    run: BenchmarkRun,
    perf_child: BenchmarkRun | None = None,
    quality_child: BenchmarkRun | None = None,
) -> dict[str, Any]:
    """JSON payload for live benchmark detail polling."""
    perf_summaries = resolve_perf_summaries(run, perf_child)
    quality_metrics = resolve_quality_metrics(run, quality_child)
    quality_results = resolve_quality_results(run, quality_child) or {}
    platform_suite = (
        (quality_results.get("platform") or {}).get("suites") or {}
    ).get("text_platform") or {}
    industry = quality_results.get("industry") or {}

    return {
        "id": run.id,
        "status": run.status,
        "benchmark_kind": run.benchmark_kind,
        "render_ready": detail_render_ready(run, perf_child, quality_child),
        "error_message": run.error_message,
        "warnings": (
            quality_child.quality_warnings
            if run.benchmark_kind == "COMPLETE" and quality_child
            else run.quality_warnings
        ),
        "phase": {
            "perf_status": perf_child.status if perf_child else None,
            "quality_status": quality_child.status if quality_child else None,
        },
        "perf_summaries": perf_summaries,
        "quality_metrics": quality_metrics,
        "quality_metric_items": [
            {"key": key, "label": quality_metric_label(key), "value": value}
            for key, value in quality_metrics.items()
        ],
        "platform_suite": platform_suite,
        "industry_skipped_reason": industry.get("reason") if industry.get("skipped") else None,
        "perf_chart": detail_perf_chart_payload(perf_summaries),
        "quality_chart": detail_quality_chart_payload(quality_metrics),
    }


def benchmark_detail_phase_message(
    run: BenchmarkRun,
    perf_child: BenchmarkRun | None = None,
    quality_child: BenchmarkRun | None = None,
) -> str:
    """Operator-facing progress text while a COMPLETE run is in flight."""
    if run.benchmark_kind != "COMPLETE" or run.status not in ("PENDING", "RUNNING"):
        return "Benchmark in progress. This may take several minutes."
    if perf_child and perf_child.status in ("PENDING", "RUNNING"):
        return "Phase 1 — Performance benchmark in progress."
    if quality_child and quality_child.status in ("PENDING", "RUNNING"):
        return (
            "Phase 2 — Quality benchmark in progress. "
            "Industry tasks can take 30+ minutes depending on hardware."
        )
    return "Finalizing complete benchmark."


def detail_perf_chart_payload(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Chart.js payload for per-scenario performance bars on run detail."""
    if not summaries:
        return {}
    labels = [str(item.get("scenario") or "scenario") for item in summaries]
    return {
        "labels": labels,
        "ttft_p50_ms": [_numeric_metric(item.get("ttft_p50_ms")) for item in summaries],
        "latency_p50_ms": [_numeric_metric(item.get("latency_p50_ms")) for item in summaries],
        "aggregate_tps": [_numeric_metric(item.get("aggregate_tps")) for item in summaries],
    }


def detail_quality_chart_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    """Chart.js payload for quality score bars on run detail."""
    if not metrics:
        return {}
    labels: list[str] = []
    values: list[float] = []
    seen: set[str] = set()
    for key in QUALITY_CHART_KEY_ORDER:
        raw_value = metrics.get(key)
        if raw_value is None:
            continue
        labels.append(quality_metric_label(key))
        values.append(float(raw_value))
        seen.add(key)
    for key, raw_value in metrics.items():
        if key in seen or raw_value is None:
            continue
        labels.append(quality_metric_label(key))
        values.append(float(raw_value))
    return {"labels": labels, "values": values}
