import json
from urllib.parse import urlencode

from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.utils.http import url_has_allowed_host_and_scheme
from .models import InferenceInstance, BenchmarkRun, ModelDownload
from .downloader import start_model_download
from .model_lifecycle import delete_installed_model
from .model_utils import reconcile_stale_downloads, sync_model_download_status, validate_model_folder_name
from .server_manager import (
    parse_launch_mode,
    start_instance,
    stop_instance,
    delete_instance,
    restart_instance,
    update_stopped_instance,
    get_instance_logs,
)
from .instance_health import refresh_instance_health
from .server_types import SERVER_TYPES
from .model_registry import (
    build_registry_defaults_json,
    build_registry_metadata_json,
)
from .server_config_schema import (
    advanced_keys_for_ui_json,
    config_fields_for_ui_json,
    resolve_server_config_from_request,
)
from .lifecycle_selectors import enrich_instance_lifecycle_ui
from .ui_selectors import (
    apply_installed_models_filters,
    fetch_hf_models,
    hf_fetch_limit_for_filters,
    list_installed_models,
    parse_installed_models_query,
    models_by_server_type_json,
    SORT_OPTIONS,
)
from .benchmark_service import (
    BENCHMARK_MAX_REQUESTS_PER_SCENARIO,
    parse_benchmark_form,
    start_benchmark,
    delete_benchmark_run,
)
from .benchmark_selectors import (
    benchmark_endpoint_kind,
    benchmark_run_label,
    benchmark_run_list_row,
    build_benchmark_history_query,
    build_comparison_snapshot,
    chart_series_for_runs,
    comparison_pair_label,
    comparison_rows,
    filter_benchmark_runs,
    find_comparison_candidates,
    list_distinct_preset_keys,
    list_filter_options,
    paginate_benchmark_runs,
    parse_benchmark_history_query,
    runs_for_chart_filters,
    benchmark_history_model_query,
)

_startup_reconciled = False


def _ensure_downloads_reconciled() -> None:
    global _startup_reconciled
    if _startup_reconciled:
        return
    reconcile_stale_downloads()
    _startup_reconciled = True


# Authentication Views
def login_view(request):
    if request.user.is_authenticated:
        return redirect('servers')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('servers')
        else:
            messages.error(request, "Invalid username or password.")
            
    return render(request, 'orchestrator/login.html')

def logout_view(request):
    logout(request)
    return redirect('login')


# Model parsing helper — kept for legacy imports; HF parsing lives in ui_selectors.


@login_required
def search_view(request):
    """Legacy route — redirects to the Models page (Hugging Face tab)."""
    params = {'tab': 'hub'}
    query = request.GET.get('q', '')
    if query:
        params['q'] = query
    return redirect(f"{reverse('models')}?{urlencode(params)}")


@login_required
def models_view(request):
    _ensure_downloads_reconciled()
    sync_model_download_status()
    active_tab = request.GET.get('tab', 'installed')
    if active_tab not in ('installed', 'hub'):
        active_tab = 'installed'

    installed_models_all = list_installed_models()
    filter_query = ""
    filter_cap = ""
    filter_sort = "name_asc"
    filter_total = 0
    filter_count = 0
    filter_has_filters = False
    filter_fetch_limit = 0
    downloading_models = ModelDownload.objects.filter(status='DOWNLOADING').order_by('-created_at')
    hf_models: list[dict] = []
    installed_models = installed_models_all

    if active_tab == 'installed':
        filter_query, filter_cap, filter_sort = parse_installed_models_query(request.GET)
        installed_models = apply_installed_models_filters(
            installed_models_all,
            query=filter_query,
            capability=filter_cap,
            sort=filter_sort,
        )
        filter_total = len(installed_models_all)
        filter_count = len(installed_models)
        filter_has_filters = bool(filter_query or filter_cap or filter_sort != 'name_asc')
    elif active_tab == 'hub':
        filter_query, filter_cap, filter_sort = parse_installed_models_query(request.GET)
        filter_fetch_limit = hf_fetch_limit_for_filters(filter_query, filter_cap, filter_sort)
        filter_has_filters = bool(filter_query or filter_cap or filter_sort != 'name_asc')
        try:
            hf_models_all = fetch_hf_models(filter_query, limit=filter_fetch_limit)
            hf_models = apply_installed_models_filters(
                hf_models_all,
                query=filter_query,
                capability=filter_cap,
                sort=filter_sort,
            )
            filter_total = len(hf_models_all)
            filter_count = len(hf_models)
        except Exception as exc:
            messages.error(request, f"Unable to reach the Hugging Face API: {exc}")

    return render(request, 'orchestrator/models.html', {
        'active_tab': active_tab,
        'installed_models': installed_models,
        'installed_models_total': len(installed_models_all),
        'filter_query': filter_query,
        'filter_cap': filter_cap,
        'filter_sort': filter_sort,
        'filter_total': filter_total,
        'filter_count': filter_count,
        'filter_has_filters': filter_has_filters,
        'filter_fetch_limit': filter_fetch_limit,
        'sort_options': SORT_OPTIONS,
        'capability_filters': SERVER_TYPES,
        'downloading_models': downloading_models,
        'models': hf_models,
    })


@login_required
def delete_model_view(request):
    if request.method != 'POST':
        return redirect('models')

    folder_name = (request.POST.get('model_name') or '').strip()
    if not folder_name:
        messages.error(request, "No model selected.")
        return redirect(f"{reverse('models')}?tab=installed")

    try:
        result = delete_installed_model(folder_name)
        messages.success(
            request,
            (
                f"Model {result.folder_name} deleted "
                f"({result.instances_removed} server(s), "
                f"{result.benchmark_runs_removed} benchmark(s), "
                f"{result.log_files_removed} log(s))."
            ),
        )
    except ValueError as exc:
        messages.error(request, str(exc))
    except Exception as exc:
        messages.error(request, f"Failed to delete model: {exc}")

    return redirect(f"{reverse('models')}?tab=installed")


# Download Trigger
@login_required
def download_model_view(request):
    if request.method == 'POST':
        repo_id = request.POST.get('repo_id')
        if repo_id:
            try:
                start_model_download(repo_id)
                messages.success(request, f"Download started in the background for model {repo_id}.")
            except ValueError as exc:
                messages.error(request, str(exc))
        else:
            messages.error(request, "No model identifier specified.")
    tab = request.POST.get('tab', 'hub')
    params: dict[str, str] = {'tab': tab}
    for key in ('q', 'cap', 'sort'):
        value = (request.POST.get(key) or '').strip()
        if value:
            params[key] = value
    return redirect(f"{reverse('models')}?{urlencode(params)}")


@login_required
def dashboard_view(request):
    """Legacy route — redirects to the Servers page."""
    return redirect('servers')


@login_required
def servers_view(request):
    _ensure_downloads_reconciled()
    installed_models = list_installed_models()
    installed_names = [model["name"] for model in installed_models]
    instances = InferenceInstance.objects.all().order_by('-created_at')
    for instance in instances:
        refresh_instance_health(instance)
        instance.server_config_json = json.dumps(instance.server_config or {})
        enrich_instance_lifecycle_ui(instance)

    return render(request, 'orchestrator/servers.html', {
        'instances': instances,
        'server_types': SERVER_TYPES,
        'models_by_mode_json': models_by_server_type_json(installed_models),
        'config_fields_json': config_fields_for_ui_json(),
        'advanced_keys_json': advanced_keys_for_ui_json(),
        'registry_defaults_json': build_registry_defaults_json(installed_names),
        'registry_metadata_json': build_registry_metadata_json(installed_names),
    })



# Start Instance Trigger
@login_required
def start_instance_view(request):
    if request.method == 'POST':
        model_name = (request.POST.get('model_name') or '').strip()
        port_raw = request.POST.get('port')
        launch_mode_raw = request.POST.get('launch_mode', 'TEXT')

        if not model_name:
            messages.error(request, "Please select a model.")
            return redirect('servers')

        port = None
        if port_raw and port_raw.strip():
            try:
                port = int(port_raw)
            except ValueError:
                messages.error(request, "The specified port must be a valid number.")
                return redirect('servers')

        try:
            launch_mode = parse_launch_mode(launch_mode_raw)
            server_config = resolve_server_config_from_request(
                request.POST,
                launch_mode,
                model_name,
            )
            instance = start_instance(model_name, port, launch_mode, server_config)
            mode_labels = {
                "MULTIMODAL": "Multimodal (mlx_vlm)",
                "EMBEDDING": "Embeddings (mlx-embeddings)",
                "RERANKER": "Rerank (local-reranker)",
                "IMAGE": "Image (mflux)",
                "TTS": "TTS (mlx-audio / Kokoro)",
                "STT": "STT (mlx-audio / Whisper)",
            }
            mode_label = mode_labels.get(instance.launch_mode, "Text (mlx_lm)")
            messages.success(
                request,
                f"{mode_label} instance started on port {instance.port} (PID: {instance.pid}).",
            )
        except ValueError as e:
            messages.error(request, str(e))
        except Exception as e:
            messages.error(request, f"Failed to start instance: {str(e)}")
            
    return redirect('servers')


# Stop Instance Trigger
@login_required
def stop_instance_view(request, instance_id):
    if request.method == 'POST':
        instance = get_object_or_404(InferenceInstance, id=instance_id)
        try:
            stop_instance(instance)
            messages.success(request, f"Instance on port {instance.port} stopped.")
        except RuntimeError as exc:
            messages.error(request, str(exc))
        except Exception as e:
            messages.error(request, f"Failed to stop instance: {str(e)}")
            
    return redirect('servers')


@login_required
def edit_instance_view(request, instance_id):
    if request.method != 'POST':
        return redirect('servers')

    instance = get_object_or_404(InferenceInstance, id=instance_id)
    port_raw = (request.POST.get('port') or '').strip()
    launch_mode_raw = request.POST.get('launch_mode')
    restart_after = request.POST.get('restart_after') == '1'

    try:
        port = int(port_raw) if port_raw else None
        launch_mode = parse_launch_mode(launch_mode_raw) if launch_mode_raw else None
        server_config = resolve_server_config_from_request(
            request.POST,
            launch_mode or instance.launch_mode,
            instance.model_name,
        )
        update_stopped_instance(
            instance,
            port=port,
            launch_mode=launch_mode,
            server_config=server_config,
        )
        instance.refresh_from_db()

        if restart_after:
            restart_instance(instance)
            messages.success(
                request,
                f"Instance {instance.model_name} updated and restarted on port {instance.port}.",
            )
        else:
            messages.success(
                request,
                f"Configuration updated for {instance.model_name} (port {instance.port}).",
            )
    except ValueError as exc:
        messages.error(request, str(exc))
    except Exception as exc:
        messages.error(request, f"Failed to update instance: {exc}")

    return redirect('servers')


@login_required
def restart_instance_view(request, instance_id):
    if request.method != 'POST':
        return redirect('servers')

    instance = get_object_or_404(InferenceInstance, id=instance_id)
    try:
        restarted = restart_instance(instance)
        messages.success(
            request,
            f"Instance {restarted.model_name} restarted on port {restarted.port} (PID: {restarted.pid}).",
        )
    except ValueError as exc:
        messages.error(request, str(exc))
    except Exception as exc:
        messages.error(request, f"Failed to restart instance: {exc}")

    return redirect('servers')


@login_required
def delete_instance_view(request, instance_id):
    if request.method == 'POST':
        instance = get_object_or_404(InferenceInstance, id=instance_id)
        model_name = instance.model_name
        port = instance.port
        try:
            delete_instance(instance)
            messages.success(
                request,
                f"Instance {model_name} (port {port}) deleted.",
            )
        except Exception as e:
            messages.error(
                request,
                f"Failed to delete instance: {str(e)}",
            )

    return redirect('servers')


# Logs View API / AJAX page
@login_required
def logs_view(request, model_name, port):
    try:
        validate_model_folder_name(model_name)
    except ValueError:
        messages.error(request, "Invalid model name.")
        return redirect('servers')

    logs = get_instance_logs(model_name, port)
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'logs': logs})
        
    return render(request, 'orchestrator/logs.html', {
        'model_name': model_name,
        'port': port,
        'logs': logs
    })


@login_required
def benchmark_view(request):
    instances = InferenceInstance.objects.filter(
        status="RUNNING",
        launch_mode__in=["TEXT", "MULTIMODAL"],
    ).order_by("-created_at")
    runs = BenchmarkRun.objects.all()[:20]
    selected_instance_id = request.GET.get("instance_id", "")

    return render(request, "orchestrator/benchmark.html", {
        "running_instances": instances,
        "runs": runs,
        "selected_instance_id": selected_instance_id,
        "gateway_port": settings.NADIR_GATEWAY_PORT,
        "max_requests_per_scenario": BENCHMARK_MAX_REQUESTS_PER_SCENARIO,
        "benchmark_endpoint_enabled": settings.NADIR_BENCHMARK_ENDPOINT_ENABLED,
    })


@login_required
def start_benchmark_view(request):
    if request.method != "POST":
        return redirect("benchmark")

    try:
        form_data = parse_benchmark_form(request.POST)
        run = start_benchmark(
            target_type=form_data["target_type"],
            instance_id=form_data["instance_id"],
            host=form_data["host"],
            port=form_data["port"],
            model_id=form_data["model_id"],
            params=form_data["params"],
            benchmark_kind=form_data["benchmark_kind"],
        )
        kind_label = form_data["benchmark_kind"].title()
        messages.success(
            request,
            f"{kind_label} benchmark #{run.id} started against {run.endpoint_url}.",
        )
        return redirect("benchmark_detail", run_id=run.id)
    except ValueError as exc:
        messages.error(request, str(exc))
    except Exception as exc:
        messages.error(request, f"Failed to start benchmark: {exc}")

    return redirect("benchmark")


@login_required
def benchmark_detail_view(request, run_id: int):
    run = get_object_or_404(BenchmarkRun, id=run_id)
    child_runs = list(run.child_runs.order_by("id"))
    perf_child = next((item for item in child_runs if item.benchmark_kind == "PERF"), None)
    quality_child = next((item for item in child_runs if item.benchmark_kind == "QUALITY"), None)
    display_run = run
    if run.benchmark_kind == "QUALITY":
        display_run = run
    elif run.benchmark_kind == "COMPLETE" and perf_child and quality_child:
        display_run = run

    return render(request, "orchestrator/benchmark_detail.html", {
        "run": run,
        "display_run": display_run,
        "perf_child": perf_child,
        "quality_child": quality_child,
        "history_model_query": benchmark_history_model_query(run),
    })


@login_required
def benchmark_status_view(request, run_id: int):
    run = get_object_or_404(BenchmarkRun, id=run_id)
    perf_child = run.child_runs.filter(benchmark_kind="PERF").first()
    quality_child = run.child_runs.filter(benchmark_kind="QUALITY").first()
    return JsonResponse({
        "id": run.id,
        "status": run.status,
        "benchmark_kind": run.benchmark_kind,
        "error_message": run.error_message,
        "warnings": run.quality_warnings,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "summaries": run.summaries,
        "quality_metrics": run.quality_metrics,
        "perf_child_id": perf_child.id if perf_child else None,
        "quality_child_id": quality_child.id if quality_child else None,
        "perf_summaries": perf_child.summaries if perf_child else [],
        "quality_metrics_child": quality_child.quality_metrics if quality_child else {},
    })


@login_required
def delete_benchmark_view(request, run_id: int):
    if request.method != "POST":
        return redirect("benchmark_history")

    try:
        delete_benchmark_run(run_id)
        messages.success(request, f"Benchmark #{run_id} deleted.")
    except ValueError as exc:
        messages.error(request, str(exc))
    except Exception:
        messages.error(request, "Failed to delete benchmark run.")

    return redirect(_safe_redirect_target(request, request.POST.get("next")))


def _safe_redirect_target(request, raw_next: str | None) -> str:
    if not raw_next:
        return reverse("benchmark_history")

    next_url = str(raw_next).strip()
    if not next_url.startswith("/") or next_url.startswith("//"):
        return reverse("benchmark_history")

    if url_has_allowed_host_and_scheme(
        next_url,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return next_url
    return reverse("benchmark_history")


@login_required
def benchmark_history_view(request):
    filters = parse_benchmark_history_query(request.GET)
    queryset = filter_benchmark_runs(filters)
    runs, page_obj = paginate_benchmark_runs(queryset, filters["page"])
    rows = [benchmark_run_list_row(run, filters["scenario"]) for run in runs]
    chart_runs = runs_for_chart_filters(filters)
    chart_data = chart_series_for_runs(chart_runs, scenario=filters["scenario"])

    return render(request, "orchestrator/benchmark_history.html", {
        "filters": filters,
        "rows": rows,
        "page_obj": page_obj,
        "filter_options": list_filter_options(),
        "preset_keys": list_distinct_preset_keys(BenchmarkRun.objects.all()),
        "chart_data_json": json.dumps(chart_data),
        "history_query": build_benchmark_history_query(
            **{**filters, "page": None},
        ),
    })


@login_required
def benchmark_compare_view(request):
    filters = parse_benchmark_history_query(request.GET)
    preset_key = filters.get("preset_key") or None
    completed = BenchmarkRun.objects.filter(status="COMPLETED").select_related("instance")
    gateway_port = settings.NADIR_GATEWAY_PORT
    suggested_pairs = find_comparison_candidates(
        completed,
        preset_key=preset_key,
        gateway_port=gateway_port,
    )

    run_a_id = str(request.GET.get("run_a") or "").strip()
    run_b_id = str(request.GET.get("run_b") or "").strip()
    run_a = run_b = None
    rows: list[dict] = []
    chart_overlay_json = "{}"

    if run_a_id.isdigit() and run_b_id.isdigit():
        run_a = get_object_or_404(BenchmarkRun, id=int(run_a_id), status="COMPLETED")
        run_b = get_object_or_404(BenchmarkRun, id=int(run_b_id), status="COMPLETED")
        rows = comparison_rows(run_a, run_b)
        chart_overlay_json = json.dumps(_comparison_chart_payload(run_a, run_b, gateway_port))

    completed_runs = list(completed.order_by("-completed_at")[:100])
    completed_run_options = [
        {"id": run.id, "label": benchmark_run_label(run, gateway_port)}
        for run in completed_runs
    ]
    suggested_pair_labels = [
        (run_a, run_b, comparison_pair_label(run_a, run_b, gateway_port))
        for run_a, run_b in suggested_pairs
    ]

    return render(request, "orchestrator/benchmark_compare.html", {
        "filters": filters,
        "suggested_pairs": suggested_pair_labels,
        "completed_runs": completed_runs,
        "completed_run_options": completed_run_options,
        "gateway_port": gateway_port,
        "run_a": run_a,
        "run_b": run_b,
        "comparison_rows": rows,
        "chart_overlay_json": chart_overlay_json,
        "preset_keys": list_distinct_preset_keys(completed),
    })


@login_required
def benchmark_compare_export_view(request):
    run_a_id = str(request.GET.get("run_a") or "").strip()
    run_b_id = str(request.GET.get("run_b") or "").strip()
    if not run_a_id.isdigit() or not run_b_id.isdigit():
        return JsonResponse({"error": "run_a and run_b are required."}, status=400)

    run_a = get_object_or_404(BenchmarkRun, id=int(run_a_id))
    run_b = get_object_or_404(BenchmarkRun, id=int(run_b_id))
    snapshot = build_comparison_snapshot(run_a, run_b)
    response = HttpResponse(
        json.dumps(snapshot, indent=2),
        content_type="application/json; charset=utf-8",
    )
    response["X-Content-Type-Options"] = "nosniff"
    response["Content-Disposition"] = (
        f'attachment; filename="bench_compare_{run_a.id}_vs_{run_b.id}.json"'
    )
    return response


def _comparison_chart_payload(
    run_a: BenchmarkRun,
    run_b: BenchmarkRun,
    gateway_port: int,
) -> dict:
    """Build grouped bar chart data for two benchmark runs."""
    rows = comparison_rows(run_a, run_b)
    labels = [row["scenario"] for row in rows]

    def metric(row: dict | None, key: str) -> float | None:
        if not row:
            return None
        raw = row.get(key)
        if raw is None or raw == "N/A":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    return {
        "labels": labels,
        "left_label": _run_chart_label(run_a, gateway_port),
        "right_label": _run_chart_label(run_b, gateway_port),
        "ttft_p50_ms": {
            "left": [metric(row["left"], "ttft_p50_ms") for row in rows],
            "right": [metric(row["right"], "ttft_p50_ms") for row in rows],
        },
        "aggregate_tps": {
            "left": [metric(row["left"], "aggregate_tps") for row in rows],
            "right": [metric(row["right"], "aggregate_tps") for row in rows],
        },
    }


def _run_chart_label(run: BenchmarkRun, gateway_port: int) -> str:
    model = run.instance.model_name if run.instance_id else (run.model_id or "endpoint")
    kind = benchmark_endpoint_kind(run, gateway_port)
    return f"#{run.id} {kind} · {model}"
