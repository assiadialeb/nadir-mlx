import json
import re
from urllib.parse import urlencode

import requests
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
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
from .ui_selectors import list_installed_models, models_by_server_type_json
from .benchmark_service import parse_benchmark_form, start_benchmark

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
            messages.error(request, "Nom d'utilisateur ou mot de passe incorrect.")
            
    return render(request, 'orchestrator/login.html')

def logout_view(request):
    logout(request)
    return redirect('login')


# Model parsing helper
def _extract_param_size(name: str) -> float | None:
    param_match = re.search(r"(\d+(?:\.\d+)?)[Bb]", name)
    return float(param_match.group(1)) if param_match else None


def _extract_quantization_bits(name: str) -> int:
    quant_match = re.search(r"(\d+)bit", name)
    if quant_match:
        return int(quant_match.group(1))

    lowered = name.lower()
    if "fp16" in lowered or "f16" in lowered:
        return 16
    if "fp32" in lowered:
        return 32
    if "q4" in lowered:
        return 4
    if "q8" in lowered:
        return 8
    return 16


def _detect_use_case(name: str, tags: list[str] | None) -> str:
    is_chat = "instruct" in name.lower() or "chat" in name.lower() or "it" in name.lower().split("-")
    if tags:
        is_chat = is_chat or any(t.lower() in ["conversational", "text-generation"] for t in tags)
    return "Chat / Instruct" if is_chat else "Text Generation"


def _estimate_ram_gb(param_size: float | None, bits: int) -> str:
    if param_size is None:
        return "Unknown"
    ram_est = param_size * (bits / 8.0) * 1.2
    return f"{ram_est:.1f} GB"


def parse_hf_model(repo_id, tags=None):
    name = repo_id.split('/')[-1]
    param_size = _extract_param_size(name)
    bits = _extract_quantization_bits(name)
    use_case = _detect_use_case(name, tags)
    ram_str = _estimate_ram_gb(param_size, bits)

    return {
        'repo_id': repo_id,
        'name': name,
        'param_size': f"{param_size}B" if param_size else "Unknown",
        'bits': f"{bits}bit",
        'use_case': use_case,
        'ram_est': ram_str
    }


def _fetch_hf_models(query: str) -> list[dict]:
    url = "https://huggingface.co/api/models"
    params = {
        'author': 'mlx-community',
        'sort': 'downloads',
        'direction': '-1',
        'limit': 24,
    }
    if query:
        params['search'] = query

    response = requests.get(url, params=params, timeout=8)
    response.raise_for_status()
    parsed_models = []
    for model_data in response.json():
        repo_id = model_data.get('id', '')
        if not repo_id:
            continue
        parsed = parse_hf_model(repo_id, model_data.get('tags', []))
        parsed['downloads'] = model_data.get('downloads', 0)
        parsed['likes'] = model_data.get('likes', 0)
        download_record = ModelDownload.objects.filter(repo_id=repo_id).first()
        if download_record:
            parsed['download_status'] = download_record.status
            parsed['error_message'] = download_record.error_message
        else:
            parsed['download_status'] = 'NOT_STARTED'
        parsed_models.append(parsed)
    return parsed_models


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

    installed_models = list_installed_models()
    downloading_models = ModelDownload.objects.filter(status='DOWNLOADING').order_by('-created_at')
    query = request.GET.get('q', '')
    hf_models = []

    if active_tab == 'hub':
        try:
            hf_models = _fetch_hf_models(query)
        except Exception as exc:
            messages.error(request, f"Impossible de contacter l'API Hugging Face : {exc}")

    return render(request, 'orchestrator/models.html', {
        'active_tab': active_tab,
        'installed_models': installed_models,
        'downloading_models': downloading_models,
        'models': hf_models,
        'query': query,
    })


@login_required
def delete_model_view(request):
    if request.method != 'POST':
        return redirect('models')

    folder_name = (request.POST.get('model_name') or '').strip()
    if not folder_name:
        messages.error(request, "Aucun modèle sélectionné.")
        return redirect(f"{reverse('models')}?tab=installed")

    try:
        result = delete_installed_model(folder_name)
        messages.success(
            request,
            (
                f"Modèle {result.folder_name} supprimé "
                f"({result.instances_removed} serveur(s), "
                f"{result.benchmark_runs_removed} benchmark(s), "
                f"{result.log_files_removed} log(s))."
            ),
        )
    except ValueError as exc:
        messages.error(request, str(exc))
    except Exception as exc:
        messages.error(request, f"Erreur lors de la suppression du modèle : {exc}")

    return redirect(f"{reverse('models')}?tab=installed")


# Download Trigger
@login_required
def download_model_view(request):
    if request.method == 'POST':
        repo_id = request.POST.get('repo_id')
        if repo_id:
            try:
                start_model_download(repo_id)
                messages.success(request, f"Le téléchargement du modèle {repo_id} a démarré en arrière-plan.")
            except ValueError as exc:
                messages.error(request, str(exc))
        else:
            messages.error(request, "Aucun identifiant de modèle spécifié.")
    tab = request.POST.get('tab', 'hub')
    return redirect(f"{reverse('models')}?{urlencode({'tab': tab})}")


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
            messages.error(request, "Veuillez sélectionner un modèle.")
            return redirect('servers')

        port = None
        if port_raw and port_raw.strip():
            try:
                port = int(port_raw)
            except ValueError:
                messages.error(request, "Le port spécifié doit être un nombre valide.")
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
            mode_label = mode_labels.get(instance.launch_mode, "Texte (mlx_lm)")
            messages.success(
                request,
                f"Instance {mode_label} lancée sur le port {instance.port} (PID: {instance.pid}).",
            )
        except ValueError as e:
            messages.error(request, str(e))
        except Exception as e:
            messages.error(request, f"Erreur lors du lancement de l'instance : {str(e)}")
            
    return redirect('servers')


# Stop Instance Trigger
@login_required
def stop_instance_view(request, instance_id):
    if request.method == 'POST':
        instance = get_object_or_404(InferenceInstance, id=instance_id)
        try:
            stop_instance(instance)
            messages.success(request, f"Instance sur le port {instance.port} arrêtée.")
        except RuntimeError as exc:
            messages.error(request, str(exc))
        except Exception as e:
            messages.error(request, f"Erreur lors de l'arrêt de l'instance : {str(e)}")
            
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
                f"Instance {model_name} (port {port}) supprimée.",
            )
        except Exception as e:
            messages.error(
                request,
                f"Erreur lors de la suppression de l'instance : {str(e)}",
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
        )
        messages.success(
            request,
            f"Benchmark #{run.id} started against {run.endpoint_url}.",
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
    return render(request, "orchestrator/benchmark_detail.html", {
        "run": run,
    })


@login_required
def benchmark_status_view(request, run_id: int):
    run = get_object_or_404(BenchmarkRun, id=run_id)
    return JsonResponse({
        "id": run.id,
        "status": run.status,
        "error_message": run.error_message,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "summaries": run.summaries,
    })
