import re
import requests
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from .models import ModelDownload, InferenceInstance, BenchmarkRun
from .downloader import start_model_download
from .model_utils import get_model_capabilities, reconcile_stale_downloads, sync_model_download_status
from .server_manager import (
    get_downloaded_models,
    parse_launch_mode,
    start_instance,
    stop_instance,
    delete_instance,
    check_instance_status,
    get_instance_logs
)
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
        return redirect('dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, "Nom d'utilisateur ou mot de passe incorrect.")
            
    return render(request, 'orchestrator/login.html')

def logout_view(request):
    logout(request)
    return redirect('login')


# Model parsing helper
def parse_hf_model(repo_id, tags=None):
    name = repo_id.split('/')[-1]
    
    # 1. Size parameter extraction (e.g. 8B, 70B, 1.5B)
    param_match = re.search(r'(\d+(?:\.\d+)?)[Bb]', name)
    param_size = float(param_match.group(1)) if param_match else None
    
    # 2. Quantization extraction (e.g. 4bit, 8bit)
    quant_match = re.search(r'(\d+)bit', name)
    if quant_match:
        bits = int(quant_match.group(1))
    else:
        # Common fallbacks
        if 'fp16' in name.lower() or 'f16' in name.lower():
            bits = 16
        elif 'fp32' in name.lower():
            bits = 32
        elif 'q4' in name.lower():
            bits = 4
        elif 'q8' in name.lower():
            bits = 8
        else:
            bits = 16

    # 3. Use case extraction
    is_chat = 'instruct' in name.lower() or 'chat' in name.lower() or 'it' in name.lower().split('-')
    if tags:
        is_chat = is_chat or any(t.lower() in ['conversational', 'text-generation'] for t in tags)
    use_case = "Chat / Instruct" if is_chat else "Text Generation"
    
    # 4. RAM estimation calculation
    if param_size:
        ram_est = param_size * (bits / 8.0) * 1.2
        ram_str = f"{ram_est:.1f} GB"
    else:
        ram_str = "Unknown"
        
    return {
        'repo_id': repo_id,
        'name': name,
        'param_size': f"{param_size}B" if param_size else "Unknown",
        'bits': f"{bits}bit",
        'use_case': use_case,
        'ram_est': ram_str
    }


# Search Hugging Face
@login_required
def search_view(request):
    _ensure_downloads_reconciled()
    sync_model_download_status()
    query = request.GET.get('q', '')
    parsed_models = []
    
    # Call HF API
    url = "https://huggingface.co/api/models"
    params = {
        'author': 'mlx-community',
        'sort': 'downloads',
        'direction': '-1',
        'limit': 24
    }
    if query:
        params['search'] = query
        
    try:
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        models_data = response.json()
    except Exception as e:
        messages.error(request, f"Impossible de contacter l'API Hugging Face : {str(e)}")
        models_data = []
        
    for m in models_data:
        repo_id = m.get('id', '')
        if not repo_id:
            continue
        tags = m.get('tags', [])
        downloads = m.get('downloads', 0)
        likes = m.get('likes', 0)
        
        parsed = parse_hf_model(repo_id, tags)
        parsed['downloads'] = downloads
        parsed['likes'] = likes
        
        # Check download record
        download_record = ModelDownload.objects.filter(repo_id=repo_id).first()
        if download_record:
            parsed['download_status'] = download_record.status
            parsed['error_message'] = download_record.error_message
        else:
            parsed['download_status'] = 'NOT_STARTED'
            
        parsed_models.append(parsed)
        
    return render(request, 'orchestrator/search.html', {
        'models': parsed_models,
        'query': query
    })


# Download Trigger
@login_required
def download_model_view(request):
    if request.method == 'POST':
        repo_id = request.POST.get('repo_id')
        if repo_id:
            start_model_download(repo_id)
            messages.success(request, f"Le téléchargement du modèle {repo_id} a démarré en arrière-plan.")
        else:
            messages.error(request, "Aucun identifiant de modèle spécifié.")
    return redirect('search')


# Instances / Dashboard View
@login_required
def dashboard_view(request):
    _ensure_downloads_reconciled()
    sync_model_download_status()
    downloaded_models = [
        {
            "name": model_name,
            **get_model_capabilities(model_name),
        }
        for model_name in get_downloaded_models()
    ]
    
    # Fetch active downloads to display in the UI
    downloading_models = ModelDownload.objects.filter(status='DOWNLOADING').order_by('-created_at')
    
    # Fetch all active instances and verify status
    instances = InferenceInstance.objects.all().order_by('-created_at')
    for instance in instances:
        check_instance_status(instance)
        
    return render(request, 'orchestrator/instances.html', {
        'downloaded_models': downloaded_models,
        'downloading_models': downloading_models,
        'instances': instances
    })



# Start Instance Trigger
@login_required
def start_instance_view(request):
    if request.method == 'POST':
        model_name = request.POST.get('model_name')
        port_raw = request.POST.get('port')
        launch_mode_raw = request.POST.get('launch_mode', 'TEXT')
        
        port = None
        if port_raw and port_raw.strip():
            try:
                port = int(port_raw)
            except ValueError:
                messages.error(request, "Le port spécifié doit être un nombre valide.")
                return redirect('dashboard')

        try:
            launch_mode = parse_launch_mode(launch_mode_raw)
            instance = start_instance(model_name, port, launch_mode)
            mode_labels = {
                "MULTIMODAL": "Multimodal (mlx_vlm)",
                "EMBEDDING": "Embeddings (mlx-embeddings)",
                "RERANKER": "Rerank (local-reranker)",
                "IMAGE": "Image (mflux)",
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
            
    return redirect('dashboard')


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
            
    return redirect('dashboard')


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

    return redirect('dashboard')


# Logs View API / AJAX page
@login_required
def logs_view(request, model_name, port):
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
