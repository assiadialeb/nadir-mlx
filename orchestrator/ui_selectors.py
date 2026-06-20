"""Read-only selectors for orchestrator UI pages."""

from __future__ import annotations

import json
import re
from typing import Any, Literal
from urllib.parse import urlencode

import requests
from django.db.models import Count

from orchestrator.model_registry import (
    get_registry_family,
    load_model_registry,
    resolve_model_registry_profile,
    resolve_registry_family,
)
from orchestrator.model_utils import get_model_capabilities, get_model_folder_size_bytes
from orchestrator.models import InferenceInstance, ModelDownload
from orchestrator.registry_builder import infer_family_id
from orchestrator.server_types import SERVER_TYPES

InstalledSort = Literal["name_asc", "name_desc", "size_asc", "size_desc"]

HF_FETCH_LIMIT_DEFAULT = 24
HF_FETCH_LIMIT_FILTERED = 100

SORT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("name_asc", "Nom A → Z"),
    ("name_desc", "Nom Z → A"),
    ("size_desc", "Taille (plus grand)"),
    ("size_asc", "Taille (plus petit)"),
)

CAPABILITY_FILTER_IDS: frozenset[str] = frozenset(server_type["id"] for server_type in SERVER_TYPES)
CAPABILITY_TO_KEY: dict[str, str] = {
    server_type["id"]: server_type["capability"] for server_type in SERVER_TYPES
}

LAUNCH_MODE_CAPABILITY_KEYS: dict[str, str] = {
    "TEXT": "supports_text",
    "MULTIMODAL": "supports_multimodal",
    "EMBEDDING": "supports_embedding",
    "RERANKER": "supports_rerank",
    "IMAGE": "supports_image",
    "TTS": "supports_tts",
    "STT": "supports_stt",
}


def _empty_capability_flags() -> dict[str, bool]:
    return {key: False for key in LAUNCH_MODE_CAPABILITY_KEYS.values()}


def launch_modes_to_capabilities(launch_modes: list[str]) -> dict[str, bool]:
    """Map registry launch modes to UI capability flags."""
    capabilities = _empty_capability_flags()
    for mode in launch_modes:
        capability_key = LAUNCH_MODE_CAPABILITY_KEYS.get(mode)
        if capability_key:
            capabilities[capability_key] = True
    return capabilities


def infer_hf_model_capabilities(
    folder_name: str,
    *,
    pipeline_tag: str = "",
    tags: list[str] | None = None,
) -> dict[str, bool]:
    """Infer launch capabilities for a Hugging Face repo from registry and metadata."""
    registry = load_model_registry()
    patterns = registry.get("patterns") or []

    family_id = resolve_registry_family(folder_name) or infer_family_id(
        folder_name,
        pipeline_tag,
        patterns,
    )

    launch_modes: list[str] = []
    if family_id:
        family = get_registry_family(family_id)
        if family:
            launch_modes = list(family.get("launch_modes") or [])

    if not launch_modes:
        launch_modes = _heuristic_launch_modes(folder_name, pipeline_tag, tags)

    return launch_modes_to_capabilities(launch_modes)


def _heuristic_launch_modes(
    folder_name: str,
    pipeline_tag: str,
    tags: list[str] | None,
) -> list[str]:
    """Fallback capability inference when the registry has no match."""
    lowered = folder_name.lower()
    pipeline = (pipeline_tag or "").lower()

    if "kokoro" in lowered or pipeline == "text-to-speech":
        return ["TTS"]
    if "whisper" in lowered or "parakeet" in lowered or pipeline == "automatic-speech-recognition":
        return ["STT"]
    if "reranker" in lowered or pipeline == "text-ranking":
        return ["RERANKER"]
    if "embedding" in lowered or pipeline == "feature-extraction":
        return ["EMBEDDING"]
    if "flux" in lowered or "schnell" in lowered or pipeline == "text-to-image":
        return ["IMAGE"]
    if pipeline in {"image-text-to-text", "any-to-any"}:
        return ["MULTIMODAL"]
    if "gemma-4" in lowered or "gemma-3" in lowered:
        return ["MULTIMODAL"]

    tag_tokens = {str(tag).lower() for tag in (tags or [])}
    if tag_tokens & {"image-text-to-text", "any-to-any"}:
        return ["MULTIMODAL"]

    return ["TEXT"]


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
        is_chat = is_chat or any(tag.lower() in ["conversational", "text-generation"] for tag in tags)
    return "Chat / Instruct" if is_chat else "Text Generation"


def _estimate_ram_gb(param_size: float | None, bits: int) -> str:
    if param_size is None:
        return "Unknown"
    ram_est = param_size * (bits / 8.0) * 1.2
    return f"{ram_est:.1f} GB"


def _estimate_model_size_bytes(param_size: float | None, bits: int) -> int:
    if param_size is None:
        return 0
    return int(param_size * 1_000_000_000 * (bits / 8.0))


def parse_hf_model_from_api(
    model_data: dict[str, Any],
    download_records: dict[str, ModelDownload],
) -> dict[str, Any]:
    """Build a UI-ready Hugging Face model dict from the HF API payload."""
    repo_id = str(model_data.get("id") or "").strip()
    name = repo_id.split("/")[-1]
    tags = model_data.get("tags") or []
    pipeline_tag = str(model_data.get("pipeline_tag") or "")

    param_size = _extract_param_size(name)
    bits = _extract_quantization_bits(name)
    estimated_size_bytes = _estimate_model_size_bytes(param_size, bits)

    parsed: dict[str, Any] = {
        "repo_id": repo_id,
        "name": name,
        "param_size": f"{param_size}B" if param_size else "Unknown",
        "bits": f"{bits}bit",
        "use_case": _detect_use_case(name, tags),
        "ram_est": _estimate_ram_gb(param_size, bits),
        "estimated_size_bytes": estimated_size_bytes,
        "disk_size_bytes": estimated_size_bytes,
        "downloads": model_data.get("downloads", 0),
        "likes": model_data.get("likes", 0),
        **infer_hf_model_capabilities(name, pipeline_tag=pipeline_tag, tags=tags),
    }

    download_record = download_records.get(repo_id)
    if download_record:
        parsed["download_status"] = download_record.status
        parsed["error_message"] = download_record.error_message
    else:
        parsed["download_status"] = "NOT_STARTED"
        parsed["error_message"] = ""

    return parsed


def fetch_hf_models(query: str = "", *, limit: int = HF_FETCH_LIMIT_DEFAULT) -> list[dict[str, Any]]:
    """Fetch mlx-community models from the Hugging Face API."""
    params: dict[str, Any] = {
        "author": "mlx-community",
        "sort": "downloads",
        "direction": "-1",
        "limit": limit,
    }
    if query:
        params["search"] = query

    response = requests.get("https://huggingface.co/api/models", params=params, timeout=8)
    response.raise_for_status()

    payload = response.json()
    repo_ids = [str(item.get("id") or "").strip() for item in payload if item.get("id")]
    download_records = {
        record.repo_id: record
        for record in ModelDownload.objects.filter(repo_id__in=repo_ids)
    }

    models: list[dict[str, Any]] = []
    for model_data in payload:
        repo_id = str(model_data.get("id") or "").strip()
        if not repo_id:
            continue
        models.append(parse_hf_model_from_api(model_data, download_records))
    return models


def _model_size_bytes(model: dict[str, Any]) -> int:
    return int(model.get("disk_size_bytes") or model.get("estimated_size_bytes") or 0)


def _model_search_text(model: dict[str, Any]) -> str:
    parts = [model.get("name", ""), model.get("repo_id", "")]
    return " ".join(str(part) for part in parts if part).lower()


def list_installed_models() -> list[dict[str, Any]]:
    """Return downloaded models enriched with capability flags."""
    from orchestrator.server_manager import get_downloaded_models

    instance_counts = {
        row["model_name"]: row["count"]
        for row in InferenceInstance.objects.values("model_name").annotate(count=Count("id"))
    }

    models = []
    for model_name in get_downloaded_models():
        disk_size_bytes = get_model_folder_size_bytes(model_name)
        models.append(
            {
                "name": model_name,
                "instance_count": instance_counts.get(model_name, 0),
                "disk_size_bytes": disk_size_bytes,
                "disk_size_label": format_disk_size(disk_size_bytes),
                **get_model_capabilities(model_name),
                "registry": resolve_model_registry_profile(
                    model_name,
                    _primary_launch_mode(model_name),
                ),
            }
        )
    return models


def parse_installed_models_query(params: dict[str, Any]) -> tuple[str, str, str]:
    """Parse q, cap, and sort query params for the installed models tab."""
    query = str(params.get("q") or "").strip()
    capability = str(params.get("cap") or "").strip().upper()
    if capability and capability not in CAPABILITY_FILTER_IDS:
        capability = ""

    sort = str(params.get("sort") or "name_asc").strip().lower()
    valid_sorts = {option[0] for option in SORT_OPTIONS}
    if sort not in valid_sorts:
        sort = "name_asc"

    return query, capability, sort


def filter_installed_models(
    models: list[dict[str, Any]],
    *,
    query: str = "",
    capability: str = "",
) -> list[dict[str, Any]]:
    """Filter models by folder/repo name and capability."""
    filtered = models
    if query:
        needle = query.lower()
        filtered = [
            model
            for model in filtered
            if needle in _model_search_text(model)
        ]

    if capability:
        capability_key = CAPABILITY_TO_KEY[capability]
        filtered = [model for model in filtered if model.get(capability_key)]

    return filtered


def sort_installed_models(
    models: list[dict[str, Any]],
    sort: str = "name_asc",
) -> list[dict[str, Any]]:
    """Sort models by name or estimated/disk size."""
    if sort == "name_desc":
        return sorted(models, key=lambda model: model["name"].lower(), reverse=True)
    if sort == "size_desc":
        return sorted(
            models,
            key=lambda model: (_model_size_bytes(model), model["name"].lower()),
            reverse=True,
        )
    if sort == "size_asc":
        return sorted(
            models,
            key=lambda model: (_model_size_bytes(model), model["name"].lower()),
        )
    return sorted(models, key=lambda model: model["name"].lower())


def apply_installed_models_filters(
    models: list[dict[str, Any]],
    *,
    query: str = "",
    capability: str = "",
    sort: str = "name_asc",
) -> list[dict[str, Any]]:
    """Filter then sort the installed models list."""
    filtered = filter_installed_models(models, query=query, capability=capability)
    return sort_installed_models(filtered, sort)


def build_installed_models_query(
    *,
    query: str = "",
    capability: str = "",
    sort: str = "name_asc",
    **overrides: str,
) -> str:
    """Build a query string for the installed models tab."""
    return build_models_tab_query(
        "installed",
        query=query,
        capability=capability,
        sort=sort,
        **overrides,
    )


def build_models_tab_query(
    tab: str,
    *,
    query: str = "",
    capability: str = "",
    sort: str = "name_asc",
    **overrides: str,
) -> str:
    """Build a query string for a models page tab."""
    params: dict[str, str] = {"tab": tab}
    if query:
        params["q"] = query
    if capability:
        params["cap"] = capability
    if sort and sort != "name_asc":
        params["sort"] = sort
    params.update({key: value for key, value in overrides.items() if value})
    return urlencode(params)


def hf_fetch_limit_for_filters(query: str, capability: str, sort: str) -> int:
    """Use a larger HF page size when local filters are active."""
    if query or capability or sort != "name_asc":
        return HF_FETCH_LIMIT_FILTERED
    return HF_FETCH_LIMIT_DEFAULT


def format_disk_size(size_bytes: int) -> str:
    """Human-readable disk size for UI display."""
    if size_bytes <= 0:
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(size_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.1f} {units[unit_index]}"


def _primary_launch_mode(model_name: str) -> str:
    caps = get_model_capabilities(model_name)
    if caps.get("supports_image"):
        return "IMAGE"
    if caps.get("supports_tts"):
        return "TTS"
    if caps.get("supports_stt"):
        return "STT"
    if caps.get("supports_rerank"):
        return "RERANKER"
    if caps.get("supports_embedding"):
        return "EMBEDDING"
    if caps.get("supports_multimodal"):
        return "MULTIMODAL"
    return "TEXT"


def build_models_by_server_type(installed_models: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Map each server type to compatible local model folder names."""
    mapping: dict[str, list[str]] = {}
    for server_type in SERVER_TYPES:
        capability_key = server_type["capability"]
        mapping[server_type["id"]] = sorted(
            model["name"]
            for model in installed_models
            if model.get(capability_key)
        )
    return mapping


def models_by_server_type_json(installed_models: list[dict[str, Any]]) -> str:
    """Serialize compatible models for client-side select filtering."""
    return json.dumps(build_models_by_server_type(installed_models))
