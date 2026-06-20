"""Build orchestrator/data/model_registry.json from Hugging Face metadata."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from orchestrator.image_model_profiles import parse_readme_inference_hints_from_text
from orchestrator.model_registry import REGISTRY_PATH, load_model_registry, resolve_registry_family
from orchestrator.model_utils import get_folder_name

HF_API_URL = "https://huggingface.co/api/models"
HF_RAW_README_URL = "https://huggingface.co/{repo_id}/resolve/main/README.md"

PIPELINE_FAMILY_FALLBACK: dict[str, str] = {
    "text-to-speech": "kokoro_tts",
    "automatic-speech-recognition": "whisper_stt",
    "image-text-to-text": "gemma4_vlm",
    "text-generation": "qwen_text",
    "feature-extraction": "qwen3_embedding",
    "text-ranking": "jina_reranker",
    "text-to-image": "flux_lite",
}

BASE_MODEL_LINE = re.compile(r"^base_model:\s*(.+)$", re.MULTILINE)
ORIGINAL_MODEL_LINK = re.compile(
    r"Original Model Link\s*:\s*\[[^\]]+\]\((https://huggingface.co/([^)]+))\)",
    re.IGNORECASE,
)
CONVERTED_FROM_LINK = re.compile(
    r"converted (?:to MLX format )?from [`\[]?(?:<)?(?:https://huggingface.co/)?([^`\]\)>]+)",
    re.IGNORECASE,
)


def fetch_top_mlx_models(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most downloaded models from mlx-community."""
    params = {
        "author": "mlx-community",
        "sort": "downloads",
        "direction": "-1",
        "limit": limit,
    }
    response = requests.get(HF_API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected Hugging Face API response.")
    return payload


def fetch_model_card(repo_id: str) -> dict[str, Any]:
    """Fetch extended model card metadata from Hugging Face."""
    response = requests.get(f"{HF_API_URL}/{repo_id}", timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected model card for {repo_id}.")
    return payload


def fetch_readme_text(repo_id: str) -> str:
    """Download README.md raw content for a Hugging Face repo."""
    response = requests.get(
        HF_RAW_README_URL.format(repo_id=repo_id),
        timeout=30,
    )
    if response.status_code == 404:
        return ""
    response.raise_for_status()
    return response.text


def _normalize_repo_reference(value: str) -> str:
    cleaned = value.strip().strip("`[]<>\"'")
    cleaned = cleaned.removeprefix("https://huggingface.co/")
    cleaned = cleaned.rstrip("/")
    if cleaned.startswith("datasets/"):
        return ""
    return cleaned


def extract_upstream_repo(
    card: dict[str, Any],
    readme_text: str,
) -> str:
    """Resolve upstream repo id from card metadata or README links."""
    card_data = card.get("cardData") or {}
    base_model = card_data.get("base_model") or card.get("base_model")
    if isinstance(base_model, list) and base_model:
        return _normalize_repo_reference(str(base_model[0]))
    if isinstance(base_model, str) and base_model.strip():
        return _normalize_repo_reference(base_model)

    for match in ORIGINAL_MODEL_LINK.finditer(readme_text):
        return _normalize_repo_reference(match.group(2))

    converted = CONVERTED_FROM_LINK.search(readme_text)
    if converted:
        return _normalize_repo_reference(converted.group(1))

    yaml_match = BASE_MODEL_LINE.search(readme_text)
    if yaml_match:
        return _normalize_repo_reference(yaml_match.group(1))

    return ""


def infer_family_id(
    folder_name: str,
    pipeline_tag: str,
) -> str | None:
    """Map a model folder name and pipeline tag to a registry family."""
    family = resolve_registry_family(folder_name)
    if family:
        return family

    pipeline = (pipeline_tag or "").strip().lower()
    if pipeline in PIPELINE_FAMILY_FALLBACK:
        return PIPELINE_FAMILY_FALLBACK[pipeline]

    lowered = folder_name.lower()
    if "schnell" in lowered:
        return "flux_schnell"
    if "kokoro" in lowered:
        return "kokoro_tts"
    if "whisper" in lowered:
        return "whisper_stt"
    if "reranker" in lowered:
        return "jina_reranker"
    if "embedding" in lowered:
        return "qwen3_embedding"
    if "gemma-4" in lowered or "gemma4" in lowered:
        return "gemma4_vlm"
    if "flux" in lowered and "lite" in lowered:
        return "flux_lite"
    return None


def build_model_entry(
    repo_id: str,
    card: dict[str, Any],
    mlx_readme: str,
    upstream_readme: str = "",
) -> dict[str, Any] | None:
    """Build one registry model entry from HF metadata."""
    folder_name = get_folder_name(repo_id)
    pipeline_tag = str(card.get("pipeline_tag") or "")
    family_id = infer_family_id(folder_name, pipeline_tag)
    if not family_id:
        return None

    upstream = extract_upstream_repo(card, mlx_readme)
    sources = ["hf:api"]
    if mlx_readme.strip():
        sources.append("readme:mlx")
    if upstream:
        sources.append(f"upstream:{upstream}")

    entry: dict[str, Any] = {
        "family": family_id,
        "repo_id": repo_id,
        "sources": sources,
    }
    if upstream:
        entry["upstream"] = upstream
    if pipeline_tag:
        entry["pipeline_tag"] = pipeline_tag

    readme_hints = parse_readme_inference_hints_from_text(mlx_readme)
    if readme_hints:
        entry["readme_hints"] = readme_hints

    if upstream_readme.strip():
        upstream_hints = parse_readme_inference_hints_from_text(upstream_readme)
        if upstream_hints:
            entry["upstream_readme_hints"] = upstream_hints
            sources.append("readme:upstream")

    entry["sources"] = sources
    return entry


def collect_registry_models(
    limit: int,
    *,
    fetch_upstream_readme: bool = False,
    sleep_seconds: float = 0.2,
) -> dict[str, dict[str, Any]]:
    """Fetch HF top models and build registry model entries keyed by folder name."""
    models: dict[str, dict[str, Any]] = {}
    skipped = 0

    for item in fetch_top_mlx_models(limit):
        repo_id = str(item.get("id") or "").strip()
        if not repo_id.startswith("mlx-community/"):
            continue

        folder_name = get_folder_name(repo_id)
        try:
            card = fetch_model_card(repo_id)
            mlx_readme = fetch_readme_text(repo_id)
            upstream = extract_upstream_repo(card, mlx_readme)
            upstream_readme = ""
            if fetch_upstream_readme and upstream:
                upstream_readme = fetch_readme_text(upstream)
                time.sleep(sleep_seconds)

            entry = build_model_entry(repo_id, card, mlx_readme, upstream_readme)
            if entry is None:
                skipped += 1
                continue
            models[folder_name] = entry
        except Exception:
            skipped += 1
            continue
        finally:
            time.sleep(sleep_seconds)

    if skipped:
        print(f"Skipped {skipped} models (unknown family or fetch error).")
    return models


def collect_local_model_entries(models_dir: Path) -> dict[str, dict[str, Any]]:
    """Build entries from locally installed model folders."""
    entries: dict[str, dict[str, Any]] = {}
    if not models_dir.is_dir():
        return entries

    for folder in sorted(models_dir.iterdir()):
        if not folder.is_dir():
            continue
        readme_path = folder / "README.md"
        readme_text = ""
        if readme_path.is_file():
            readme_text = readme_path.read_text(encoding="utf-8", errors="replace")

        card: dict[str, Any] = {}
        if readme_text:
            yaml_match = BASE_MODEL_LINE.search(readme_text)
            if yaml_match:
                card["base_model"] = yaml_match.group(1).strip()

        pipeline_match = re.search(r"^pipeline_tag:\s*(.+)$", readme_text, re.MULTILINE)
        if pipeline_match:
            card["pipeline_tag"] = pipeline_match.group(1).strip()

        repo_match = re.search(r"^#\s*mlx-community/([^\s]+)", readme_text, re.MULTILINE)
        repo_id = f"mlx-community/{repo_match.group(1)}" if repo_match else ""

        entry = build_model_entry(repo_id or f"local/{folder.name}", card, readme_text)
        if entry:
            if repo_id:
                entry["repo_id"] = repo_id
            entries[folder.name] = entry
    return entries


def merge_registry(
    existing: dict[str, Any],
    generated_models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Merge generated model entries into an existing registry payload."""
    merged = {
        "version": existing.get("version", 1),
        "families": dict(existing.get("families") or {}),
        "models": dict(existing.get("models") or {}),
        "patterns": list(existing.get("patterns") or []),
    }

    for folder_name, entry in generated_models.items():
        current = dict(merged["models"].get(folder_name) or {})
        current.update(entry)
        merged["models"][folder_name] = current
    return merged


def write_registry(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_registry_file(
    *,
    limit: int = 50,
    output_path: Path | None = None,
    models_dir: Path | None = None,
    include_local: bool = True,
    fetch_upstream_readme: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fetch HF models, merge with existing registry, and optionally write output."""
    target = output_path or REGISTRY_PATH
    existing = load_model_registry()

    generated = collect_registry_models(
        limit,
        fetch_upstream_readme=fetch_upstream_readme,
    )
    if include_local and models_dir is not None:
        generated.update(collect_local_model_entries(models_dir))

    merged = merge_registry(existing, generated)
    if not dry_run:
        write_registry(merged, target)
    return merged
