"""Build orchestrator/data/model_registry.json from Hugging Face metadata."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from orchestrator.huggingface_client import huggingface_get
from orchestrator.image_model_profiles import parse_readme_inference_hints_from_text
from orchestrator.model_registry import REGISTRY_PATH, load_model_registry
from orchestrator.model_utils import get_folder_name, validate_hf_repo_id
from orchestrator.security_utils import validate_huggingface_api_url

HF_API_URL = "https://huggingface.co/api/models"
HF_RAW_README_URL = "https://huggingface.co/{repo_id}/resolve/main/README.md"

PIPELINE_FAMILY_FALLBACK: dict[str, str] = {
    "text-to-speech": "kokoro_tts",
    "text-ranking": "jina_reranker",
    "feature-extraction": "qwen3_embedding",
    "text-to-image": "flux_lite",
    "any-to-any": "gemma4_vlm",
}

EXTRA_FAMILIES: dict[str, dict[str, Any]] = {
    "llama_text": {
        "launch_modes": ["TEXT"],
        "server_config": {"max_tokens": 512},
        "sources": ["registry:mlx_lm"],
    },
    "parakeet_stt": {
        "launch_modes": ["STT"],
        "server_config": {"chunk_duration": 30.0},
        "sources": ["registry:mlx-audio"],
    },
    "gemma3_vlm": {
        "launch_modes": ["MULTIMODAL"],
        "server_config": {"max_tokens": 512, "max_kv_size": 8192},
        "sources": ["registry:mlx_vlm"],
    },
    "gemma3_text": {
        "launch_modes": ["TEXT"],
        "server_config": {"max_tokens": 512},
        "sources": ["registry:mlx_lm"],
    },
}

EXTRA_PATTERNS: list[dict[str, Any]] = [
    {"match": "llama", "family": "llama_text"},
    {"match": "meta-llama", "family": "llama_text"},
    {"match": "parakeet", "family": "parakeet_stt"},
    {"match": "gemma-3", "family": "gemma3_vlm"},
    {"match": "gemma-2", "family": "gemma3_text"},
]

ORIGINAL_MODEL_LINK = re.compile(
    r"Original Model Link\s*:\s*\[[^\]]+\]\((https://huggingface.co/([^)]+))\)",
    re.IGNORECASE,
)
CONVERTED_FROM_URL = re.compile(
    r"converted (?:to MLX format )?from\s+\[[^\]]+\]\((https://huggingface.co/([^)]+))\)",
    re.IGNORECASE,
)
CONVERTED_FROM_BACKTICK = re.compile(
    r"from\s+\[`([^`]+)`\]",
    re.IGNORECASE,
)
CONVERTED_FROM_PLAIN = re.compile(
    r"converted (?:to MLX format )?from [`\[]?(?:<)?(?:https://huggingface.co/)?([^`\]\)>]+)",
    re.IGNORECASE,
)
BASE_MODEL_LINE = re.compile(r"^base_model:\s*(.+)$", re.MULTILINE)
BASE_MODEL_LIST_BLOCK = re.compile(
    r"^base_model:\s*\n((?:[ \t]*-\s+.+\n?)+)",
    re.MULTILINE,
)
BASE_MODEL_LIST_ITEM = re.compile(r"^[ \t]*-\s*(.+)$", re.MULTILINE)


def fetch_top_mlx_models(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most downloaded models from mlx-community."""
    params = {
        "author": "mlx-community",
        "sort": "downloads",
        "direction": "-1",
        "limit": limit,
    }
    response = huggingface_get(validate_huggingface_api_url(HF_API_URL), params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected Hugging Face API response.")
    return payload


def fetch_model_card(repo_id: str) -> dict[str, Any]:
    """Fetch extended model card metadata from Hugging Face."""
    safe_repo_id = validate_hf_repo_id(repo_id)
    url = validate_huggingface_api_url(f"{HF_API_URL}/{safe_repo_id}")
    response = huggingface_get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected model card for {repo_id}.")
    return payload


def fetch_readme_text(repo_id: str) -> str:
    """Download README.md raw content for a Hugging Face repo."""
    safe_repo_id = validate_hf_repo_id(repo_id)
    url = validate_huggingface_api_url(HF_RAW_README_URL.format(repo_id=safe_repo_id))
    response = huggingface_get(url, timeout=30)
    if response.status_code == 404:
        return ""
    response.raise_for_status()
    return response.text


def _normalize_repo_reference(value: str) -> str:
    cleaned = value.strip().strip("`[]<>\"'")
    cleaned = cleaned.removeprefix("https://huggingface.co/")
    cleaned = cleaned.lstrip("-").strip()
    cleaned = cleaned.rstrip("/")
    if cleaned.startswith("datasets/") or not cleaned:
        return ""
    if cleaned.startswith(("/", "~")):
        return ""
    if len(cleaned) >= 2 and cleaned[1] == ":" and cleaned[0].isalpha():
        return ""
    if cleaned.count("/") != 1:
        return ""
    return cleaned


def _is_canonical_mlx_repo(repo_id: str | None) -> bool:
    return bool(repo_id) and str(repo_id).startswith("mlx-community/")


def _parse_yaml_base_models(readme_text: str) -> list[str]:
    block = BASE_MODEL_LIST_BLOCK.search(readme_text)
    if block:
        return [
            ref
            for item in BASE_MODEL_LIST_ITEM.findall(block.group(1))
            if (ref := _normalize_repo_reference(item))
        ]

    match = BASE_MODEL_LINE.search(readme_text)
    if match:
        ref = _normalize_repo_reference(match.group(1))
        return [ref] if ref else []
    return []


def _parse_card_base_models(card: dict[str, Any]) -> list[str]:
    card_data = card.get("cardData") or {}
    base_model = card_data.get("base_model") or card.get("base_model")
    if isinstance(base_model, list):
        return [
            ref
            for item in base_model
            if (ref := _normalize_repo_reference(str(item)))
        ]
    if isinstance(base_model, str) and base_model.strip():
        ref = _normalize_repo_reference(base_model)
        return [ref] if ref else []
    return []


def _collect_upstream_candidates(card: dict[str, Any], readme_text: str) -> list[str]:
    """Collect valid upstream refs in README/card priority order."""
    seen: set[str] = set()
    candidates: list[str] = []

    def add(value: str) -> None:
        ref = _normalize_repo_reference(value)
        if ref and ref not in seen:
            seen.add(ref)
            candidates.append(ref)

    for match in ORIGINAL_MODEL_LINK.finditer(readme_text):
        add(match.group(2))

    converted_url = CONVERTED_FROM_URL.search(readme_text)
    if converted_url:
        add(converted_url.group(2))

    converted_backtick = CONVERTED_FROM_BACKTICK.search(readme_text)
    if converted_backtick:
        add(converted_backtick.group(1))

    converted_plain = CONVERTED_FROM_PLAIN.search(readme_text)
    if converted_plain:
        add(converted_plain.group(1))

    for ref in _parse_card_base_models(card):
        add(ref)

    for ref in _parse_yaml_base_models(readme_text):
        add(ref)

    return candidates


def _infer_upstream_from_folder(folder_name: str, family_id: str) -> list[str]:
    """Guess likely Google/Qwen upstream ids from MLX folder naming."""
    name = folder_name.lower()
    inferred: list[str] = []

    qat_match = re.match(r"gemma-(\d+)-([\d.]+b)-it-qat-4bit", name)
    if qat_match and family_id.startswith("gemma3"):
        version, size = qat_match.groups()
        inferred.extend(
            [
                f"google/gemma-{version}-{size}-it-qat-q4_0-unquantized",
                f"google/gemma-{version}-{size}-it-qat-q4_0",
            ]
        )

    gemma4_match = re.match(r"gemma-4-([\w-]+)-it(?:-[\w-]+)?", name)
    if gemma4_match and family_id.startswith("gemma4"):
        suffix = gemma4_match.group(1)
        inferred.append(f"google/gemma-4-{suffix}-it")

    return inferred


def _token_overlap_score(folder: str, candidate_name: str) -> int:
    folder_tokens = {token for token in re.split(r"[^a-z0-9]+", folder) if len(token) >= 2}
    candidate_tokens = {
        token for token in re.split(r"[^a-z0-9]+", candidate_name) if len(token) >= 2
    }
    return len(folder_tokens & candidate_tokens) * 2


def _brand_alignment_score(folder: str, candidate_name: str) -> int:
    score = 0
    if "gemma" in folder and "gemma" in candidate_name:
        score += 6
    if "gemma" in folder and "internvl" in candidate_name:
        score -= 12
    if "llama" in folder and "llama" in candidate_name:
        score += 6
    if "qwen" in folder and "qwen" in candidate_name:
        score += 6
    if "kokoro" in folder and "kokoro" in candidate_name:
        score += 6
    if "parakeet" in folder and "parakeet" in candidate_name:
        score += 6
    return score


def _org_alignment_score(family_id: str, org: str) -> int:
    if family_id.startswith("gemma") and org == "google":
        return 4
    if family_id == "llama_text" and org in {"meta-llama", "mlx-community"}:
        return 3
    if family_id.startswith("qwen") and org == "qwen":
        return 4
    return 0


def _score_upstream_match(folder_name: str, family_id: str, candidate: str) -> int:
    folder = folder_name.lower()
    candidate_lower = candidate.lower()
    candidate_name = candidate_lower.split("/", 1)[-1]
    org = candidate_lower.split("/", 1)[0]

    return (
        _token_overlap_score(folder, candidate_name)
        + _brand_alignment_score(folder, candidate_name)
        + _org_alignment_score(family_id, org)
    )


def extract_upstream_repo(
    card: dict[str, Any],
    readme_text: str,
    *,
    folder_name: str = "",
    family_id: str = "",
) -> str:
    """Resolve upstream repo id; prefer README links, then best folder match."""
    candidates = _collect_upstream_candidates(card, readme_text)
    for ref in _infer_upstream_from_folder(folder_name, family_id):
        if ref not in candidates:
            candidates.append(ref)

    if not candidates:
        return ""
    if not folder_name:
        return candidates[0]

    return max(
        candidates,
        key=lambda candidate: (
            _score_upstream_match(folder_name, family_id, candidate),
            -candidates.index(candidate),
        ),
    )


def _match_pattern_rules(folder_name: str, patterns: list[dict[str, Any]]) -> str | None:
    name = folder_name.lower()
    best_family = ""
    best_score = 0
    for rule in patterns:
        match_token = str(rule.get("match", "")).lower()
        if not match_token or match_token not in name:
            continue
        required = rule.get("requires") or []
        if required and not all(str(token).lower() in name for token in required):
            continue
        score = len(match_token)
        if score > best_score:
            best_score = score
            best_family = str(rule.get("family") or "")
    return best_family or None


def _family_from_audio_tokens(lowered: str) -> str | None:
    if "kokoro" in lowered:
        return "kokoro_tts"
    if "parakeet" in lowered:
        return "parakeet_stt"
    if "whisper" in lowered:
        return "whisper_stt"
    return None


def _family_from_task_tokens(lowered: str) -> str | None:
    if "reranker" in lowered:
        return "jina_reranker"
    if "embedding" in lowered:
        return "qwen3_embedding"
    if "llama" in lowered or "meta-llama" in lowered:
        return "llama_text"
    return None


def _family_from_image_tokens(lowered: str) -> str | None:
    if "schnell" in lowered:
        return "flux_schnell"
    if "flux" in lowered and "lite" in lowered:
        return "flux_lite"
    if "z-image-turbo" in lowered or "z_image_turbo" in lowered:
        return "z_image_turbo"
    return None


def _family_from_gemma_tokens(lowered: str, pipeline: str) -> str | None:
    if "gemma-4" in lowered or "gemma4" in lowered or pipeline == "any-to-any":
        return "gemma4_vlm"
    if "gemma-3" in lowered or "gemma3" in lowered:
        if pipeline in {"image-text-to-text", "any-to-any"}:
            return "gemma3_vlm"
        return "gemma3_text"
    if "gemma-2" in lowered or "gemma2" in lowered:
        return "gemma3_text"
    return None


def _family_from_pipeline_tag(pipeline: str, lowered: str) -> str | None:
    if pipeline == "automatic-speech-recognition":
        return "parakeet_stt" if "parakeet" in lowered else "whisper_stt"
    if "qwen" in lowered and pipeline == "text-generation":
        return "qwen_text"
    return PIPELINE_FAMILY_FALLBACK.get(pipeline)


def infer_family_id(
    folder_name: str,
    pipeline_tag: str,
    patterns: list[dict[str, Any]] | None = None,
) -> str | None:
    """Map folder name and pipeline tag to a registry family (builder logic)."""
    lowered = folder_name.lower()
    pipeline = (pipeline_tag or "").strip().lower()

    for resolver in (
        lambda: _family_from_audio_tokens(lowered),
        lambda: _family_from_task_tokens(lowered),
        lambda: _family_from_image_tokens(lowered),
        lambda: _family_from_gemma_tokens(lowered, pipeline),
        lambda: _family_from_pipeline_tag(pipeline, lowered),
    ):
        family_id = resolver()
        if family_id:
            return family_id

    if patterns:
        return _match_pattern_rules(folder_name, patterns)

    return None


def build_model_entry(
    repo_id: str,
    card: dict[str, Any],
    mlx_readme: str,
    upstream_readme: str = "",
    *,
    patterns: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Build one registry model entry from HF metadata."""
    folder_name = get_folder_name(repo_id) if "/" in repo_id else repo_id
    pipeline_tag = str(card.get("pipeline_tag") or "")
    family_id = infer_family_id(folder_name, pipeline_tag, patterns)
    if not family_id:
        return None

    upstream = extract_upstream_repo(
        card,
        mlx_readme,
        folder_name=folder_name,
        family_id=family_id,
    )
    sources = ["hf:api"]
    if mlx_readme.strip():
        sources.append("readme:mlx")
    if upstream:
        sources.append(f"upstream:{upstream}")

    canonical_repo_id = repo_id if _is_canonical_mlx_repo(repo_id) else ""
    entry: dict[str, Any] = {
        "family": family_id,
        "repo_id": canonical_repo_id or repo_id,
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


def _should_keep_existing_repo_id(merged: dict[str, Any], incoming_repo_id: str) -> bool:
    return _is_canonical_mlx_repo(merged.get("repo_id")) and not _is_canonical_mlx_repo(
        incoming_repo_id
    )


def _merge_registry_field(merged: dict[str, Any], key: str, value: Any) -> None:
    if key == "repo_id" and _should_keep_existing_repo_id(merged, value):
        return
    if key == "upstream" and merged.get("upstream") and not value:
        return
    if key == "sources":
        merged[key] = sorted(set(list(merged.get(key) or []) + list(value or [])))
        return
    if key == "readme_hints" and merged.get(key) and not value:
        return
    merged[key] = value


def merge_model_entry(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Merge model entries without downgrading canonical repo ids or upstream refs."""
    merged = dict(existing)
    for key, value in incoming.items():
        _merge_registry_field(merged, key, value)
    return merged


def _fetch_upstream_readme_entry(
    repo_id: str,
    card: dict[str, Any],
    mlx_readme: str,
    upstream: str,
    *,
    patterns: list[dict[str, Any]] | None,
    sleep_seconds: float,
) -> dict[str, Any] | None:
    upstream_readme = fetch_readme_text(upstream)
    time.sleep(sleep_seconds)
    if not upstream_readme.strip():
        return None
    return build_model_entry(
        repo_id,
        card,
        mlx_readme,
        upstream_readme,
        patterns=patterns,
    )


def _build_registry_model_entry(
    repo_id: str,
    *,
    patterns: list[dict[str, Any]] | None,
    fetch_upstream_readme: bool,
    sleep_seconds: float,
) -> dict[str, Any] | None:
    card = fetch_model_card(repo_id)
    mlx_readme = fetch_readme_text(repo_id)
    entry = build_model_entry(repo_id, card, mlx_readme, patterns=patterns)
    if entry is None:
        return None

    upstream = str(entry.get("upstream") or "")
    if fetch_upstream_readme and upstream:
        upstream_entry = _fetch_upstream_readme_entry(
            repo_id,
            card,
            mlx_readme,
            upstream,
            patterns=patterns,
            sleep_seconds=sleep_seconds,
        )
        if upstream_entry is not None:
            entry = upstream_entry
    return entry


def collect_registry_models(
    limit: int,
    *,
    patterns: list[dict[str, Any]] | None = None,
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
            entry = _build_registry_model_entry(
                repo_id,
                patterns=patterns,
                fetch_upstream_readme=fetch_upstream_readme,
                sleep_seconds=sleep_seconds,
            )
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


def collect_local_model_entries(
    models_dir: Path,
    *,
    patterns: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
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
        pipeline_match = re.search(r"^pipeline_tag:\s*(.+)$", readme_text, re.MULTILINE)
        if pipeline_match:
            card["pipeline_tag"] = pipeline_match.group(1).strip()

        repo_match = re.search(
            r"mlx-community/([A-Za-z0-9._-]+)",
            readme_text,
        )
        repo_id = f"mlx-community/{repo_match.group(1)}" if repo_match else ""

        entry = build_model_entry(
            repo_id or folder.name,
            card,
            readme_text,
            patterns=patterns,
        )
        if not entry:
            continue
        if repo_id:
            entry["repo_id"] = repo_id
        entries[folder.name] = entry
    return entries


def _ensure_extended_families(families: dict[str, Any]) -> dict[str, Any]:
    merged = dict(families)
    merged.update(EXTRA_FAMILIES)
    return merged


def _ensure_extended_patterns(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = {json.dumps(rule, sort_keys=True) for rule in patterns}
    merged = list(patterns)
    for rule in EXTRA_PATTERNS:
        key = json.dumps(rule, sort_keys=True)
        if key not in existing:
            merged.append(rule)
    return merged


def merge_registry(
    existing: dict[str, Any],
    generated_models: dict[str, dict[str, Any]],
    *,
    fresh_models: bool = False,
) -> dict[str, Any]:
    """Merge generated model entries into an existing registry payload."""
    merged = {
        "version": existing.get("version", 1),
        "families": _ensure_extended_families(dict(existing.get("families") or {})),
        "models": {} if fresh_models else dict(existing.get("models") or {}),
        "patterns": _ensure_extended_patterns(list(existing.get("patterns") or [])),
    }

    for folder_name, entry in generated_models.items():
        current = dict(merged["models"].get(folder_name) or {})
        merged["models"][folder_name] = merge_model_entry(current, entry)
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
    fresh_models: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fetch HF models, merge with existing registry, and optionally write output."""
    target = output_path or REGISTRY_PATH
    existing = load_model_registry()
    patterns = _ensure_extended_patterns(list(existing.get("patterns") or []))

    generated = collect_registry_models(
        limit,
        patterns=patterns,
        fetch_upstream_readme=fetch_upstream_readme,
    )
    if include_local and models_dir is not None:
        local_entries = collect_local_model_entries(models_dir, patterns=patterns)
        for folder_name, entry in local_entries.items():
            if folder_name in generated:
                generated[folder_name] = merge_model_entry(generated[folder_name], entry)
            else:
                generated[folder_name] = entry

    merged = merge_registry(existing, generated, fresh_models=fresh_models)
    if not dry_run:
        write_registry(merged, target)
        load_model_registry.cache_clear()
    return merged
