"""Curated model defaults loaded from orchestrator/data/model_registry.json."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).resolve().parent / "data" / "model_registry.json"


@lru_cache(maxsize=1)
def load_model_registry() -> dict[str, Any]:
    """Load and cache the bundled model registry."""
    with open(REGISTRY_PATH, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("model_registry.json must contain a JSON object.")
    return payload


def _normalize_key(value: str) -> str:
    return value.strip().lower()


def _pattern_matches(folder_name: str, rule: dict[str, Any]) -> bool:
    name = _normalize_key(folder_name)
    match_token = _normalize_key(str(rule.get("match", "")))
    if not match_token or match_token not in name:
        return False
    required = rule.get("requires") or []
    if not required:
        return True
    return all(_normalize_key(token) in name for token in required)


def resolve_registry_family(folder_name: str) -> str | None:
    """Return a registry family id for a local model folder name."""
    registry = load_model_registry()
    models = registry.get("models") or {}
    normalized = _normalize_key(folder_name)

    for model_key, entry in models.items():
        if _normalize_key(model_key) == normalized:
            return str(entry.get("family") or "")

    best_family = ""
    best_score = 0
    for rule in registry.get("patterns") or []:
        if not _pattern_matches(folder_name, rule):
            continue
        score = len(str(rule.get("match", "")))
        if score > best_score:
            best_score = score
            best_family = str(rule.get("family") or "")
    return best_family or None


def resolve_registry_entry(folder_name: str) -> dict[str, Any] | None:
    """Return the registry model entry when explicitly listed."""
    registry = load_model_registry()
    models = registry.get("models") or {}
    normalized = _normalize_key(folder_name)
    for model_key, entry in models.items():
        if _normalize_key(model_key) == normalized:
            merged = dict(entry)
            merged["folder_name"] = model_key
            return merged
    return None


def get_registry_family(family_id: str) -> dict[str, Any] | None:
    registry = load_model_registry()
    family = (registry.get("families") or {}).get(family_id)
    if isinstance(family, dict):
        return family
    return None


def resolve_model_registry_profile(
    folder_name: str,
    launch_mode: str,
) -> dict[str, Any] | None:
    """Return merged registry metadata for a folder and launch mode."""
    family_id = resolve_registry_family(folder_name)
    if not family_id:
        return None

    family = get_registry_family(family_id)
    if not family:
        return None

    launch_modes = family.get("launch_modes") or []
    if launch_mode not in launch_modes:
        return None

    entry = resolve_registry_entry(folder_name) or {"folder_name": folder_name}
    return {
        "family_id": family_id,
        "folder_name": entry.get("folder_name", folder_name),
        "repo_id": entry.get("repo_id"),
        "upstream": entry.get("upstream"),
        "image_profile_ref": family.get("image_profile_ref"),
        "server_config": dict(family.get("server_config") or {}),
        "sources": list(family.get("sources") or []),
    }


def apply_registry_server_defaults(
    launch_mode: str,
    model_name: str,
    base_config: dict[str, Any],
) -> dict[str, Any]:
    """Merge registry defaults into a server_config dict without overriding explicit values."""
    profile = resolve_model_registry_profile(model_name, launch_mode)
    if not profile:
        return base_config

    merged = dict(base_config)
    for key, value in (profile.get("server_config") or {}).items():
        if key not in merged or merged[key] in (None, ""):
            merged[key] = value

    registry_meta = merged.setdefault("registry", {})
    registry_meta.update({
        "family_id": profile["family_id"],
        "sources": profile.get("sources") or [],
    })
    if profile.get("repo_id"):
        registry_meta["repo_id"] = profile["repo_id"]
    if profile.get("upstream"):
        registry_meta["upstream"] = profile["upstream"]
    if profile.get("image_profile_ref"):
        registry_meta["image_profile_ref"] = profile["image_profile_ref"]
    return merged


def registry_defaults_for_ui() -> dict[str, dict[str, dict[str, Any]]]:
    """Map launch_mode -> folder_name -> server_config defaults for the UI."""
    return build_registry_defaults_for_folders([])


def build_registry_defaults_for_folders(
    folder_names: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    launch_modes = ("TEXT", "MULTIMODAL", "EMBEDDING", "RERANKER", "IMAGE", "TTS", "STT")
    payload: dict[str, dict[str, dict[str, Any]]] = {mode: {} for mode in launch_modes}

    for folder_name in folder_names:
        for launch_mode in launch_modes:
            profile = resolve_model_registry_profile(folder_name, launch_mode)
            if profile:
                payload[launch_mode][folder_name] = profile.get("server_config") or {}

    return payload


def registry_defaults_for_ui_json() -> str:
    return json.dumps(registry_defaults_for_ui())


def build_registry_defaults_json(folder_names: list[str]) -> str:
    return json.dumps(build_registry_defaults_for_folders(folder_names))


def registry_metadata_for_ui() -> dict[str, dict[str, Any]]:
    """Per-folder registry metadata (upstream, family) for UI hints."""
    registry = load_model_registry()
    metadata: dict[str, dict[str, Any]] = {}
    for folder_name, entry in (registry.get("models") or {}).items():
        family_id = str(entry.get("family") or "")
        family = get_registry_family(family_id) or {}
        metadata[folder_name] = {
            "family_id": family_id,
            "repo_id": entry.get("repo_id"),
            "upstream": entry.get("upstream"),
            "launch_modes": family.get("launch_modes") or [],
            "sources": family.get("sources") or [],
        }
    return metadata


def build_registry_metadata_for_folders(folder_names: list[str]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for folder_name in folder_names:
        family_id = resolve_registry_family(folder_name)
        if not family_id:
            continue
        family = get_registry_family(family_id) or {}
        entry = resolve_registry_entry(folder_name) or {}
        metadata[folder_name] = {
            "family_id": family_id,
            "repo_id": entry.get("repo_id"),
            "upstream": entry.get("upstream"),
            "launch_modes": family.get("launch_modes") or [],
            "sources": family.get("sources") or [],
        }
    return metadata


def build_registry_metadata_json(folder_names: list[str]) -> str:
    return json.dumps(build_registry_metadata_for_folders(folder_names))
