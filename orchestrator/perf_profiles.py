"""One-click performance profiles for server configuration."""

from __future__ import annotations

import re
from typing import Any

from orchestrator.model_registry import load_model_registry, resolve_registry_family

_E4B_RE = re.compile(r"e4b", re.IGNORECASE)
_E2B_RE = re.compile(r"e2b", re.IGNORECASE)


def suggest_mtp_assistant_folder(target_folder: str) -> str | None:
    """Suggest a local bf16 MTP assistant folder for a Gemma 4 target checkpoint."""
    name = target_folder.strip()
    if not name:
        return None
    lowered = name.lower()
    if "qat" not in lowered:
        return None
    if _E4B_RE.search(name):
        return "gemma-4-E4B-it-assistant-bf16"
    if _E2B_RE.search(name):
        return "gemma-4-E2B-it-assistant-bf16"
    return None


def build_mtp_assistant_suggestions(folder_names: list[str]) -> dict[str, str]:
    """Map installed model folders to suggested bf16 MTP assistant folders."""
    suggestions: dict[str, str] = {}
    for folder_name in folder_names:
        assistant = suggest_mtp_assistant_folder(folder_name)
        if assistant:
            suggestions[folder_name] = assistant
    return suggestions


def mtp_assistant_suggestions_for_ui_json(folder_names: list[str]) -> str:
    import json

    return json.dumps(build_mtp_assistant_suggestions(folder_names))


def _pattern_matches(folder_name: str, rule: dict[str, Any]) -> bool:
    name = folder_name.lower()
    match_token = str(rule.get("match", "")).lower()
    if not match_token or match_token not in name:
        return False
    required = rule.get("requires") or []
    return all(str(token).lower() in name for token in required)


def _resolve_draft_model(folder_name: str, profile: dict[str, Any]) -> str | None:
    explicit = str((profile.get("server_config") or {}).get("advanced", {}).get("draft_model") or "").strip()
    if explicit:
        return explicit
    if profile.get("resolve_mtp_assistant"):
        return suggest_mtp_assistant_folder(folder_name)
    return None


def list_perf_profiles() -> list[dict[str, Any]]:
    """Return curated perf profiles from model_registry.json."""
    registry = load_model_registry()
    profiles = registry.get("perf_profiles") or []
    return [item for item in profiles if isinstance(item, dict)]


def resolve_perf_profiles_for_model(folder_name: str, launch_mode: str) -> list[dict[str, Any]]:
    """Return applicable perf profiles for a model folder and launch mode."""
    family_id = resolve_registry_family(folder_name)
    resolved: list[dict[str, Any]] = []

    for profile in list_perf_profiles():
        if str(profile.get("launch_mode") or "") != launch_mode:
            continue
        profile_family = str(profile.get("family") or "").strip()
        if profile_family and profile_family != family_id:
            continue
        if not _pattern_matches(folder_name, profile):
            continue

        server_config = dict(profile.get("server_config") or {})
        advanced = dict(server_config.get("advanced") or {})
        draft_model = _resolve_draft_model(folder_name, profile)
        if draft_model:
            advanced["draft_model"] = draft_model
        if advanced:
            server_config["advanced"] = advanced

        resolved.append({
            "id": str(profile.get("id") or ""),
            "label": str(profile.get("label") or profile.get("id") or "Profile"),
            "server_config": server_config,
        })

    return resolved


def build_perf_profiles_for_folders(folder_names: list[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Map launch_mode -> folder_name -> list of perf profile options for the UI."""
    launch_modes = ("TEXT", "MULTIMODAL", "EMBEDDING", "RERANKER", "IMAGE", "TTS", "STT")
    payload: dict[str, dict[str, list[dict[str, Any]]]] = {mode: {} for mode in launch_modes}

    for folder_name in folder_names:
        for launch_mode in launch_modes:
            profiles = resolve_perf_profiles_for_model(folder_name, launch_mode)
            if profiles:
                payload[launch_mode][folder_name] = profiles

    return payload


def perf_profiles_for_ui_json(folder_names: list[str]) -> str:
    import json

    return json.dumps(build_perf_profiles_for_folders(folder_names))
