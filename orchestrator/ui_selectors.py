"""Read-only selectors for orchestrator UI pages."""

from __future__ import annotations

import json
from typing import Any

from orchestrator.model_utils import get_model_capabilities
from orchestrator.server_types import SERVER_TYPES


def list_installed_models() -> list[dict[str, Any]]:
    """Return downloaded models enriched with capability flags."""
    from orchestrator.server_manager import get_downloaded_models

    return [
        {"name": model_name, **get_model_capabilities(model_name)}
        for model_name in get_downloaded_models()
    ]


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
