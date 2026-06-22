"""Gateway routing aliases stored in instance server_config.model_id."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.models import InferenceInstance

_GATEWAY_ALIAS_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]*$")
_MAX_GATEWAY_ALIAS_LENGTH = 255


def normalize_gateway_alias(raw_value: str) -> str:
    """Strip whitespace from a gateway alias."""
    return raw_value.strip()


def validate_gateway_alias_format(alias: str) -> None:
    """Raise ValueError when the alias is empty or uses invalid characters."""
    if not alias:
        raise ValueError("Gateway alias is required.")
    if len(alias) > _MAX_GATEWAY_ALIAS_LENGTH:
        raise ValueError(
            f"Gateway alias cannot exceed {_MAX_GATEWAY_ALIAS_LENGTH} characters."
        )
    if not _GATEWAY_ALIAS_PATTERN.match(alias):
        raise ValueError(
            "Gateway alias may only contain letters, digits, hyphens, "
            "underscores, dots, colons, or slashes."
        )


def instance_gateway_alias(instance: InferenceInstance, *, model_name: str | None = None) -> str:
    """Return the alias exposed to LiteLLM / Nadir gateway for an instance."""
    folder_name = model_name or instance.model_name
    config = instance.server_config or {}
    raw_alias = config.get("model_id") or folder_name
    return normalize_gateway_alias(str(raw_alias))


def validate_gateway_alias_unique(
    alias: str,
    *,
    exclude_instance_id: int | None = None,
) -> None:
    """Ensure no other instance already uses this alias (case-insensitive)."""
    from orchestrator.models import InferenceInstance

    normalized = normalize_gateway_alias(alias)
    validate_gateway_alias_format(normalized)
    alias_key = normalized.casefold()

    for instance in InferenceInstance.objects.all().only("id", "model_name", "server_config"):
        if exclude_instance_id is not None and instance.id == exclude_instance_id:
            continue
        if instance_gateway_alias(instance).casefold() == alias_key:
            raise ValueError(
                f"Gateway alias '{normalized}' is already used by another instance."
            )


def find_instance_by_gateway_alias(alias: str) -> InferenceInstance | None:
    """Resolve a gateway alias to a stored inference instance."""
    from orchestrator.models import InferenceInstance

    normalized = normalize_gateway_alias(alias)
    if not normalized:
        return None
    alias_key = normalized.casefold()

    for instance in InferenceInstance.objects.all().order_by("-created_at"):
        if instance_gateway_alias(instance).casefold() == alias_key:
            return instance
    return None
