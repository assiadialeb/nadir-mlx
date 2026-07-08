"""Gateway routing aliases stored in instance server_config."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orchestrator.models import InferenceInstance

_GATEWAY_ALIAS_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]*$")
_MAX_GATEWAY_ALIAS_LENGTH = 255
_MAX_EXTRA_GATEWAY_ALIASES = 8


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


def _dedupe_aliases(aliases: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for alias in aliases:
        key = alias.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(alias)
    return unique


def normalize_gateway_aliases_list(
    raw_aliases: Any,
    *,
    primary_alias: str,
) -> list[str]:
    """Parse and validate extra gateway aliases (excludes the primary alias)."""
    if raw_aliases is None or raw_aliases == "":
        return []
    if isinstance(raw_aliases, str):
        candidates = [
            normalize_gateway_alias(part)
            for part in raw_aliases.split(",")
            if part.strip()
        ]
    elif isinstance(raw_aliases, list):
        candidates = [
            normalize_gateway_alias(str(part))
            for part in raw_aliases
            if str(part).strip()
        ]
    else:
        raise ValueError("gateway_aliases must be a list or comma-separated string.")

    if len(candidates) > _MAX_EXTRA_GATEWAY_ALIASES:
        raise ValueError(
            f"At most {_MAX_EXTRA_GATEWAY_ALIASES} extra gateway aliases are allowed."
        )

    primary_key = primary_alias.casefold()
    extras: list[str] = []
    for alias in candidates:
        validate_gateway_alias_format(alias)
        if alias.casefold() == primary_key:
            continue
        extras.append(alias)
    return _dedupe_aliases(extras)


def instance_primary_gateway_alias(
    instance: InferenceInstance,
    *,
    model_name: str | None = None,
) -> str:
    """Return the primary alias exposed to the Nadir gateway for an instance."""
    folder_name = model_name or instance.model_name
    config = instance.server_config or {}
    raw_alias = config.get("model_id") or folder_name
    return normalize_gateway_alias(str(raw_alias))


def instance_gateway_alias(instance: InferenceInstance, *, model_name: str | None = None) -> str:
    """Return the primary gateway alias (backward-compatible helper)."""
    return instance_primary_gateway_alias(instance, model_name=model_name)


def instance_gateway_aliases(
    instance: InferenceInstance,
    *,
    model_name: str | None = None,
) -> list[str]:
    """Return primary plus extra aliases for gateway routing and /v1/models."""
    primary = instance_primary_gateway_alias(instance, model_name=model_name)
    config = instance.server_config or {}
    extras = normalize_gateway_aliases_list(
        config.get("gateway_aliases"),
        primary_alias=primary,
    )
    return _dedupe_aliases([primary, *extras])


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
        for existing_alias in instance_gateway_aliases(instance):
            if existing_alias.casefold() == alias_key:
                raise ValueError(
                    f"Gateway alias '{normalized}' is already used by another instance."
                )


def validate_instance_gateway_aliases(
    server_config: dict[str, Any],
    model_name: str,
    *,
    exclude_instance_id: int | None = None,
) -> None:
    """Validate primary and extra aliases for uniqueness."""
    primary = normalize_gateway_alias(str(server_config.get("model_id") or model_name))
    validate_gateway_alias_format(primary)
    extras = normalize_gateway_aliases_list(
        server_config.get("gateway_aliases"),
        primary_alias=primary,
    )
    for alias in _dedupe_aliases([primary, *extras]):
        validate_gateway_alias_unique(alias, exclude_instance_id=exclude_instance_id)


def find_instance_by_gateway_alias(alias: str) -> InferenceInstance | None:
    """Resolve a gateway alias to a stored inference instance."""
    from orchestrator.models import InferenceInstance

    normalized = normalize_gateway_alias(alias)
    if not normalized:
        return None
    alias_key = normalized.casefold()

    for instance in InferenceInstance.objects.all().order_by("-created_at"):
        for existing_alias in instance_gateway_aliases(instance):
            if existing_alias.casefold() == alias_key:
                return instance
    return None
