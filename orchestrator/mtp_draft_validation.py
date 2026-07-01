"""Validate Gemma 4 MTP drafter compatibility before server launch."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from django.conf import settings

_QUANTIZED_DRAFT_NAME_RE = re.compile(r"(qat|[-_]4bit|[-_]8bit)", re.IGNORECASE)

_MTP_ORDERED_ASSISTANT_ERROR = (
    "MTP with Gemma 4 E2B/E4B assistants requires an unquantized drafter "
    "(for example mlx-community/gemma-4-E4B-it-assistant-bf16). "
    "Quantized assistant checkpoints (qat / 4bit) are not supported by "
    "mlx-vlm MaskedEmbedder in mlx-vlm 0.6.x. The target model may stay "
    "quantized; only the draft_model must be bf16."
)


def resolve_draft_model_directory(draft_model: str) -> Path | None:
    """Resolve a local draft model directory from advanced.draft_model."""
    value = draft_model.strip()
    if not value or "/" in value and not value.startswith("/"):
        return None

    candidate = Path(value)
    if candidate.is_dir():
        return candidate.resolve()

    models_root = Path(settings.MODELS_DIR).resolve()
    folder = (models_root / value).resolve()
    if folder.is_relative_to(models_root) and folder.is_dir():
        return folder
    return None


def _read_model_config(model_dir: Path) -> dict[str, Any] | None:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return None
    try:
        with open(config_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _quantization_bits(config: dict[str, Any]) -> int | None:
    for key in ("quantization", "quantization_config"):
        quant = config.get(key)
        if not isinstance(quant, dict):
            continue
        bits = quant.get("bits")
        if isinstance(bits, (int, float)):
            return int(bits)
    return None


def is_incompatible_mtp_ordered_assistant(config: dict[str, Any]) -> bool:
    """Return True when mlx-vlm MTP cannot run this gemma4_assistant checkpoint."""
    if config.get("model_type") != "gemma4_assistant":
        return False
    if not config.get("use_ordered_embeddings"):
        return False
    bits = _quantization_bits(config)
    return bits is not None and bits < 16


def validate_mtp_draft_advanced(launch_mode: str, advanced: dict[str, Any]) -> None:
    """Reject MTP configs known to crash mlx-vlm at generation time."""
    if launch_mode != "MULTIMODAL":
        return

    draft_kind = str(advanced.get("draft_kind") or "").strip().lower()
    draft_model = str(advanced.get("draft_model") or "").strip()
    if draft_kind != "mtp" or not draft_model:
        return

    draft_dir = resolve_draft_model_directory(draft_model)
    if draft_dir is not None:
        config = _read_model_config(draft_dir)
        if config and is_incompatible_mtp_ordered_assistant(config):
            raise ValueError(_MTP_ORDERED_ASSISTANT_ERROR)
        return

    if _QUANTIZED_DRAFT_NAME_RE.search(draft_model):
        raise ValueError(_MTP_ORDERED_ASSISTANT_ERROR)
