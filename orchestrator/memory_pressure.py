"""Host memory pressure helpers for LRU eviction (MLX-112 / MLX-127).

Footprint estimate = weights (disk × factor) + active KV + prompt-cache budget.
Planning uses a capped context window so ``max_position_embeddings`` (often 128k+)
does not pretend the whole window is always resident.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Any

from django.conf import settings

from orchestrator.model_utils import get_model_folder_size_bytes, get_model_path

logger = logging.getLogger(__name__)

# Weights on disk underestimate resident MLX footprint; apply a conservative factor.
_DISK_TO_RUNTIME_FACTOR = 1.15
_DEFAULT_FOOTPRINT_BYTES = 2 * 1024**3
_DEFAULT_PLANNING_CONTEXT_TOKENS = 8192
_DEFAULT_PROMPT_PREFIX_TOKENS = 4096
_DEFAULT_KV_BITS = 16.0
_LAUNCH_MODES_WITH_KV = frozenset({"TEXT", "MULTIMODAL"})


@dataclass(frozen=True)
class TransformerShape:
    """Attention geometry used for KV sizing."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_position_embeddings: int | None = None


@dataclass(frozen=True)
class MemoryFootprintEstimate:
    """Breakdown of a planned resident footprint for one instance."""

    weights_bytes: int
    kv_bytes: int
    prompt_cache_bytes: int
    total_bytes: int
    context_tokens: int
    prompt_prefix_tokens: int
    kv_bits: float


def lru_eviction_enabled() -> bool:
    raw = os.environ.get("NADIR_LRU_EVICTION_ENABLED")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(getattr(settings, "NADIR_LRU_EVICTION_ENABLED", True))


def memory_reserve_bytes() -> int:
    raw = os.environ.get("NADIR_MEMORY_RESERVE_GB")
    if raw:
        try:
            return max(0, int(float(raw) * 1024**3))
        except ValueError:
            pass
    gb = float(getattr(settings, "NADIR_MEMORY_RESERVE_GB", 4.0))
    return max(0, int(gb * 1024**3))


def planning_context_tokens() -> int:
    """Default context tokens assumed resident when sizing KV for eviction."""
    raw = os.environ.get("NADIR_MEMORY_PLANNING_CONTEXT_TOKENS")
    if raw:
        try:
            return max(512, int(raw))
        except ValueError:
            pass
    return int(
        getattr(
            settings,
            "NADIR_MEMORY_PLANNING_CONTEXT_TOKENS",
            _DEFAULT_PLANNING_CONTEXT_TOKENS,
        )
    )


def planning_prompt_prefix_tokens() -> int:
    """Assumed prefix length per prompt-cache entry when no byte cap is set."""
    raw = os.environ.get("NADIR_MEMORY_PLANNING_PROMPT_PREFIX_TOKENS")
    if raw:
        try:
            return max(256, int(raw))
        except ValueError:
            pass
    return int(
        getattr(
            settings,
            "NADIR_MEMORY_PLANNING_PROMPT_PREFIX_TOKENS",
            _DEFAULT_PROMPT_PREFIX_TOKENS,
        )
    )


def estimate_instance_footprint_bytes(
    model_name: str,
    server_config: dict[str, Any] | None = None,
    *,
    launch_mode: str | None = None,
) -> int:
    """Estimate resident bytes for a model (weights + KV + prompt cache)."""
    return estimate_instance_footprint(
        model_name, server_config, launch_mode=launch_mode
    ).total_bytes


def estimate_instance_footprint(
    model_name: str,
    server_config: dict[str, Any] | None = None,
    *,
    launch_mode: str | None = None,
) -> MemoryFootprintEstimate:
    """Return a structured footprint used by LRU eviction and (later) Mac-fit UI."""
    config = dict(server_config or {})
    resolved_mode = str(
        launch_mode or config.get("launch_mode") or "TEXT"
    ).upper()

    weights = _estimate_weights_bytes(model_name)
    if resolved_mode not in _LAUNCH_MODES_WITH_KV:
        total = max(_DEFAULT_FOOTPRINT_BYTES, weights)
        return MemoryFootprintEstimate(
            weights_bytes=weights,
            kv_bytes=0,
            prompt_cache_bytes=0,
            total_bytes=total,
            context_tokens=0,
            prompt_prefix_tokens=0,
            kv_bits=_DEFAULT_KV_BITS,
        )

    shape = load_transformer_shape(model_name)
    context = _resolve_context_tokens(config, shape)
    kv_bits = _resolve_kv_bits(config)
    kv_bytes = estimate_kv_cache_bytes(shape, context_tokens=context, kv_bits=kv_bits)
    prefix_tokens = min(context, planning_prompt_prefix_tokens()) if context else 0
    prompt_cache = estimate_prompt_cache_budget_bytes(
        shape,
        server_config=config,
        kv_bits=kv_bits,
        prefix_tokens=prefix_tokens,
        launch_mode=resolved_mode,
    )
    total = max(_DEFAULT_FOOTPRINT_BYTES, weights + kv_bytes + prompt_cache)
    return MemoryFootprintEstimate(
        weights_bytes=weights,
        kv_bytes=kv_bytes,
        prompt_cache_bytes=prompt_cache,
        total_bytes=total,
        context_tokens=context,
        prompt_prefix_tokens=prefix_tokens,
        kv_bits=kv_bits,
    )


def load_transformer_shape(model_name: str) -> TransformerShape | None:
    """Read attention geometry from ``config.json`` (and nested ``text_config``)."""
    try:
        from orchestrator.model_utils import _read_model_config as read_config

        config = read_config(get_model_path(model_name))
    except Exception:
        logger.debug("Unable to read model config for %s", model_name, exc_info=True)
        return None
    return transformer_shape_from_config(config)


def transformer_shape_from_config(config: dict[str, Any] | None) -> TransformerShape | None:
    """Parse HF/MLX config dicts into a KV shape (supports nested text configs)."""
    if not config:
        return None
    candidates: list[dict[str, Any]] = [config]
    for key in ("text_config", "llm_config", "language_config", "model_config"):
        nested = config.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)

    for candidate in candidates:
        shape = _shape_from_flat_config(candidate)
        if shape is not None:
            return shape
    return None


def estimate_kv_cache_bytes(
    shape: TransformerShape | None,
    *,
    context_tokens: int,
    kv_bits: float = _DEFAULT_KV_BITS,
) -> int:
    """Bytes for one full KV cache at ``context_tokens`` (K + V, all layers)."""
    if shape is None or context_tokens <= 0:
        return 0
    bytes_per_value = max(kv_bits, 1.0) / 8.0
    # 2 = keys + values
    return int(
        2
        * shape.num_layers
        * shape.num_kv_heads
        * shape.head_dim
        * context_tokens
        * bytes_per_value
    )


def estimate_prompt_cache_budget_bytes(
    shape: TransformerShape | None,
    *,
    server_config: dict[str, Any] | None,
    kv_bits: float,
    prefix_tokens: int,
    launch_mode: str,
) -> int:
    """Budget for mlx-lm prompt-cache entries (TEXT only).

    Explicit ``prompt_cache_bytes`` wins. Otherwise multiply entry count by the
    KV size of an assumed shared agent prefix.
    """
    if launch_mode != "TEXT" or shape is None:
        return 0
    advanced = dict((server_config or {}).get("advanced") or {})
    if "prompt_cache_bytes" in advanced and advanced["prompt_cache_bytes"] is not None:
        try:
            return max(0, int(advanced["prompt_cache_bytes"]))
        except (TypeError, ValueError):
            return 0

    from orchestrator.prompt_cache import default_prompt_cache_settings

    size = advanced.get("prompt_cache_size")
    if size is None:
        size = default_prompt_cache_settings().get("prompt_cache_size", 10)
    try:
        entries = max(0, int(size))
    except (TypeError, ValueError):
        entries = 10
    if entries <= 0 or prefix_tokens <= 0:
        return 0
    return entries * estimate_kv_cache_bytes(
        shape, context_tokens=prefix_tokens, kv_bits=kv_bits
    )


def host_available_bytes() -> int | None:
    """Return approximate available unified/system memory, or None if unknown."""
    if platform.system() == "Darwin":
        return _darwin_available_bytes()
    return _linux_available_bytes()


def _estimate_weights_bytes(model_name: str) -> int:
    try:
        disk = int(get_model_folder_size_bytes(model_name) or 0)
    except Exception:
        disk = 0
    if disk <= 0:
        return _DEFAULT_FOOTPRINT_BYTES
    return max(_DEFAULT_FOOTPRINT_BYTES, int(disk * _DISK_TO_RUNTIME_FACTOR))


def _resolve_context_tokens(
    server_config: dict[str, Any],
    shape: TransformerShape | None,
) -> int:
    planned = planning_context_tokens()
    raw_max_kv = server_config.get("max_kv_size")
    if raw_max_kv is not None:
        try:
            planned = min(planned, max(512, int(raw_max_kv)))
        except (TypeError, ValueError):
            pass
    if shape and shape.max_position_embeddings:
        planned = min(planned, int(shape.max_position_embeddings))
    return max(0, planned)


def _resolve_kv_bits(server_config: dict[str, Any]) -> float:
    advanced = dict(server_config.get("advanced") or {})
    raw = advanced.get("kv_bits")
    if raw is None:
        return _DEFAULT_KV_BITS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_KV_BITS
    return value if value > 0 else _DEFAULT_KV_BITS


def _shape_from_flat_config(config: dict[str, Any]) -> TransformerShape | None:
    layers = _first_int(config, ("num_hidden_layers", "n_layer", "n_layers", "num_layers"))
    kv_heads = _first_int(
        config,
        ("num_key_value_heads", "num_kv_heads", "n_kv_heads"),
    )
    heads = _first_int(config, ("num_attention_heads", "n_head", "n_heads"))
    if kv_heads is None:
        kv_heads = heads
    head_dim = _first_int(config, ("head_dim", "attention_head_dim"))
    if head_dim is None and heads:
        hidden = _first_int(config, ("hidden_size", "n_embd", "d_model"))
        if hidden:
            head_dim = hidden // heads
    if not layers or not kv_heads or not head_dim:
        return None
    max_pos = _first_int(
        config,
        ("max_position_embeddings", "max_sequence_length", "max_seq_len"),
    )
    return TransformerShape(
        num_layers=layers,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max_pos,
    )


def _first_int(config: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        raw = config.get(key)
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _darwin_available_bytes() -> int | None:
    try:
        page_size = int(
            subprocess.check_output(["pagesize"], text=True, encoding="utf-8").strip()
        )
        vm_stat = subprocess.check_output(
            ["vm_stat"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
    except (OSError, ValueError, subprocess.CalledProcessError):
        logger.debug("Unable to read Darwin vm_stat for memory pressure", exc_info=True)
        return None

    free_pages = 0
    for line in vm_stat.splitlines():
        key, _, value = line.partition(":")
        if key.strip() not in {"Pages free", "Pages inactive", "Pages speculative"}:
            continue
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            free_pages += int(digits)
    if free_pages <= 0:
        return None
    available = free_pages * page_size
    reserve = memory_reserve_bytes()
    return max(0, available - reserve)


def _linux_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            meminfo = handle.read()
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in meminfo.splitlines():
        key, _, rest = line.partition(":")
        digits = "".join(ch for ch in rest if ch.isdigit())
        if digits:
            values[key] = int(digits) * 1024
    available = values.get("MemAvailable") or values.get("MemFree")
    if available is None:
        return None
    return max(0, available - memory_reserve_bytes())
