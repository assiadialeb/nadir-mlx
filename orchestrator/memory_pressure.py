"""Host memory pressure helpers for LRU eviction (MLX-112 / MLX-127).

Footprint estimate = weights (disk × factor) + active KV × parallel sequences
+ a small prompt-cache planning budget. KV is sized layer-by-layer so sliding /
shared / linear-attention architectures are not treated as full dense caches.
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
# Cold load has an empty prompt cache; grow with use — do not reserve every slot.
_DEFAULT_PLANNING_PROMPT_CACHE_ENTRIES = 1
# Concurrent decode slots assumed for eviction planning (not the mlx default of 32).
_DEFAULT_PLANNING_PARALLEL_SEQUENCES = 2
_DEFAULT_KV_BITS = 16.0
_LAUNCH_MODES_WITH_KV = frozenset({"TEXT", "MULTIMODAL"})
_FULL_ATTENTION = "full_attention"
_SLIDING_ATTENTION = "sliding_attention"
_LINEAR_ATTENTION = "linear_attention"


@dataclass(frozen=True)
class LinearAttentionShape:
    """Fixed-size GatedDeltaNet / linear-attention state (Qwen3.5+)."""

    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_dim: int = 4


@dataclass(frozen=True)
class TransformerShape:
    """Attention geometry used for KV sizing."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_position_embeddings: int | None = None
    layer_types: tuple[str, ...] = ()
    sliding_window: int | None = None
    num_kv_shared_layers: int = 0
    attention_k_eq_v: bool = False
    linear: LinearAttentionShape | None = None


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
    parallel_sequences: int = 1


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
    return _positive_setting_or_env(
        "NADIR_MEMORY_PLANNING_CONTEXT_TOKENS",
        _DEFAULT_PLANNING_CONTEXT_TOKENS,
        minimum=512,
    )


def planning_prompt_prefix_tokens() -> int:
    """Assumed prefix length per prompt-cache entry when no byte cap is set."""
    return _positive_setting_or_env(
        "NADIR_MEMORY_PLANNING_PROMPT_PREFIX_TOKENS",
        _DEFAULT_PROMPT_PREFIX_TOKENS,
        minimum=256,
    )


def planning_prompt_cache_entries() -> int:
    """Prompt-cache slots reserved at load time (cold cache is empty)."""
    return _positive_setting_or_env(
        "NADIR_MEMORY_PLANNING_PROMPT_CACHE_ENTRIES",
        _DEFAULT_PLANNING_PROMPT_CACHE_ENTRIES,
        minimum=0,
        allow_zero=True,
    )


def planning_parallel_sequences_cap() -> int:
    """Upper bound on concurrent sequences assumed for eviction planning."""
    return _positive_setting_or_env(
        "NADIR_MEMORY_PLANNING_PARALLEL_SEQUENCES",
        _DEFAULT_PLANNING_PARALLEL_SEQUENCES,
        minimum=1,
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
    """Return a structured footprint used by LRU eviction and Mac-fit UI."""
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
            parallel_sequences=1,
        )

    shape = load_transformer_shape(model_name)
    context = _resolve_context_tokens(config, shape)
    kv_bits = _resolve_kv_bits(config)
    parallel = resolve_planning_parallel_sequences(config)
    per_sequence = estimate_kv_cache_bytes(shape, context_tokens=context, kv_bits=kv_bits)
    kv_bytes = per_sequence * parallel
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
        parallel_sequences=parallel,
    )


def resolve_planning_parallel_sequences(server_config: dict[str, Any] | None) -> int:
    """``min(decode_concurrency, gateway caps, planning cap)`` — at least 1."""
    caps = [planning_parallel_sequences_cap()]
    advanced = dict((server_config or {}).get("advanced") or {})
    for raw in (
        advanced.get("decode_concurrency"),
        (server_config or {}).get("max_concurrent_upstream"),
        getattr(settings, "NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM", None),
        getattr(settings, "NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL", None),
    ):
        value = _positive_int(raw)
        if value is not None:
            caps.append(value)
    return max(1, min(caps))


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
    """Bytes for one sequence's KV / linear state at ``context_tokens``."""
    if shape is None or context_tokens < 0:
        return 0
    if context_tokens == 0 and not shape.layer_types:
        return 0

    bytes_per_value = max(kv_bits, 1.0) / 8.0
    owning_layers = max(0, shape.num_layers - max(0, shape.num_kv_shared_layers))
    total = 0
    for layer_index in range(owning_layers):
        layer_type = _layer_type_at(shape, layer_index)
        if layer_type == _LINEAR_ATTENTION:
            total += _linear_state_bytes(shape.linear)
            continue
        tokens = _tokens_for_layer(shape, layer_type, context_tokens)
        if tokens <= 0:
            continue
        tensors = 1 if (layer_type == _FULL_ATTENTION and shape.attention_k_eq_v) else 2
        total += int(
            tensors
            * shape.num_kv_heads
            * shape.head_dim
            * tokens
            * bytes_per_value
        )
    return total


def estimate_prompt_cache_budget_bytes(
    shape: TransformerShape | None,
    *,
    server_config: dict[str, Any] | None,
    kv_bits: float,
    prefix_tokens: int,
    launch_mode: str,
) -> int:
    """Budget for mlx-lm prompt-cache entries reserved at load (TEXT only).

    Explicit ``prompt_cache_bytes`` wins. Otherwise count ``min(configured size,
    planning entries)`` × KV of an assumed shared agent prefix — cold loads do
    not reserve the full LRU capacity.
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
    configured = _positive_int(size) or 0
    entries = min(configured, planning_prompt_cache_entries())
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
    sliding = _first_int(config, ("sliding_window", "sliding_window_size"))
    shared = _first_int(config, ("num_kv_shared_layers",)) or 0
    layer_types = _parse_layer_types(config.get("layer_types"), layers)
    return TransformerShape(
        num_layers=layers,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max_pos,
        layer_types=layer_types,
        sliding_window=sliding,
        num_kv_shared_layers=min(shared, layers),
        attention_k_eq_v=bool(config.get("attention_k_eq_v")),
        linear=_linear_shape_from_config(config),
    )


def _linear_shape_from_config(config: dict[str, Any]) -> LinearAttentionShape | None:
    key_heads = _first_int(config, ("linear_num_key_heads",))
    value_heads = _first_int(config, ("linear_num_value_heads",))
    key_dim = _first_int(config, ("linear_key_head_dim",))
    value_dim = _first_int(config, ("linear_value_head_dim",))
    if not key_heads or not value_heads or not key_dim or not value_dim:
        return None
    conv = _first_int(config, ("linear_conv_kernel_dim",)) or 4
    return LinearAttentionShape(
        num_key_heads=key_heads,
        num_value_heads=value_heads,
        key_head_dim=key_dim,
        value_head_dim=value_dim,
        conv_kernel_dim=conv,
    )


def _parse_layer_types(raw: object, num_layers: int) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        return ()
    normalized: list[str] = []
    for entry in raw[:num_layers]:
        name = str(entry).strip().lower()
        if "linear" in name:
            normalized.append(_LINEAR_ATTENTION)
        elif "sliding" in name or "window" in name:
            normalized.append(_SLIDING_ATTENTION)
        else:
            normalized.append(_FULL_ATTENTION)
    return tuple(normalized)


def _layer_type_at(shape: TransformerShape, layer_index: int) -> str:
    if layer_index < len(shape.layer_types):
        return shape.layer_types[layer_index]
    return _FULL_ATTENTION


def _tokens_for_layer(
    shape: TransformerShape,
    layer_type: str,
    context_tokens: int,
) -> int:
    if layer_type != _SLIDING_ATTENTION or not shape.sliding_window:
        return context_tokens
    return min(context_tokens, int(shape.sliding_window))


def _linear_state_bytes(linear: LinearAttentionShape | None) -> int:
    """One-sequence GatedDeltaNet state: float32 recurrent + bf16 conv window."""
    if linear is None:
        return 0
    # state shape (1, Hv, Dv, Dk) in float32 — see mlx_lm.models.gated_delta
    recurrent = (
        linear.num_value_heads
        * linear.value_head_dim
        * linear.key_head_dim
        * 4
    )
    conv_channels = (
        2 * linear.num_key_heads * linear.key_head_dim
        + linear.num_value_heads * linear.value_head_dim
    )
    conv = max(0, linear.conv_kernel_dim - 1) * conv_channels * 2
    return recurrent + conv


def _positive_setting_or_env(
    name: str,
    default: int,
    *,
    minimum: int,
    allow_zero: bool = False,
) -> int:
    raw = os.environ.get(name)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = default
    else:
        value = int(getattr(settings, name, default))
    if allow_zero and value == 0:
        return 0
    return max(minimum, value)


def _positive_int(raw: object) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _first_int(config: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _positive_int(config.get(key))
        if value is not None:
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
