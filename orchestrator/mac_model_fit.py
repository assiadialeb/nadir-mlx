"""Mac-fit tiers and interactive catalog scoring for HF models (MLX-130 / MLX-131)."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Literal

import platform

from orchestrator.memory_pressure import (
    TransformerShape,
    estimate_kv_cache_bytes,
    estimate_prompt_cache_budget_bytes,
    memory_reserve_bytes,
    planning_context_tokens,
    planning_parallel_sequences_cap,
    planning_prompt_prefix_tokens,
)
from orchestrator.prompt_cache import host_total_bytes

MacFitTier = Literal["comfortable", "tight", "too_large", "unknown"]
BandwidthClass = Literal["low", "medium", "high", "unknown"]
ChipTier = Literal["base", "pro", "max", "ultra"]

_MOE_ACTIVE_RE = re.compile(r"-A(\d+(?:\.\d+)?)[Bb]")
_MOE_HINT_RE = re.compile(r"\bmoe\b|experts?", re.I)
_DWQ_HINT_RE = re.compile(r"\bDWQ\b|dequant", re.I)
_CHIP_RE = re.compile(
    r"\bm(\d+)\s*(ultra|max|pro)?\b",
    re.IGNORECASE,
)

# Approximate unified-memory bandwidth (GB/s) by Apple Silicon generation + SKU.
_BANDWIDTH_GBPS: dict[tuple[int, ChipTier], int] = {
    (1, "base"): 68,
    (1, "pro"): 200,
    (1, "max"): 400,
    (1, "ultra"): 800,
    (2, "base"): 100,
    (2, "pro"): 200,
    (2, "max"): 400,
    (2, "ultra"): 800,
    (3, "base"): 100,
    (3, "pro"): 150,
    (3, "max"): 400,
    (3, "ultra"): 800,
    (4, "base"): 120,
    (4, "pro"): 273,
    (4, "max"): 546,
    (4, "ultra"): 819,
}

_TIER_FALLBACK_BANDWIDTH: dict[ChipTier, int] = {
    "base": 120,
    "pro": 200,
    "max": 400,
    "ultra": 800,
}

_COMFORTABLE_RATIO = 0.55
_TIGHT_RATIO = 0.85
# mlx wired GPU pool is typically ~75% of unified RAM on Apple Silicon.
_GPU_POOL_RATIO = 0.75
_MIN_FOOTPRINT_BYTES = 2 * 1024**3
_RUNTIME_WEIGHT_FACTOR = 1.15
_PROFILE_CACHE_TTL_SECONDS = 300.0


@dataclass(frozen=True)
class MacHardwareProfile:
    """Local Apple Silicon / host memory profile for catalog hints."""

    total_ram_bytes: int | None
    chip_label: str
    bandwidth_class: BandwidthClass
    is_apple_silicon: bool


@dataclass(frozen=True)
class MacFitAssessment:
    """Result of comparing a catalog footprint to the local Mac."""

    tier: MacFitTier
    label: str
    badge_classes: str
    footprint_bytes: int
    footprint_label: str
    usable_budget_bytes: int
    interactive_score: int


_cached_profile: MacHardwareProfile | None = None
_cached_profile_at: float = 0.0


def clear_mac_hardware_profile_cache() -> None:
    """Tests / process refresh."""
    global _cached_profile, _cached_profile_at
    _cached_profile = None
    _cached_profile_at = 0.0


def load_mac_hardware_profile(*, force_refresh: bool = False) -> MacHardwareProfile:
    """Detect unified RAM and a coarse memory-bandwidth class from the host."""
    global _cached_profile, _cached_profile_at
    now = time.monotonic()
    if (
        not force_refresh
        and _cached_profile is not None
        and (now - _cached_profile_at) < _PROFILE_CACHE_TTL_SECONDS
    ):
        return _cached_profile

    profile = _detect_mac_hardware_profile()
    _cached_profile = profile
    _cached_profile_at = now
    return profile


def is_moe_model_name(name: str) -> bool:
    """Return True when the repo name suggests a mixture-of-experts checkpoint."""
    if _MOE_ACTIVE_RE.search(name):
        return True
    return bool(_MOE_HINT_RE.search(name.lower()))


def extract_active_param_size_b(name: str, total_param_size_b: float | None) -> float | None:
    """Active experts for MoE names like ``30B-A3B`` (speed scoring only)."""
    match = _MOE_ACTIVE_RE.search(name)
    if match:
        return float(match.group(1))
    return total_param_size_b


def estimate_catalog_text_footprint_bytes(
    *,
    param_size_b: float | None,
    bits: int,
) -> int:
    """Weights + planned KV + prompt cache (MLX-127 defaults, generic shape).

    MoE checkpoints keep **all** experts resident — weights use total parameters,
    never the active-expert count from ``-A3B``-style names.
    """
    if param_size_b is None or param_size_b <= 0:
        return 0

    weights = int(
        param_size_b * 1_000_000_000 * (bits / 8.0) * _RUNTIME_WEIGHT_FACTOR
    )
    shape = _generic_transformer_shape(param_size_b)
    context = planning_context_tokens()
    kv_bits = 16.0
    parallel = planning_parallel_sequences_cap()
    per_sequence = estimate_kv_cache_bytes(
        shape, context_tokens=context, kv_bits=kv_bits
    )
    kv_bytes = per_sequence * parallel
    prefix_tokens = min(context, planning_prompt_prefix_tokens()) if context else 0
    prompt_cache = estimate_prompt_cache_budget_bytes(
        shape,
        server_config=None,
        kv_bits=kv_bits,
        prefix_tokens=prefix_tokens,
        launch_mode="TEXT",
    )
    return max(_MIN_FOOTPRINT_BYTES, weights + kv_bytes + prompt_cache)


def usable_memory_budget_bytes(profile: MacHardwareProfile) -> int:
    """Bytes treated as available for one TEXT instance after OS reserve."""
    if not profile.total_ram_bytes:
        return 0
    return max(0, profile.total_ram_bytes - memory_reserve_bytes())


def classify_mac_fit(
    footprint_bytes: int,
    profile: MacHardwareProfile,
) -> MacFitAssessment:
    """Map footprint to comfort tiers; never blocks download."""
    budget = usable_memory_budget_bytes(profile)
    if footprint_bytes <= 0 or budget <= 0:
        return MacFitAssessment(
            tier="unknown",
            label="",
            badge_classes="",
            footprint_bytes=footprint_bytes,
            footprint_label=_format_gib(footprint_bytes),
            usable_budget_bytes=budget,
            interactive_score=0,
        )

    comfortable_cap = int(budget * _COMFORTABLE_RATIO)
    tight_cap = _tight_cap_bytes(profile, budget)
    if footprint_bytes <= comfortable_cap:
        tier: MacFitTier = "comfortable"
        label = "Recommended for your Mac"
        badge_classes = (
            "bg-emerald-50 text-emerald-800 border border-emerald-200"
        )
        score = 100
    elif footprint_bytes <= tight_cap:
        tier = "tight"
        label = "May fit — tight"
        badge_classes = "bg-amber-50 text-amber-800 border border-amber-200"
        score = 40
    else:
        tier = "too_large"
        label = "Likely too large"
        badge_classes = "bg-rose-50 text-rose-800 border border-rose-200"
        score = 0

    return MacFitAssessment(
        tier=tier,
        label=label,
        badge_classes=badge_classes,
        footprint_bytes=footprint_bytes,
        footprint_label=_format_gib(footprint_bytes),
        usable_budget_bytes=budget,
        interactive_score=score,
    )


def interactive_recommendation_score(
    *,
    name: str,
    bits: int,
    assessment: MacFitAssessment,
    profile: MacHardwareProfile,
    supports_text: bool,
    param_size_b: float | None = None,
) -> int:
    """MLX-130 scoring: fit tier + MoE / 4-bit / bandwidth heuristics.

    Active MoE params improve the **speed** score only — they never shrink
    the memory footprint used for fit tiers.
    """
    if not supports_text:
        return assessment.interactive_score

    score = assessment.interactive_score
    lowered = name.lower()
    if is_moe_model_name(name):
        score += 25
    if bits <= 4 or _DWQ_HINT_RE.search(name):
        score += 15
    if "instruct" in lowered or "chat" in lowered or "-it" in lowered:
        score += 5

    total_b = param_size_b
    if total_b is None:
        param_match = re.search(r"(\d+(?:\.\d+)?)[Bb]", name)
        total_b = float(param_match.group(1)) if param_match else 0.0
    active_b = extract_active_param_size_b(name, total_b) or total_b or 0.0
    if profile.bandwidth_class in {"low", "medium"} and active_b >= 24 and not is_moe_model_name(name):
        score -= 20
    if profile.bandwidth_class == "high" and is_moe_model_name(name) and bits <= 4:
        score += 10
    return max(0, score)


def enrich_hf_model_mac_fit(
    model: dict[str, Any],
    *,
    profile: MacHardwareProfile | None = None,
) -> dict[str, Any]:
    """Add Mac-fit and recommendation fields to a parsed HF model dict."""
    profile = profile or load_mac_hardware_profile()
    name = str(model.get("name") or "")
    param_size_b = model.get("param_size_b")
    bits = int(model.get("quant_bits") or 16)
    total_b = float(param_size_b) if param_size_b is not None else None
    footprint = estimate_catalog_text_footprint_bytes(
        param_size_b=total_b,
        bits=bits,
    )
    assessment = classify_mac_fit(footprint, profile)
    score = interactive_recommendation_score(
        name=name,
        bits=bits,
        assessment=assessment,
        profile=profile,
        supports_text=bool(model.get("supports_text")),
        param_size_b=total_b,
    )
    model["is_moe"] = is_moe_model_name(name)
    model["mac_footprint_bytes"] = footprint
    model["mac_fit_tier"] = assessment.tier
    model["mac_fit_label"] = assessment.label
    model["mac_fit_badge_classes"] = assessment.badge_classes
    model["mac_fit_footprint_label"] = assessment.footprint_label
    model["interactive_recommendation_score"] = score
    if footprint > 0:
        model["ram_est"] = f"{assessment.footprint_label} (plan.)"
        model["estimated_size_bytes"] = footprint
    return model


def hub_mac_profile_summary(profile: MacHardwareProfile | None = None) -> str:
    """One-line copy for the HF catalog header."""
    profile = profile or load_mac_hardware_profile()
    if not profile.total_ram_bytes:
        return "Mac profile unknown — RAM estimates are indicative only."
    gib = profile.total_ram_bytes / (1024**3)
    chip = profile.chip_label
    return (
        f"{chip} · {gib:.0f} GB unified — badges use planned TEXT footprint "
        f"(all experts for MoE; weights + KV + cache)."
    )


def bandwidth_class_from_chip(chip_label: str) -> BandwidthClass:
    """Public helper for tests — map a CPU brand string to bandwidth class."""
    return _bandwidth_class_from_chip(chip_label)


def _detect_mac_hardware_profile() -> MacHardwareProfile:
    system = platform.system()
    chip_label = "Unknown host"
    is_apple = system == "Darwin"
    if is_apple:
        chip_label = _darwin_cpu_brand() or "Apple Silicon"
    elif system == "Linux":
        chip_label = "Linux (non-Mac)"

    total = host_total_bytes()
    bandwidth = _bandwidth_class_from_chip(chip_label) if is_apple else "unknown"
    return MacHardwareProfile(
        total_ram_bytes=total,
        chip_label=chip_label,
        bandwidth_class=bandwidth,
        is_apple_silicon=is_apple,
    )


def _tight_cap_bytes(profile: MacHardwareProfile, budget: int) -> int:
    """Cap tight tier at min(85% of budget, ~75% of unified RAM)."""
    ratio_cap = int(budget * _TIGHT_RATIO)
    if not profile.total_ram_bytes:
        return ratio_cap
    gpu_pool_cap = int(profile.total_ram_bytes * _GPU_POOL_RATIO)
    return min(ratio_cap, gpu_pool_cap)


def _generic_transformer_shape(param_b: float) -> TransformerShape:
    if param_b <= 4:
        layers, ctx = 32, 8192
    elif param_b <= 14:
        layers, ctx = 40, 8192
    elif param_b <= 30:
        layers, ctx = 48, 32768
    else:
        layers, ctx = 64, 32768
    return TransformerShape(
        num_layers=layers,
        num_kv_heads=8,
        head_dim=128,
        max_position_embeddings=ctx,
    )


def _format_gib(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "Unknown"
    gib = size_bytes / (1024**3)
    return f"{gib:.1f} GB"


def _darwin_cpu_brand() -> str | None:
    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    return raw or None


def _bandwidth_class_from_chip(chip_label: str) -> BandwidthClass:
    parsed = _parse_apple_silicon(chip_label)
    if parsed is None:
        if "apple" in chip_label.lower():
            return "low"
        return "unknown"
    generation, tier = parsed
    gbps = _BANDWIDTH_GBPS.get((generation, tier))
    if gbps is None:
        gbps = _TIER_FALLBACK_BANDWIDTH[tier]
    return _bandwidth_class_from_gbps(gbps)


def _parse_apple_silicon(chip_label: str) -> tuple[int, ChipTier] | None:
    match = _CHIP_RE.search(chip_label)
    if not match:
        return None
    generation = int(match.group(1))
    raw_tier = (match.group(2) or "base").lower()
    tier: ChipTier
    if raw_tier == "ultra":
        tier = "ultra"
    elif raw_tier == "max":
        tier = "max"
    elif raw_tier == "pro":
        tier = "pro"
    else:
        tier = "base"
    return generation, tier


def _bandwidth_class_from_gbps(gbps: int) -> BandwidthClass:
    if gbps >= 400:
        return "high"
    if gbps >= 150:
        return "medium"
    return "low"
