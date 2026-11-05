"""Tests for Mac-fit catalog heuristics (MLX-130 / MLX-131)."""

from orchestrator.mac_model_fit import (
    MacHardwareProfile,
    bandwidth_class_from_chip,
    classify_mac_fit,
    enrich_hf_model_mac_fit,
    estimate_catalog_text_footprint_bytes,
    extract_active_param_size_b,
    interactive_recommendation_score,
    is_moe_model_name,
    usable_memory_budget_bytes,
)


def _profile_48gb() -> MacHardwareProfile:
    return MacHardwareProfile(
        total_ram_bytes=48 * 1024**3,
        chip_label="Apple M4 Max",
        bandwidth_class="high",
        is_apple_silicon=True,
    )


def test_extract_active_param_size_from_moe_name() -> None:
    assert extract_active_param_size_b("Qwen3-30B-A3B-Instruct-4bit", 30.0) == 3.0
    assert is_moe_model_name("Qwen3-30B-A3B-Instruct-4bit") is True


def test_moe_footprint_uses_total_params_not_active() -> None:
    """All experts stay resident — MoE must not under-count weights."""
    dense = estimate_catalog_text_footprint_bytes(param_size_b=30.0, bits=4)
    moe = estimate_catalog_text_footprint_bytes(param_size_b=30.0, bits=4)
    assert moe == dense

    footprint = estimate_catalog_text_footprint_bytes(param_size_b=30.0, bits=4)
    gib = footprint / (1024**3)
    # ~17 GB weights + planned KV/cache — not the ~4 GB active-only bug.
    assert 15.0 <= gib <= 30.0


def test_moe_235b_a22b_is_too_large_on_48gb() -> None:
    footprint = estimate_catalog_text_footprint_bytes(param_size_b=235.0, bits=4)
    gib = footprint / (1024**3)
    assert gib >= 100.0

    assessment = classify_mac_fit(footprint, _profile_48gb())
    assert assessment.tier == "too_large"
    assert assessment.interactive_score == 0


def test_moe_30b_a3b_recommended_on_48gb() -> None:
    model = {
        "name": "Qwen3-30B-A3B-4bit",
        "param_size_b": 30.0,
        "quant_bits": 4,
        "supports_text": True,
        "downloads": 100,
    }
    enrich_hf_model_mac_fit(model, profile=_profile_48gb())
    assert model["mac_fit_tier"] in {"comfortable", "tight"}
    assert 15.0 <= model["mac_footprint_bytes"] / (1024**3) <= 30.0


def test_classify_mac_fit_tiers_on_48gb_profile() -> None:
    profile = _profile_48gb()
    budget = usable_memory_budget_bytes(profile)
    assert budget > 0

    comfortable = classify_mac_fit(int(budget * 0.40), profile)
    assert comfortable.tier == "comfortable"
    assert comfortable.label == "Recommended for your Mac"

    tight = classify_mac_fit(int(budget * 0.70), profile)
    assert tight.tier == "tight"

    too_large = classify_mac_fit(int(budget * 0.95), profile)
    assert too_large.tier == "too_large"


def test_tight_cap_respects_gpu_pool_ratio() -> None:
    """Tight must not exceed ~75% of unified RAM (mlx GPU pool)."""
    profile = _profile_48gb()
    gpu_pool = int(48 * 1024**3 * 0.75)
    just_under = classify_mac_fit(gpu_pool - 1024**2, profile)
    just_over = classify_mac_fit(gpu_pool + 1024**2, profile)
    assert just_under.tier in {"comfortable", "tight"}
    assert just_over.tier == "too_large"


def test_classify_mac_fit_unknown_without_ram() -> None:
    profile = MacHardwareProfile(
        total_ram_bytes=None,
        chip_label="Unknown",
        bandwidth_class="unknown",
        is_apple_silicon=False,
    )
    result = classify_mac_fit(8 * 1024**3, profile)
    assert result.tier == "unknown"
    assert result.label == ""


def test_interactive_score_boosts_moe_and_4bit_when_fit() -> None:
    profile = _profile_48gb()
    assessment = classify_mac_fit(8 * 1024**3, profile)
    score = interactive_recommendation_score(
        name="Qwen3-30B-A3B-Instruct-4bit",
        bits=4,
        assessment=assessment,
        profile=profile,
        supports_text=True,
        param_size_b=30.0,
    )
    plain = interactive_recommendation_score(
        name="Some-32B-Instruct-4bit",
        bits=4,
        assessment=assessment,
        profile=profile,
        supports_text=True,
        param_size_b=32.0,
    )
    assert score > plain


def test_bandwidth_class_by_generation_and_tier() -> None:
    assert bandwidth_class_from_chip("Apple M1 Max") == "high"
    assert bandwidth_class_from_chip("Apple M2 Max") == "high"
    assert bandwidth_class_from_chip("Apple M4") == "low"
    assert bandwidth_class_from_chip("Apple M4 Pro") == "medium"
    assert bandwidth_class_from_chip("Apple M4 Max") == "high"
    assert bandwidth_class_from_chip("Apple M2 Pro") == "medium"
    assert bandwidth_class_from_chip("Apple M3 Pro") == "medium"
    # Unknown generation → tier fallback (Max → high).
    assert bandwidth_class_from_chip("Apple M5 Max") == "high"
    assert bandwidth_class_from_chip("Apple M5 Ultra") == "high"
    assert bandwidth_class_from_chip("Apple M5 Pro") == "medium"
    assert bandwidth_class_from_chip("Apple M5") == "low"


def test_enrich_hf_model_mac_fit_sets_badge_fields() -> None:
    model = {
        "name": "Qwen3-30B-A3B-Instruct-4bit",
        "param_size_b": 30.0,
        "quant_bits": 4,
        "supports_text": True,
        "downloads": 100,
    }
    enrich_hf_model_mac_fit(model, profile=_profile_48gb())
    assert model["mac_fit_tier"] in {"comfortable", "tight", "too_large", "unknown"}
    assert model["is_moe"] is True
    if model["mac_fit_tier"] != "unknown":
        assert model["mac_fit_label"]
        assert model["interactive_recommendation_score"] >= 0


def test_param_size_from_safetensors_total() -> None:
    from orchestrator.ui_selectors import _param_size_from_hf_payload

    size = _param_size_from_hf_payload(
        {"safetensors": {"total": 30_000_000_000}},
        "ignored-name",
    )
    assert size == 30.0
