"""Load mflux models and run image generation."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from orchestrator.image_model_profiles import ImageModelProfile, resolve_image_profile


def resolve_image_model_spec(model_path: Path) -> ImageModelProfile:
    """Backward-compatible alias for profile resolution."""
    return resolve_image_profile(model_path)


def load_image_model(model_path: Path, profile: ImageModelProfile) -> Any:
    """Load an mflux model instance from a local checkpoint directory."""
    from mflux.models.common.config import ModelConfig

    path_str = str(model_path.resolve())

    if profile.family == "flux1":
        from mflux.models.flux.variants.txt2img.flux import Flux1

        if not profile.flux_base_model:
            raise ValueError(f"Profile '{profile.profile_id}' is missing flux_base_model.")

        model_config = ModelConfig.from_name(
            model_path.name,
            base_model=profile.flux_base_model,
        )
        return Flux1(
            quantize=profile.quantize,
            model_path=path_str,
            model_config=model_config,
        )

    model_config = getattr(ModelConfig, profile.config_attr)()

    if profile.family == "z_image":
        from mflux.models.z_image import ZImageTurbo

        return ZImageTurbo(
            quantize=profile.quantize,
            model_path=path_str,
            model_config=model_config,
        )
    if profile.family == "flux2":
        from mflux.models.flux2.variants import Flux2Klein

        return Flux2Klein(
            quantize=profile.quantize,
            model_path=path_str,
            model_config=model_config,
        )
    if profile.family == "qwen_image":
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

        return QwenImage(
            quantize=profile.quantize,
            model_path=path_str,
            model_config=model_config,
        )
    if profile.family == "fibo":
        from mflux.models.fibo.variants.txt2img.fibo import FIBO

        return FIBO(
            quantize=profile.quantize,
            model_path=path_str,
            model_config=model_config,
        )

    raise ValueError(f"Unknown image model family: {profile.family}")


def generate_image_bytes(
    model: Any,
    profile: ImageModelProfile,
    *,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    width: int,
    height: int,
    guidance: float,
    negative_prompt: str | None = None,
) -> bytes:
    """Run inference and return PNG bytes."""
    kwargs: dict[str, Any] = {
        "seed": seed,
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "width": width,
        "height": height,
        "scheduler": profile.scheduler,
    }
    if profile.use_guidance and guidance > 0:
        kwargs["guidance"] = guidance
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    generated = model.generate_image(**kwargs)
    buffer = io.BytesIO()
    generated.image.save(buffer, format="PNG")
    return buffer.getvalue()
