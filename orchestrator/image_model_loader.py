"""Resolve and load mflux image-generation models from local checkpoints."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageModelSpec:
    """Runtime configuration inferred from a local model folder name."""

    family: str
    config_attr: str
    quantize: int | None
    default_steps: int
    default_guidance: float


def infer_quantize_from_name(folder_name: str) -> int | None:
    """Extract quantization bits from common Hugging Face folder naming."""
    match = re.search(r"(\d+)bit", folder_name, re.IGNORECASE)
    if match:
        bits = int(match.group(1))
        if bits in (3, 4, 5, 6, 8):
            return bits

    q_match = re.search(r"[-_]q(\d)\b", folder_name, re.IGNORECASE)
    if q_match:
        bits = int(q_match.group(1))
        if bits in (3, 4, 5, 6, 8):
            return bits

    return None


def resolve_image_model_spec(model_path: Path) -> ImageModelSpec:
    """Map a local folder name to an mflux model family and defaults."""
    name = model_path.name.lower()
    quantize = infer_quantize_from_name(model_path.name)

    if "z-image" in name or "z_image" in name:
        return ImageModelSpec("z_image", "z_image_turbo", quantize, 9, 0.0)
    if "flux2" in name or "klein" in name:
        if "9b" in name:
            return ImageModelSpec("flux2", "flux2_klein_9b", quantize, 4, 1.0)
        return ImageModelSpec("flux2", "flux2_klein_4b", quantize, 4, 1.0)
    if "qwen-image" in name or "qwen_image" in name:
        return ImageModelSpec("qwen_image", "qwen_image", quantize, 20, 3.5)
    if "krea" in name:
        return ImageModelSpec("flux1", "krea_dev", quantize, 20, 4.0)
    if "lite" in name and "flux" in name:
        # flux.1-lite is distilled from FLUX.1-dev (see model README on HF)
        return ImageModelSpec("flux1", "dev", quantize or 4, 50, 4.0)
    if "dev" in name and "flux" in name:
        return ImageModelSpec("flux1", "dev", quantize, 20, 4.0)
    if "schnell" in name:
        return ImageModelSpec("flux1", "schnell", quantize, 4, 0.0)
    if "flux" in name or "fibo" in name:
        return ImageModelSpec("flux1", "dev", quantize, 20, 4.0)

    raise ValueError(
        f"Unsupported image model folder '{model_path.name}'. "
        "Expected flux, schnell, z-image, qwen-image, or flux2/klein in the name."
    )


def load_image_model(model_path: Path, spec: ImageModelSpec) -> Any:
    """Load an mflux model instance from a local checkpoint directory."""
    from mflux.models.common.config import ModelConfig

    path_str = str(model_path.resolve())
    model_config = getattr(ModelConfig, spec.config_attr)()

    if spec.family == "flux1":
        from mflux.models.flux.variants.txt2img.flux import Flux1

        return Flux1(
            quantize=spec.quantize,
            model_path=path_str,
            model_config=model_config,
        )
    if spec.family == "z_image":
        from mflux.models.z_image import ZImageTurbo

        return ZImageTurbo(
            quantize=spec.quantize,
            model_path=path_str,
            model_config=model_config,
        )
    if spec.family == "flux2":
        from mflux.models.flux2.variants import Flux2Klein

        return Flux2Klein(
            quantize=spec.quantize,
            model_path=path_str,
            model_config=model_config,
        )
    if spec.family == "qwen_image":
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

        return QwenImage(
            quantize=spec.quantize,
            model_path=path_str,
            model_config=model_config,
        )

    raise ValueError(f"Unknown image model family: {spec.family}")


def generate_image_bytes(
    model: Any,
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
    }
    if guidance > 0:
        kwargs["guidance"] = guidance
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    generated = model.generate_image(**kwargs)
    buffer = io.BytesIO()
    generated.image.save(buffer, format="PNG")
    return buffer.getvalue()
