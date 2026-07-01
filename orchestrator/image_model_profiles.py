"""Per-model inference profiles for mflux image generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from mflux.cli.defaults import defaults as mflux_defaults

MFLUX_GENERATE_CMD = re.compile(
    r"mflux-generate(?:-[\w-]+)?\s+([^`\n]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ImageModelProfile:
    """Recommended inference settings for a local image checkpoint."""

    profile_id: str
    family: str
    config_attr: str
    flux_base_model: str | None
    quantize: int | None
    default_steps: int
    fast_steps: int
    quality_steps: int
    default_guidance: float
    default_width: int
    default_height: int
    scheduler: str
    use_guidance: bool
    source: str = "registry"

    @property
    def default_size(self) -> str:
        return f"{self.default_width}x{self.default_height}"

    def resolve_steps(
        self,
        quality: str = "balanced",
        override: int | None = None,
    ) -> int:
        if override is not None:
            return override
        if quality == "fast":
            return self.fast_steps
        if quality == "quality":
            return self.quality_steps
        return self.default_steps

    def as_api_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "family": self.family,
            "config_attr": self.config_attr,
            "flux_base_model": self.flux_base_model,
            "quantize": self.quantize,
            "num_inference_steps": self.default_steps,
            "quality_presets": {
                "fast": self.fast_steps,
                "balanced": self.default_steps,
                "quality": self.quality_steps,
            },
            "guidance": self.default_guidance if self.use_guidance else 0.0,
            "size": self.default_size,
            "scheduler": self.scheduler,
            "source": self.source,
        }


@dataclass(frozen=True)
class _ProfileTemplate:
    profile_id: str
    family: str
    config_attr: str
    flux_base_model: str | None
    default_steps: int
    default_guidance: float
    use_guidance: bool
    fast_steps: int | None = None
    quality_steps: int | None = None
    scheduler: str = "linear"
    default_width: int = 1024
    default_height: int = 1024
    name_patterns: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()


PROFILE_REGISTRY: tuple[_ProfileTemplate, ...] = (
    _ProfileTemplate(
        profile_id="flux_lite",
        family="flux1",
        config_attr="dev",
        flux_base_model="dev",
        default_steps=20,
        fast_steps=12,
        quality_steps=50,
        default_guidance=4.0,
        use_guidance=True,
        name_patterns=("lite", "flux"),
    ),
    _ProfileTemplate(
        profile_id="flux_schnell",
        family="flux1",
        config_attr="schnell",
        flux_base_model="schnell",
        default_steps=mflux_defaults.MODEL_INFERENCE_STEPS["schnell"],
        default_guidance=0.0,
        use_guidance=False,
        name_patterns=("schnell",),
    ),
    _ProfileTemplate(
        profile_id="flux_krea",
        family="flux1",
        config_attr="krea_dev",
        flux_base_model="krea-dev",
        default_steps=mflux_defaults.MODEL_INFERENCE_STEPS["krea-dev"],
        default_guidance=mflux_defaults.GUIDANCE_SCALE,
        use_guidance=True,
        name_patterns=("krea",),
    ),
    _ProfileTemplate(
        profile_id="flux_dev",
        family="flux1",
        config_attr="dev",
        flux_base_model="dev",
        default_steps=mflux_defaults.MODEL_INFERENCE_STEPS["dev"],
        default_guidance=mflux_defaults.GUIDANCE_SCALE,
        use_guidance=True,
        name_patterns=("flux-dev", "flux.1-dev", "flux1-dev"),
    ),
    _ProfileTemplate(
        profile_id="flux2_klein_base_9b",
        family="flux2",
        config_attr="flux2_klein_base_9b",
        flux_base_model=None,
        default_steps=mflux_defaults.MODEL_INFERENCE_STEPS["flux2-klein-base-9b"],
        default_guidance=1.0,
        use_guidance=True,
        scheduler="flow_match_euler_discrete",
        name_patterns=("klein-base-9b",),
    ),
    _ProfileTemplate(
        profile_id="flux2_klein_base_4b",
        family="flux2",
        config_attr="flux2_klein_base_4b",
        flux_base_model=None,
        default_steps=mflux_defaults.MODEL_INFERENCE_STEPS["flux2-klein-base-4b"],
        default_guidance=1.0,
        use_guidance=True,
        scheduler="flow_match_euler_discrete",
        name_patterns=("klein-base-4b",),
    ),
    _ProfileTemplate(
        profile_id="flux2_klein_9b",
        family="flux2",
        config_attr="flux2_klein_9b",
        flux_base_model=None,
        default_steps=mflux_defaults.MODEL_INFERENCE_STEPS["flux2-klein-9b"],
        default_guidance=1.0,
        use_guidance=True,
        scheduler="flow_match_euler_discrete",
        name_patterns=("klein-9b",),
        forbidden_patterns=("base",),
    ),
    _ProfileTemplate(
        profile_id="flux2_klein_4b",
        family="flux2",
        config_attr="flux2_klein_4b",
        flux_base_model=None,
        default_steps=mflux_defaults.MODEL_INFERENCE_STEPS["flux2-klein-4b"],
        default_guidance=1.0,
        use_guidance=True,
        scheduler="flow_match_euler_discrete",
        name_patterns=("flux2", "klein"),
        forbidden_patterns=("base", "9b"),
    ),
    _ProfileTemplate(
        profile_id="z_image_turbo",
        family="z_image",
        config_attr="z_image_turbo",
        flux_base_model=None,
        default_steps=mflux_defaults.MODEL_INFERENCE_STEPS["z-image-turbo"],
        default_guidance=0.0,
        use_guidance=False,
        scheduler="flow_match_euler_discrete",
        name_patterns=("z-image-turbo",),
    ),
    _ProfileTemplate(
        profile_id="z_image",
        family="z_image",
        config_attr="z_image",
        flux_base_model=None,
        default_steps=mflux_defaults.MODEL_INFERENCE_STEPS["z-image"],
        default_guidance=mflux_defaults.GUIDANCE_SCALE,
        use_guidance=True,
        scheduler="flow_match_euler_discrete",
        name_patterns=("z-image",),
        forbidden_patterns=("turbo",),
    ),
    _ProfileTemplate(
        profile_id="qwen_image",
        family="qwen_image",
        config_attr="qwen_image",
        flux_base_model=None,
        default_steps=mflux_defaults.MODEL_INFERENCE_STEPS["qwen-image"],
        default_guidance=4.0,
        use_guidance=True,
        name_patterns=("qwen-image", "qwen_image"),
    ),
    _ProfileTemplate(
        profile_id="fibo_lite",
        family="fibo",
        config_attr="fibo_lite",
        flux_base_model=None,
        default_steps=mflux_defaults.MODEL_INFERENCE_STEPS["fibo-lite"],
        default_guidance=1.0,
        use_guidance=True,
        scheduler="flow_match_euler_discrete",
        name_patterns=("fibo-lite", "fibo_lite"),
    ),
    _ProfileTemplate(
        profile_id="fibo",
        family="fibo",
        config_attr="fibo",
        flux_base_model=None,
        default_steps=mflux_defaults.MODEL_INFERENCE_STEPS["fibo"],
        default_guidance=mflux_defaults.GUIDANCE_SCALE,
        use_guidance=True,
        scheduler="flow_match_euler_discrete",
        name_patterns=("fibo",),
    ),
    _ProfileTemplate(
        profile_id="flux_generic",
        family="flux1",
        config_attr="dev",
        flux_base_model="dev",
        default_steps=mflux_defaults.MODEL_INFERENCE_STEPS["dev"],
        default_guidance=mflux_defaults.GUIDANCE_SCALE,
        use_guidance=True,
        name_patterns=("flux",),
        forbidden_patterns=("lite", "schnell", "klein", "flux2", "z-image", "qwen", "fibo"),
    ),
)


def infer_quantize_from_name(folder_name: str) -> int | None:
    """Extract quantization bits from common Hugging Face folder naming."""
    match = re.search(r"(\d+)bit", folder_name, re.IGNORECASE)
    if match:
        bits = int(match.group(1))
        if bits in mflux_defaults.QUANTIZE_CHOICES:
            return bits

    q_match = re.search(r"[-_]q(\d)\b", folder_name, re.IGNORECASE)
    if q_match:
        bits = int(q_match.group(1))
        if bits in mflux_defaults.QUANTIZE_CHOICES:
            return bits

    return None


def _parse_flag_int(command: str, *flags: str) -> int | None:
    for flag in flags:
        match = re.search(rf"{re.escape(flag)}\s+(\d+)", command)
        if match:
            return int(match.group(1))
    return None


def _parse_flag_float(command: str, flag: str) -> float | None:
    match = re.search(rf"{re.escape(flag)}\s+([\d.]+)", command)
    if match:
        return float(match.group(1))
    return None


def _parse_flag_str(command: str, flag: str) -> str | None:
    match = re.search(rf"{re.escape(flag)}\s+(\S+)", command)
    if match:
        return match.group(1)
    return None


def parse_readme_inference_hints_from_text(content: str) -> dict[str, Any]:
    """Parse mflux-generate flags from README markdown text."""
    commands = MFLUX_GENERATE_CMD.findall(content)
    if not commands:
        return {}

    command = max(commands, key=len)
    hints: dict[str, Any] = {"source": "readme"}

    base_model = _parse_flag_str(command, "--base-model")
    if base_model:
        hints["flux_base_model"] = base_model

    steps = _parse_flag_int(command, "--steps")
    if steps is not None:
        hints["quality_steps"] = steps

    guidance = _parse_flag_float(command, "--guidance")
    if guidance is not None:
        hints["default_guidance"] = guidance
        hints["use_guidance"] = guidance > 0

    quantize = _parse_flag_int(command, "--quantize", "-q")
    if quantize is not None:
        hints["quantize"] = quantize

    width = _parse_flag_int(command, "--width")
    height = _parse_flag_int(command, "--height")
    if width is not None:
        hints["default_width"] = width
    if height is not None:
        hints["default_height"] = height

    return hints


def parse_readme_inference_hints(model_path: Path) -> dict[str, Any]:
    """Parse mflux-generate flags from a model README when present."""
    readme_path = model_path / "README.md"
    if not readme_path.is_file():
        return {}

    content = readme_path.read_text(encoding="utf-8", errors="replace")
    return parse_readme_inference_hints_from_text(content)


def _template_match_score(name: str, template: _ProfileTemplate) -> int:
    if template.forbidden_patterns and any(
        forbidden in name for forbidden in template.forbidden_patterns
    ):
        return 0
    if not template.name_patterns:
        return 0

    matched = [pattern for pattern in template.name_patterns if pattern in name]
    if not matched:
        return 0

    if template.profile_id == "flux_lite" and not all(
        pattern in name for pattern in template.name_patterns
    ):
        return 0

    return max(len(pattern) for pattern in matched)


def _match_profile_template(folder_name: str) -> _ProfileTemplate:
    name = folder_name.lower()
    best_template: _ProfileTemplate | None = None
    best_score = 0

    for template in PROFILE_REGISTRY:
        score = _template_match_score(name, template)
        if score > best_score:
            best_score = score
            best_template = template

    if best_template is None:
        raise ValueError(
            f"Unsupported image model folder '{folder_name}'. "
            "Expected flux, schnell, z-image, qwen-image, fibo, or flux2/klein in the name."
        )
    return best_template


def _apply_hints(profile: ImageModelProfile, hints: dict[str, Any]) -> ImageModelProfile:
    if not hints:
        return profile

    updates: dict[str, Any] = {}
    for field in (
        "flux_base_model",
        "quantize",
        "default_steps",
        "fast_steps",
        "quality_steps",
        "default_guidance",
        "default_width",
        "default_height",
        "use_guidance",
    ):
        if field in hints:
            updates[field] = hints[field]

    if updates:
        updates["source"] = hints.get("source", "merged")
        return cast(ImageModelProfile, replace(profile, **updates))
    return profile


def _template_to_profile(template: _ProfileTemplate, folder_name: str) -> ImageModelProfile:
    quantize = infer_quantize_from_name(folder_name)
    if template.profile_id == "flux_lite" and quantize is None:
        quantize = 4

    balanced_steps = template.default_steps
    fast_steps = template.fast_steps if template.fast_steps is not None else max(4, balanced_steps // 2)
    quality_steps = (
        template.quality_steps if template.quality_steps is not None else balanced_steps
    )

    return ImageModelProfile(
        profile_id=template.profile_id,
        family=template.family,
        config_attr=template.config_attr,
        flux_base_model=template.flux_base_model,
        quantize=quantize,
        default_steps=balanced_steps,
        fast_steps=fast_steps,
        quality_steps=quality_steps,
        default_guidance=template.default_guidance,
        default_width=template.default_width,
        default_height=template.default_height,
        scheduler=template.scheduler,
        use_guidance=template.use_guidance,
        source="registry",
    )


def resolve_image_profile(model_path: Path) -> ImageModelProfile:
    """Resolve the best inference profile for a local image checkpoint."""
    template = _match_profile_template(model_path.name)
    profile = _template_to_profile(template, model_path.name)
    hints = parse_readme_inference_hints(model_path)
    return _apply_hints(profile, hints)


def apply_quantize_override(
    profile: ImageModelProfile,
    quantize_override: int | None,
) -> ImageModelProfile:
    """Apply server_config advanced.quantize_override on top of auto-detected bits."""
    if quantize_override is None:
        return profile
    if quantize_override <= 0:
        raise ValueError("quantize_override must be a positive integer.")
    return replace(profile, quantize=quantize_override)
