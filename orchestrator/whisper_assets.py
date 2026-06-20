"""Bootstrap Hugging Face Whisper assets for legacy MLX .npz checkpoints."""

from __future__ import annotations

import re
from pathlib import Path

WHISPER_ASSET_FILENAMES: tuple[str, ...] = (
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "generation_config.json",
)


def is_legacy_mlx_whisper_checkpoint(model_path: Path) -> bool:
    """Return True for mlx-examples style Whisper folders (weights.npz only)."""
    if not model_path.is_dir():
        return False
    if not (model_path / "weights.npz").is_file():
        return False
    if (model_path / "model.safetensors").is_file():
        return False
    if (model_path / "model.safetensors.index.json").is_file():
        return False
    config = model_path / "config.json"
    if not config.is_file():
        return False
    return True


def resolve_legacy_whisper_processor_repo(folder_name: str) -> str | None:
    """Map mlx-community legacy Whisper folder names to OpenAI processor repos."""
    name = folder_name.lower()

    if re.search(r"small\.en|small-en", name):
        return "openai/whisper-small.en"
    if re.search(r"tiny\.en|tiny-en", name):
        return "openai/whisper-tiny.en"
    if re.search(r"base\.en|base-en", name):
        return "openai/whisper-base.en"
    if re.search(r"medium\.en|medium-en", name):
        return "openai/whisper-medium.en"
    if "large-v3" in name or "large_v3" in name:
        return "openai/whisper-large-v3"
    if re.search(r"\blarge\b", name):
        return "openai/whisper-large-v3"
    if re.search(r"\bmedium\b", name):
        return "openai/whisper-medium"
    if re.search(r"\bsmall\b", name):
        return "openai/whisper-small"
    if re.search(r"\btiny\b", name):
        return "openai/whisper-tiny"
    if re.search(r"\bbase\b", name):
        return "openai/whisper-base"
    return None


def has_whisper_processor_assets(model_path: Path) -> bool:
    return (model_path / "preprocessor_config.json").is_file()


def ensure_whisper_hf_assets(model_path: os.PathLike[str] | str) -> None:
    """Download missing tokenizer/processor files for legacy MLX Whisper checkpoints."""
    path = Path(model_path).resolve()
    if has_whisper_processor_assets(path):
        return

    processor_repo = resolve_legacy_whisper_processor_repo(path.name)
    if processor_repo is None:
        raise ValueError(
            f"Whisper model '{path.name}' is missing preprocessor/tokenizer files "
            "required by mlx-audio. Use a full mlx-community checkpoint such as "
            "whisper-large-v3-turbo-asr-fp16, or a known whisper-*-mlx folder name."
        )

    from huggingface_hub import hf_hub_download

    for filename in WHISPER_ASSET_FILENAMES:
        target = path / filename
        if target.is_file():
            continue
        try:
            hf_hub_download(
                processor_repo,
                filename,
                local_dir=str(path),
            )
        except Exception:
            continue

    if not has_whisper_processor_assets(path):
        raise ValueError(
            f"Failed to bootstrap Whisper processor assets for '{path.name}' "
            f"from {processor_repo}."
        )


def is_stt_servable(model_path: os.PathLike[str] | str) -> bool:
    """Return True when mlx-audio can transcribe with this local folder."""
    path = Path(model_path)
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    if not (
        (path / "model.safetensors").is_file()
        or (path / "weights.npz").is_file()
        or list(path.glob("*.safetensors"))
    ):
        return False

    try:
        import json

        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    model_type = config.get("model_type", "").lower()
    is_whisper = (
        "whisper" in path.name.lower()
        or model_type == "whisper"
        or "whisper" in " ".join(config.get("architectures", [])).lower()
    )
    if not is_whisper:
        return False

    if has_whisper_processor_assets(path):
        return True

    if is_legacy_mlx_whisper_checkpoint(path):
        return resolve_legacy_whisper_processor_repo(path.name) is not None

    return False
