import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Literal, Optional

from django.conf import settings

LaunchMode = Literal["TEXT", "MULTIMODAL", "EMBEDDING", "RERANKER", "IMAGE"]

EMBEDDING_NAME_PATTERN = re.compile(
    r"(embedding|embed|e5|bge|nomic|gte|retrieval)",
    re.IGNORECASE,
)

RERANK_NAME_PATTERN = re.compile(
    r"(rerank|reranker|jina-reranker|bge-reranker)",
    re.IGNORECASE,
)

IMAGE_NAME_PATTERN = re.compile(
    r"(flux|schnell|z-image|z_image|qwen-image|qwen_image|klein|fibo|text-to-image)",
    re.IGNORECASE,
)

EMBEDDING_MODEL_TYPES = {
    "bert",
    "xlm_roberta",
    "modernbert",
    "llama_bidirec",
    "lfm2",
    "gemma3_text",
    "colqwen2_5",
    "colidefics3",
    "siglip",
    "qwen3_vl",
    "llama_nemotron_vl",
}

VLM_MODEL_TYPES = {
    "gemma4_unified",
    "gemma4",
    "gemma3",
    "gemma3n",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "llava",
    "llava_baichuan",
    "idefics2",
    "idefics3",
    "mllama",
    "pixtral",
    "smolvlm",
    "paligemma",
    "aya_vision",
    "internvl",
    "deepseek_vl",
    "deepseek_vl_v2",
    "kimi_vl",
    "molmo",
    "lfm2_vl",
}


def get_folder_name(repo_id: str) -> str:
    """Extract the local folder name from a Hugging Face repo id."""
    return repo_id.split("/")[-1]


def get_model_path(folder_name: str) -> Path:
    """Return the absolute path to a model folder."""
    return Path(settings.MODELS_DIR) / folder_name


def _read_model_config(model_path: Path) -> dict[str, Any]:
    """Read config.json, preferring the original backup when available."""
    backup_path = model_path / "config.json.orig"
    config_path = backup_path if backup_path.is_file() else model_path / "config.json"
    if not config_path.is_file():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _has_diffusion_weights(path: Path) -> bool:
    """Return True when a diffusion checkpoint contains usable safetensors."""
    safetensors = list(path.rglob("*.safetensors"))
    return bool(safetensors) and any(file.stat().st_size > 0 for file in safetensors)


def is_model_complete(model_path: os.PathLike[str] | str) -> bool:
    """Check whether a model directory contains all required weight files."""
    path = Path(model_path)
    if _looks_like_image_model_folder(path.name):
        return _has_diffusion_weights(path)

    config_path = path / "config.json"
    if not config_path.is_file():
        return False

    single_weight = path / "model.safetensors"
    if single_weight.is_file() and single_weight.stat().st_size > 0:
        return True

    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        shard_files = set(index.get("weight_map", {}).values())
        if not shard_files:
            return False
        return all(
            (path / shard).is_file() and (path / shard).stat().st_size > 0
            for shard in shard_files
        )

    safetensors = list(path.glob("*.safetensors"))
    return bool(safetensors) and all(f.stat().st_size > 0 for f in safetensors)


def get_model_type(model_path: os.PathLike[str] | str) -> Optional[str]:
    """Read the native model_type from the model config."""
    config = _read_model_config(Path(model_path))
    return config.get("model_type")


def supports_multimodal_mode(model_path: os.PathLike[str] | str) -> bool:
    """Return True when the model can be served with mlx_vlm."""
    path = Path(model_path)
    config = _read_model_config(path)
    model_type = config.get("model_type", "")

    if model_type == "gemma4_unified":
        return True
    if model_type in VLM_MODEL_TYPES and model_type != "gemma4":
        return True
    if config.get("vision_config") or config.get("image_token_id"):
        return True
    if model_type == "gemma4" and (path / "processor_config.json").is_file():
        return True
    return False


def _mlx_embeddings_supports_config(config: dict[str, Any]) -> bool:
    try:
        from mlx_embeddings.utils import _get_model_arch

        _get_model_arch(config)
        return True
    except Exception:
        return False


def _looks_like_image_model_folder(folder_name: str) -> bool:
    return bool(IMAGE_NAME_PATTERN.search(folder_name))


def is_image_focused_model(model_path: os.PathLike[str] | str) -> bool:
    """Return True when the model should be launched as image generation, not chat."""
    path = Path(model_path)
    if not _looks_like_image_model_folder(path.name):
        return False
    return _has_diffusion_weights(path)


def supports_image_mode(model_path: os.PathLike[str] | str) -> bool:
    """Return True when the model can be served with mflux."""
    return is_image_focused_model(model_path)


def is_rerank_focused_model(model_path: os.PathLike[str] | str) -> bool:
    """Return True when the model should be launched as reranker, not chat/embed."""
    path = Path(model_path)
    if not is_model_complete(path):
        return False

    if RERANK_NAME_PATTERN.search(path.name):
        return True

    config = _read_model_config(path)
    architectures = " ".join(config.get("architectures", [])).lower()
    return "rerank" in architectures


def supports_rerank_mode(model_path: os.PathLike[str] | str) -> bool:
    """Return True when the model can be served with local-reranker (MLX backend)."""
    return is_rerank_focused_model(model_path)


def supports_embedding_mode(model_path: os.PathLike[str] | str) -> bool:
    """Return True when the model can be served with mlx-embeddings."""
    path = Path(model_path)
    if not is_model_complete(path):
        return False

    if is_rerank_focused_model(path):
        return False

    config = _read_model_config(path)
    if not config:
        return False

    folder_name = path.name
    model_type = config.get("model_type", "").replace("-", "_")
    architectures = " ".join(config.get("architectures", [])).lower()

    if EMBEDDING_NAME_PATTERN.search(folder_name):
        return _mlx_embeddings_supports_config(config)

    if model_type in EMBEDDING_MODEL_TYPES:
        return _mlx_embeddings_supports_config(config)

    if "embedding" in architectures or "embed" in architectures:
        return _mlx_embeddings_supports_config(config)

    return False


def is_embedding_focused_model(model_path: os.PathLike[str] | str) -> bool:
    """Return True when the model should be launched as embedding, not chat."""
    path = Path(model_path)
    if not supports_embedding_mode(path):
        return False

    config = _read_model_config(path)
    model_type = config.get("model_type", "").replace("-", "_")
    if EMBEDDING_NAME_PATTERN.search(path.name):
        return True
    if model_type in EMBEDDING_MODEL_TYPES:
        return True
    return "embedding" in " ".join(config.get("architectures", [])).lower()


def get_model_capabilities(folder_name: str) -> dict[str, bool]:
    """Return launch capabilities for a downloaded model folder."""
    model_path = get_model_path(folder_name)
    embedding_focused = is_embedding_focused_model(model_path)
    rerank_focused = is_rerank_focused_model(model_path)
    image_focused = is_image_focused_model(model_path)
    return {
        "supports_text": is_model_complete(model_path)
        and not embedding_focused
        and not rerank_focused
        and not image_focused,
        "supports_multimodal": supports_multimodal_mode(model_path)
        and not image_focused,
        "supports_embedding": supports_embedding_mode(model_path),
        "supports_rerank": supports_rerank_mode(model_path),
        "supports_image": supports_image_mode(model_path),
    }


def prepare_model_for_text_inference(model_path: os.PathLike[str] | str) -> None:
    """Patch configs that mlx_lm cannot load natively (e.g. gemma4_unified)."""
    path = Path(model_path)
    config_path = path / "config.json"
    if not config_path.is_file():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    if model_type != "gemma4_unified":
        return

    backup_path = path / "config.json.orig"
    if not backup_path.is_file():
        shutil.copy2(config_path, backup_path)

    config["model_type"] = "gemma4"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def prepare_model_for_multimodal_inference(model_path: os.PathLike[str] | str) -> None:
    """Restore the original unified config required by mlx_vlm."""
    path = Path(model_path)
    backup_path = path / "config.json.orig"
    config_path = path / "config.json"
    if backup_path.is_file():
        shutil.copy2(backup_path, config_path)


def requires_relaxed_weight_loading(model_path: os.PathLike[str] | str) -> bool:
    """Return True when mlx_lm must ignore extra weight tensors."""
    backup_path = Path(model_path) / "config.json.orig"
    if backup_path.is_file():
        return True
    return get_model_type(model_path) == "gemma4_unified"


def sync_model_download_status() -> None:
    """Reconcile ModelDownload records with the actual filesystem state."""
    from .models import ModelDownload

    for record in ModelDownload.objects.all():
        if is_model_complete(record.local_path):
            if record.status != "COMPLETED":
                record.status = "COMPLETED"
                record.error_message = None
                record.save(update_fields=["status", "error_message"])


def reconcile_stale_downloads() -> None:
    """Mark interrupted downloads as failed; promote completed ones after a restart."""
    from .models import ModelDownload

    for record in ModelDownload.objects.filter(status="DOWNLOADING"):
        if is_model_complete(record.local_path):
            record.status = "COMPLETED"
            record.error_message = None
            record.save(update_fields=["status", "error_message"])
            continue

        record.status = "FAILED"
        record.error_message = "Download interrupted by server restart."
        record.save(update_fields=["status", "error_message"])
