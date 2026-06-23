import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Literal, Optional

from django.conf import settings

from orchestrator.security_utils import assert_path_under_directory, models_root_path

LaunchMode = Literal["TEXT", "MULTIMODAL", "EMBEDDING", "RERANKER", "IMAGE", "TTS", "STT"]

CONFIG_JSON = "config.json"
CONFIG_JSON_ORIG = "config.json.orig"

SAFE_MODEL_FOLDER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
HF_REPO_ID_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")

INVALID_MODEL_FOLDER_NAME = "Invalid model folder name."

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

TTS_NAME_PATTERN = re.compile(r"kokoro", re.IGNORECASE)

STT_NAME_PATTERN = re.compile(r"whisper", re.IGNORECASE)

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


def validate_model_folder_name(folder_name: str) -> str:
    """Reject path traversal and unsafe characters in a local model folder name."""
    name = folder_name.strip()
    if not name or name in {".", ".."}:
        raise ValueError(INVALID_MODEL_FOLDER_NAME)
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError(INVALID_MODEL_FOLDER_NAME)
    if not SAFE_MODEL_FOLDER_PATTERN.fullmatch(name):
        raise ValueError(INVALID_MODEL_FOLDER_NAME)
    return name


def validate_hf_repo_id(repo_id: str) -> str:
    """Validate a Hugging Face repo id before download (org/model format)."""
    cleaned = repo_id.strip()
    if not cleaned or ".." in cleaned or cleaned.startswith(("/", "\\")):
        raise ValueError("Invalid Hugging Face repo id.")
    if not HF_REPO_ID_PATTERN.fullmatch(cleaned):
        raise ValueError("Invalid Hugging Face repo id.")
    validate_model_folder_name(cleaned.split("/")[-1])
    return cleaned


def resolve_model_dir(folder_name: str) -> Path:
    """Return a model directory path constrained under MODELS_DIR."""
    name = validate_model_folder_name(folder_name)
    models_root = Path(settings.MODELS_DIR).resolve()
    resolved = (models_root / name).resolve()
    if not resolved.is_relative_to(models_root):
        raise ValueError("Invalid model folder path.")
    return resolved


def resolve_log_file_path(model_name: str, port: int) -> Path:
    """Return a log file path constrained under LOGS_DIR."""
    if port < 1 or port > 65535:
        raise ValueError("Invalid port.")
    name = validate_model_folder_name(model_name)
    logs_root = Path(settings.LOGS_DIR).resolve()
    resolved = (logs_root / f"{name}_{port}.log").resolve()
    if not resolved.is_relative_to(logs_root):
        raise ValueError("Invalid log file path.")
    return resolved


def get_folder_name(repo_id: str) -> str:
    """Extract the local folder name from a Hugging Face repo id."""
    return validate_model_folder_name(repo_id.split("/")[-1])


def get_model_path(folder_name: str) -> Path:
    """Return the absolute path to a model folder."""
    return resolve_model_dir(folder_name)


def _read_model_config(model_path: Path) -> dict[str, Any]:
    """Read config.json, preferring the original backup when available."""
    backup_path = model_path / CONFIG_JSON_ORIG
    config_path = backup_path if backup_path.is_file() else model_path / CONFIG_JSON
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


def _has_npz_weights(path: Path) -> bool:
    """Return True for MLX checkpoints that ship weights as .npz (legacy Whisper, etc.)."""
    weight_npz = path / "weights.npz"
    if weight_npz.is_file() and weight_npz.stat().st_size > 0:
        return True

    config = _read_model_config(path)
    if config.get("model_type", "").lower() != "whisper":
        return False

    npz_files = list(path.glob("*.npz"))
    return bool(npz_files) and all(file.stat().st_size > 0 for file in npz_files)


def is_model_complete(model_path: os.PathLike[str] | str) -> bool:
    """Check whether a model directory contains all required weight files."""
    path = Path(model_path)
    if _looks_like_image_model_folder(path.name):
        return _has_diffusion_weights(path)

    config_path = path / CONFIG_JSON
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
    if safetensors and all(file.stat().st_size > 0 for file in safetensors):
        return True

    return _has_npz_weights(path)


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


def is_tts_focused_model(model_path: os.PathLike[str] | str) -> bool:
    """Return True when the model should be launched as TTS, not chat."""
    path = Path(model_path)
    if not is_model_complete(path):
        return False
    if TTS_NAME_PATTERN.search(path.name):
        return True
    config = _read_model_config(path)
    return config.get("model_type", "").lower() == "kokoro"


def supports_tts_mode(model_path: os.PathLike[str] | str) -> bool:
    """Return True when the model can be served with mlx-audio Kokoro."""
    return is_tts_focused_model(model_path)


def is_stt_focused_model(model_path: os.PathLike[str] | str) -> bool:
    """Return True when the model should be launched as STT, not chat."""
    path = Path(model_path)
    if not is_model_complete(path):
        return False
    if STT_NAME_PATTERN.search(path.name):
        return True
    config = _read_model_config(path)
    model_type = config.get("model_type", "").lower()
    architectures = " ".join(config.get("architectures", [])).lower()
    return model_type == "whisper" or "whisper" in architectures


def supports_stt_mode(model_path: os.PathLike[str] | str) -> bool:
    """Return True when the model can be served with mlx-audio Whisper."""
    from orchestrator.whisper_assets import is_stt_servable

    return is_stt_servable(model_path)


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
    tts_focused = is_tts_focused_model(model_path)
    stt_focused = is_stt_focused_model(model_path)
    return {
        "supports_text": is_model_complete(model_path)
        and not embedding_focused
        and not rerank_focused
        and not image_focused
        and not tts_focused
        and not stt_focused,
        "supports_multimodal": supports_multimodal_mode(model_path)
        and not image_focused
        and not tts_focused
        and not stt_focused,
        "supports_embedding": supports_embedding_mode(model_path),
        "supports_rerank": supports_rerank_mode(model_path),
        "supports_image": supports_image_mode(model_path),
        "supports_tts": supports_tts_mode(model_path),
        "supports_stt": supports_stt_mode(model_path),
    }


def get_model_folder_size_bytes(folder_name: str) -> int:
    """Return total on-disk size for a model folder (bytes)."""
    model_path = assert_path_under_directory(get_model_path(folder_name), models_root_path())
    if not model_path.is_dir():
        return 0

    total = 0
    for root, _dirs, files in os.walk(model_path):
        current_root = assert_path_under_directory(Path(root), models_root_path())
        for file_name in files:
            file_path = assert_path_under_directory(current_root / file_name, models_root_path())
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def prepare_model_for_text_inference(model_path: os.PathLike[str] | str) -> None:
    """Patch configs that mlx_lm cannot load natively (e.g. gemma4_unified)."""
    path = assert_path_under_directory(Path(model_path), models_root_path())
    config_path = path / CONFIG_JSON
    if not config_path.is_file():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    if model_type != "gemma4_unified":
        return

    backup_path = path / CONFIG_JSON_ORIG
    if not backup_path.is_file():
        shutil.copy2(config_path, backup_path)

    config["model_type"] = "gemma4"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def prepare_model_for_multimodal_inference(model_path: os.PathLike[str] | str) -> None:
    """Restore the original unified config required by mlx_vlm."""
    path = assert_path_under_directory(Path(model_path), models_root_path())
    backup_path = path / CONFIG_JSON_ORIG
    config_path = path / CONFIG_JSON
    if backup_path.is_file():
        shutil.copy2(backup_path, config_path)


def _safetensors_metadata(model_path: Path) -> dict[str, str]:
    """Read safetensors header metadata from the primary weights file."""
    weight_path = model_path / "model.safetensors"
    if not weight_path.is_file():
        return {}
    try:
        from safetensors import safe_open
    except ImportError:
        return {}
    try:
        with safe_open(str(weight_path), framework="np") as handle:
            return dict(handle.metadata() or {})
    except (OSError, RuntimeError, ValueError):
        return {}


def _requires_relaxed_gemma4_kv_shared_weights(model_path: Path) -> bool:
    """Return True for mlx-community Gemma 4 E2B/E4B KV-shared checkpoints."""
    config = _read_model_config(model_path)
    if config.get("model_type") != "gemma4":
        return False

    text_config = config.get("text_config") or {}
    if not text_config.get("num_kv_shared_layers"):
        return False

    return _safetensors_metadata(model_path).get("format") == "mlx"


def requires_relaxed_weight_loading(model_path: os.PathLike[str] | str) -> bool:
    """Return True when mlx_lm / mlx_vlm must ignore extra weight tensors."""
    path = Path(model_path)
    backup_path = path / CONFIG_JSON_ORIG
    if backup_path.is_file():
        return True
    if get_model_type(model_path) == "gemma4_unified":
        return True
    return _requires_relaxed_gemma4_kv_shared_weights(path)


def sync_model_download_status() -> None:
    """Reconcile ModelDownload records with the actual filesystem state."""
    from .models import ModelDownload

    for record in ModelDownload.objects.all():
        if is_model_complete(record.local_path) and record.status != "COMPLETED":
            record.status = "COMPLETED"
            record.error_message = ""
            record.save(update_fields=["status", "error_message"])


def reconcile_stale_downloads() -> None:
    """Mark interrupted downloads as failed; promote completed ones after a restart."""
    from .models import ModelDownload

    for record in ModelDownload.objects.filter(status="DOWNLOADING"):
        if is_model_complete(record.local_path):
            record.status = "COMPLETED"
            record.error_message = ""
            record.save(update_fields=["status", "error_message"])
            continue

        record.status = "FAILED"
        record.error_message = "Download interrupted by server restart."
        record.save(update_fields=["status", "error_message"])
