"""Local PNG storage and public URLs for OpenAI-style image responses."""

from __future__ import annotations

import base64
import os
import re
import time
import uuid
from pathlib import Path

from orchestrator.security_utils import safe_path_under_root

_FILE_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def _project_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def image_output_dir() -> Path:
    """Return the directory where generated PNG assets are stored."""
    raw = os.environ.get("IMAGE_OUTPUT_DIR")
    if raw:
        path = Path(raw)
    else:
        try:
            from django.conf import settings

            path = Path(settings.IMAGE_OUTPUT_DIR)
        except Exception:
            path = _project_base_dir() / "data" / "generated_images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_output_ttl_seconds() -> int:
    """Return TTL for locally stored generated images."""
    raw = os.environ.get("IMAGE_OUTPUT_TTL_SECONDS")
    if raw:
        try:
            return max(60, int(raw))
        except ValueError:
            pass
    try:
        from django.conf import settings

        return int(getattr(settings, "IMAGE_OUTPUT_TTL_SECONDS", 3600))
    except Exception:
        return 3600


def gateway_public_base_url() -> str:
    """Return the client-facing base URL used in image response URLs."""
    from django.conf import settings

    from orchestrator.env_utils import env_int, env_str

    env_url = env_str("NADIR_GATEWAY_PUBLIC_BASE_URL", "")
    if env_url:
        return env_url.rstrip("/")

    configured = str(getattr(settings, "NADIR_GATEWAY_PUBLIC_BASE_URL", "")).strip()
    if configured:
        return configured.rstrip("/")

    try:
        host = env_str("NADIR_GATEWAY_HOST", str(settings.NADIR_GATEWAY_HOST))
        port = env_int("NADIR_GATEWAY_PORT", int(settings.NADIR_GATEWAY_PORT))
        return f"http://{host}:{port}"
    except Exception:
        return "http://127.0.0.1:11380"


def is_valid_image_file_id(file_id: str) -> bool:
    """Return True when the id matches a stored asset filename stem."""
    return bool(_FILE_ID_PATTERN.match(file_id))


def purge_expired_image_assets() -> None:
    """Delete PNG assets older than the configured TTL."""
    output_dir = image_output_dir()
    cutoff = time.time() - image_output_ttl_seconds()
    for path in output_dir.glob("*.png"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue


def store_image_png(png_bytes: bytes) -> str:
    """Persist PNG bytes and return a opaque file id."""
    purge_expired_image_assets()
    file_id = uuid.uuid4().hex
    target = image_output_dir() / f"{file_id}.png"
    target.write_bytes(png_bytes)
    return file_id


def resolve_image_png_path(file_id: str) -> Path | None:
    """Resolve a file id to an on-disk PNG path when it exists."""
    if not is_valid_image_file_id(file_id):
        return None
    try:
        path = safe_path_under_root(image_output_dir().resolve(), f"{file_id}.png")
    except ValueError:
        return None
    if not path.is_file():
        return None
    return path


def build_public_image_url(file_id: str) -> str:
    """Build the OpenAI-style URL served by Nadir Gateway."""
    return f"{gateway_public_base_url()}/v1/images/files/{file_id}"


def build_generation_response_entry(
    png_bytes: bytes,
    response_format: str,
) -> dict[str, str]:
    """Build one OpenAI images.generations data entry."""
    if response_format == "url":
        file_id = store_image_png(png_bytes)
        return {"url": build_public_image_url(file_id)}
    return {"b64_json": base64.b64encode(png_bytes).decode("ascii")}
