"""Local blob storage for OpenAI-style batch input/output files."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from orchestrator.env_utils import env_float
from orchestrator.security_utils import safe_path_under_root

_FILE_ID_PATTERN = re.compile(r"^file-[a-f0-9]{32}$")
_STORAGE_NAME_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def batch_files_dir() -> Path:
    """Directory for batch JSONL blobs (``data/batch_files`` by default)."""
    raw = os.environ.get("NADIR_BATCH_FILES_DIR")
    if raw:
        path = Path(raw)
    else:
        try:
            from django.conf import settings

            path = Path(getattr(settings, "NADIR_BATCH_FILES_DIR", ""))
            if not str(path):
                raise ValueError
        except Exception:
            path = Path(__file__).resolve().parent.parent / "data" / "batch_files"
    path.mkdir(parents=True, exist_ok=True)
    return path


def batch_file_max_bytes() -> int:
    """Max upload size for ``POST /v1/files`` (default 200 MiB)."""
    mb = env_float("NADIR_BATCH_FILE_MAX_MB", 200.0)
    return max(1, int(mb * 1024 * 1024))


def new_file_id() -> str:
    return f"file-{uuid.uuid4().hex}"


def new_storage_name() -> str:
    return uuid.uuid4().hex


def is_valid_batch_file_id(file_id: str) -> bool:
    return bool(_FILE_ID_PATTERN.match(file_id))


def storage_path_for_name(storage_name: str) -> Path | None:
    if not _STORAGE_NAME_PATTERN.match(storage_name):
        return None
    root = batch_files_dir()
    return safe_path_under_root(root, storage_name)


def write_upload_bytes(storage_name: str, data: bytes) -> Path:
    path = storage_path_for_name(storage_name)
    if path is None:
        raise ValueError("Invalid storage name")
    path.write_bytes(data)
    return path


def read_file_bytes(storage_name: str) -> bytes | None:
    path = storage_path_for_name(storage_name)
    if path is None or not path.is_file():
        return None
    return path.read_bytes()


def delete_storage_blob(storage_name: str) -> None:
    path = storage_path_for_name(storage_name)
    if path is None or not path.is_file():
        return
    path.unlink(missing_ok=True)
