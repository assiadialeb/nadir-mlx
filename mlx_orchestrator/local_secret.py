"""Persist a local Django secret key for first-run ``manage.py serve``."""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

SECRET_FILENAME = "django_secret_key"
ENV_VAR = "DJANGO_SECRET_KEY"


def secret_key_path(data_dir: Path) -> Path:
    """Return the on-disk path for the persisted secret key."""
    return data_dir / SECRET_FILENAME


def _read_persisted_secret(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _write_persisted_secret(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        logger.warning("Could not chmod 600 on %s", path)


def resolve_django_secret_key(
    *,
    data_dir: Path,
    env: dict[str, str] | None = None,
    allow_generate: bool = True,
) -> str:
    """Resolve DJANGO_SECRET_KEY from env, then disk, optionally generating one.

    Generated keys are written under ``data_dir`` with mode ``0o600`` so a fresh
    clone can run ``manage.py serve`` without a hand-written ``.env``.
    """
    environ = env if env is not None else os.environ
    from_env = (environ.get(ENV_VAR) or "").strip()
    if from_env:
        return from_env

    path = secret_key_path(data_dir)
    existing = _read_persisted_secret(path)
    if existing:
        return existing

    if not allow_generate:
        return ""

    generated = secrets.token_urlsafe(48)
    _write_persisted_secret(path, generated)
    logger.info("Generated local Django secret key at %s", path)
    return generated
