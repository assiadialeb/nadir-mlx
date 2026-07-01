"""Environment variable helpers with empty-string treated as unset."""

from __future__ import annotations

import os


def env_str(key: str, fallback: str) -> str:
    """Return a stripped env value, or fallback when missing or blank."""
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return fallback
    return raw.strip()


def env_int(key: str, fallback: int) -> int:
    """Return an int env value, or fallback when missing or blank."""
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return fallback
    return int(raw)
