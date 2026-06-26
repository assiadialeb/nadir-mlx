"""Database configuration for Nadir (SQLite local dev or external PostgreSQL)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _postgres_from_url(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme not in ("postgres", "postgresql"):
        raise ValueError("NADIR_DATABASE_URL must use postgres:// or postgresql://")
    database = (parsed.path or "").lstrip("/") or "nadir_db"
    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": database,
        "USER": unquote(parsed.username or ""),
        "PASSWORD": unquote(parsed.password or ""),
        "HOST": parsed.hostname or "127.0.0.1",
        "PORT": str(parsed.port or 5432),
        "CONN_MAX_AGE": int(os.environ.get("NADIR_DB_CONN_MAX_AGE", "60")),
        "OPTIONS": _postgres_ssl_options(),
    }


def _postgres_from_env() -> dict[str, Any] | None:
    url = os.environ.get("NADIR_DATABASE_URL", "").strip()
    if url:
        return _postgres_from_url(url)

    host = os.environ.get("NADIR_DB_HOST", "").strip()
    if not host:
        return None

    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("NADIR_DB_NAME", "nadir_db"),
        "USER": os.environ.get("NADIR_DB_USER", "nadir_user"),
        "PASSWORD": os.environ.get("NADIR_DB_PASSWORD", ""),
        "HOST": host,
        "PORT": os.environ.get("NADIR_DB_PORT", "5432"),
        "CONN_MAX_AGE": int(os.environ.get("NADIR_DB_CONN_MAX_AGE", "60")),
        "OPTIONS": _postgres_ssl_options(),
    }


def _postgres_ssl_options() -> dict[str, str]:
    if _env_bool("NADIR_DB_SSL"):
        return {"sslmode": os.environ.get("NADIR_DB_SSLMODE", "prefer")}
    return {}


def build_database_config(base_dir: Path) -> dict[str, dict[str, Any]]:
    """Return Django DATABASES with SQLite fallback when Postgres is not configured."""
    postgres = _postgres_from_env()
    if postgres is not None:
        return {"default": postgres}
    return {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": base_dir / "db.sqlite3",
        }
    }
