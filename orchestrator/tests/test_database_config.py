"""Tests for database configuration helpers."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_orchestrator.database import build_database_config
from mlx_orchestrator.settings import _build_csrf_trusted_origins


class BuildDatabaseConfigTests(unittest.TestCase):
    def test_defaults_to_sqlite_without_postgres_env(self) -> None:
        base_dir = Path("/tmp/nadir-test")
        with patch.dict(os.environ, {}, clear=True):
            config = build_database_config(base_dir)
        self.assertEqual(config["default"]["ENGINE"], "django.db.backends.sqlite3")
        self.assertEqual(config["default"]["NAME"], base_dir / "db.sqlite3")

    def test_uses_postgres_when_host_is_set(self) -> None:
        base_dir = Path("/tmp/nadir-test")
        env = {
            "NADIR_DB_HOST": "127.0.0.1",
            "NADIR_DB_NAME": "nadir_db",
            "NADIR_DB_USER": "nadir_user",
            "NADIR_DB_PASSWORD": "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = build_database_config(base_dir)
        self.assertEqual(config["default"]["ENGINE"], "django.db.backends.postgresql")
        self.assertEqual(config["default"]["NAME"], "nadir_db")
        self.assertEqual(config["default"]["HOST"], "127.0.0.1")
        self.assertEqual(config["default"]["PORT"], "5432")

    def test_builds_csrf_origins_from_allowed_hosts(self) -> None:
        with patch.dict(
            os.environ,
            {"DJANGO_ALLOWED_HOSTS": "127.0.0.1,192.168.1.134", "DJANGO_HTTP_PORT": "8000"},
            clear=True,
        ):
            origins = _build_csrf_trusted_origins(["127.0.0.1", "192.168.1.134"])
        self.assertIn("http://127.0.0.1:8000", origins)
        self.assertIn("http://192.168.1.134:8000", origins)

    def test_explicit_csrf_origins_override_auto_build(self) -> None:
        env = {
            "DJANGO_CSRF_TRUSTED_ORIGINS": "https://nadir.example.com",
            "DJANGO_ALLOWED_HOSTS": "127.0.0.1",
        }
        with patch.dict(os.environ, env, clear=True):
            origins = _build_csrf_trusted_origins(["127.0.0.1"])
        self.assertEqual(origins, ["https://nadir.example.com"])

    def test_database_url_overrides_discrete_vars(self) -> None:
        base_dir = Path("/tmp/nadir-test")
        env = {
            "NADIR_DATABASE_URL": "postgresql://nadir_user:pw@127.0.0.1:5432/nadir_db",
            "NADIR_DB_HOST": "ignored",
        }
        with patch.dict(os.environ, env, clear=True):
            config = build_database_config(base_dir)
        self.assertEqual(config["default"]["USER"], "nadir_user")
        self.assertEqual(config["default"]["PASSWORD"], "pw")
