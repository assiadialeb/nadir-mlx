"""Tests for local Django secret key persistence."""

from __future__ import annotations

import stat
from pathlib import Path

from mlx_orchestrator.local_secret import resolve_django_secret_key, secret_key_path


def test_resolve_prefers_environment(tmp_path: Path) -> None:
    key = resolve_django_secret_key(
        data_dir=tmp_path,
        env={"DJANGO_SECRET_KEY": "from-env"},
        allow_generate=True,
    )
    assert key == "from-env"
    assert not secret_key_path(tmp_path).exists()


def test_resolve_reads_persisted_file(tmp_path: Path) -> None:
    path = secret_key_path(tmp_path)
    path.write_text("persisted-key\n", encoding="utf-8")
    key = resolve_django_secret_key(data_dir=tmp_path, env={}, allow_generate=True)
    assert key == "persisted-key"


def test_resolve_generates_and_persists_mode_600(tmp_path: Path) -> None:
    key = resolve_django_secret_key(data_dir=tmp_path, env={}, allow_generate=True)
    path = secret_key_path(tmp_path)
    assert key
    assert path.read_text(encoding="utf-8").strip() == key
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600 or mode == stat.S_IMODE(path.stat().st_mode)


def test_resolve_allow_generate_false_returns_empty(tmp_path: Path) -> None:
    key = resolve_django_secret_key(data_dir=tmp_path, env={}, allow_generate=False)
    assert key == ""
    assert not secret_key_path(tmp_path).exists()
