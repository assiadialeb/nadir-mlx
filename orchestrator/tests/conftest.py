"""Pytest hooks for CI (ubuntu-latest) where MLX runtime is not available."""

from __future__ import annotations

import importlib.machinery
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _make_stub_module(name: str, *, is_package: bool = False) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__spec__", None) is not None:
        return existing

    module = ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    if is_package:
        module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def _stub_mlx_runtime() -> None:
    """Stub MLX imports for test collection without pulling CUDA wheels on Linux CI."""
    if not _env_flag("NADIR_CI_STUB_MLX"):
        return

    mlx = _make_stub_module("mlx", is_package=True)
    mlx_core = MagicMock(name="mlx.core")
    mlx_core.__spec__ = importlib.machinery.ModuleSpec("mlx.core", loader=None)
    mlx_core.array = MagicMock(name="mlx.array")
    sys.modules["mlx.core"] = mlx_core
    mlx.core = mlx_core
    setattr(mlx, "core", mlx_core)

    for package_name in (
        "mlx.nn",
        "mlx_lm",
        "mlx_lm.server",
        "mlx_lm.utils",
        "mlx_vlm",
        "mlx_vlm.utils",
        "mlx_vlm.server.cli",
        "mlx_audio",
        "mlx_audio.utils",
        "mlx_audio.stt.utils",
        "mlx_embeddings",
        "mlx_embeddings.utils",
    ):
        _make_stub_module(package_name, is_package=True)

    mlx_audio_io = MagicMock(name="mlx_audio.audio_io")
    mlx_audio_io.__spec__ = importlib.machinery.ModuleSpec("mlx_audio.audio_io", loader=None)
    mlx_audio_io.write = MagicMock(name="mlx_audio.audio_io.write")
    mlx_audio_io.read = MagicMock(name="mlx_audio.audio_io.read")
    sys.modules["mlx_audio.audio_io"] = mlx_audio_io

    mlx_stt_utils = MagicMock(name="mlx_audio.stt.utils")
    mlx_stt_utils.__spec__ = importlib.machinery.ModuleSpec("mlx_audio.stt.utils", loader=None)
    mlx_stt_utils.SAMPLE_RATE = 16_000
    mlx_stt_utils.resample_audio = MagicMock(name="mlx_audio.stt.utils.resample_audio")
    sys.modules["mlx_audio.stt.utils"] = mlx_stt_utils


_stub_mlx_runtime()
