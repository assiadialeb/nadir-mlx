"""Detect mlx-lm server CLI features (version-dependent knobs)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Iterable

from django.conf import settings

TEXT_KV_CLI_FLAGS: tuple[str, ...] = ("kv-bits", "kv-quant-scheme", "kv-group-size")
_PROBE_FAILURE_TTL_SECONDS = 300.0

_cached_cli_flags: frozenset[str] | None = None
_probe_failed_until: float = 0.0


def _python_for_mlx_lm() -> str:
    """Same interpreter as inference launch (``server_manager._get_python_bin``)."""
    venv_python = os.path.join(settings.BASE_DIR, "venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def mlx_lm_server_cli_flags() -> frozenset[str]:
    """Return long-option names from ``python -m mlx_lm.server --help``.

    Successful probes are cached for the process lifetime. Probe failures are
    cached briefly (``_PROBE_FAILURE_TTL_SECONDS``) so a missing/broken mlx-lm
    does not re-run ``--help`` on every form validation or schema render.
    """
    global _cached_cli_flags, _probe_failed_until
    if _cached_cli_flags is not None:
        return _cached_cli_flags
    if time.monotonic() < _probe_failed_until:
        return frozenset()
    flags = _probe_mlx_lm_server_cli_flags()
    if flags:
        _cached_cli_flags = flags
        _probe_failed_until = 0.0
        return flags
    _probe_failed_until = time.monotonic() + _PROBE_FAILURE_TTL_SECONDS
    return frozenset()


def mlx_lm_server_supports_flag(flag: str) -> bool:
    """True when ``--{flag}`` appears on the installed mlx-lm server."""
    normalized = flag.strip().lstrip("-")
    return normalized in mlx_lm_server_cli_flags()


def text_kv_quantization_enabled() -> bool:
    """TEXT may use KV quant keys when mlx-lm exposes all related server flags."""
    override = os.environ.get("NADIR_TEXT_KV_BITS_PREVIEW", "").strip()
    if override == "1":
        return True
    if override == "0":
        return False
    return all(mlx_lm_server_supports_flag(name) for name in TEXT_KV_CLI_FLAGS)


def clear_mlx_lm_capability_cache() -> None:
    """Tests only."""
    global _cached_cli_flags, _probe_failed_until
    _cached_cli_flags = None
    _probe_failed_until = 0.0


def _probe_mlx_lm_server_cli_flags() -> frozenset[str]:
    try:
        result = subprocess.run(
            [_python_for_mlx_lm(), "-m", "mlx_lm.server", "--help"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return frozenset()
    if getattr(result, "returncode", 0) not in (0, None) and not (
        result.stdout or result.stderr
    ):
        return frozenset()
    text = (result.stdout or "") + (result.stderr or "")
    parsed = frozenset(_parse_help_flags(text))
    return parsed


def _parse_help_flags(help_text: str) -> Iterable[str]:
    for line in help_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("--"):
            continue
        token = stripped.split(",", 1)[0].strip()
        if not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0].split()[0].strip()
        if name:
            yield name
