"""Detect mlx-lm server CLI features (version-dependent knobs)."""

from __future__ import annotations

import functools
import subprocess
import sys
from typing import Iterable


def _python_for_mlx_lm() -> str:
    return sys.executable


@functools.lru_cache(maxsize=1)
def mlx_lm_server_cli_flags() -> frozenset[str]:
    """Return long-option names from ``python -m mlx_lm.server --help``."""
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
    text = (result.stdout or "") + (result.stderr or "")
    return frozenset(_parse_help_flags(text))


def mlx_lm_server_supports_flag(flag: str) -> bool:
    """True when ``--{flag}`` appears on the installed mlx-lm server."""
    normalized = flag.strip().lstrip("-")
    return normalized in mlx_lm_server_cli_flags()


def text_kv_quantization_enabled() -> bool:
    """TEXT may use ``kv_bits`` when mlx-lm exposes ``--kv-bits``."""
    import os

    override = os.environ.get("NADIR_TEXT_KV_BITS_PREVIEW", "").strip()
    if override == "1":
        return True
    if override == "0":
        return False
    return mlx_lm_server_supports_flag("kv-bits")


def clear_mlx_lm_capability_cache() -> None:
    """Tests only."""
    mlx_lm_server_cli_flags.cache_clear()


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
