"""Default mlx-lm prompt-cache sizing from host unified memory (MLX-125)."""

from __future__ import annotations

import logging
import platform
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_SIZE = 10
_MIN_CACHE_BYTES = 512 * 1024**2
_MAX_CACHE_BYTES = 8 * 1024**3
_CACHE_BYTES_FRACTION = 0.02


def host_total_bytes() -> int | None:
    """Return total unified/system RAM in bytes, or None if unknown."""
    if platform.system() == "Darwin":
        return _darwin_total_bytes()
    return _linux_total_bytes()


def default_prompt_cache_settings(
    total_bytes: int | None = None,
) -> dict[str, int]:
    """Return RAM-aware ``prompt_cache_size`` and ``prompt_cache_bytes`` defaults.

    mlx-lm defaults to ``--prompt-cache-size 10`` with no byte cap. On Apple
    Silicon, a small fraction of unified memory is reserved for the LRU of
    distinct prefix KV caches so multi-turn agents keep hitting shared
    system+tools prefixes without unbounded growth.
    """
    total = total_bytes if total_bytes is not None else host_total_bytes()
    size = _DEFAULT_CACHE_SIZE
    if total is None or total <= 0:
        return {
            "prompt_cache_size": size,
            "prompt_cache_bytes": _MIN_CACHE_BYTES,
        }
    gib = total / (1024**3)
    if gib >= 64:
        size = 32
    elif gib >= 36:
        size = 24
    elif gib >= 24:
        size = 16
    bytes_budget = int(total * _CACHE_BYTES_FRACTION)
    bytes_budget = max(_MIN_CACHE_BYTES, min(_MAX_CACHE_BYTES, bytes_budget))
    return {"prompt_cache_size": size, "prompt_cache_bytes": bytes_budget}


def resolve_prompt_cache_cli_flags(advanced: dict[str, Any] | None) -> dict[str, Any]:
    """Merge explicit advanced knobs with RAM-aware defaults for TEXT launch."""
    advanced = advanced or {}
    defaults = default_prompt_cache_settings()
    size = advanced.get("prompt_cache_size", defaults["prompt_cache_size"])
    cache_bytes = advanced.get("prompt_cache_bytes", defaults["prompt_cache_bytes"])
    flags: dict[str, Any] = {}
    if size is not None:
        flags["prompt-cache-size"] = int(size)
    if cache_bytes is not None:
        flags["prompt-cache-bytes"] = int(cache_bytes)
    return flags


def _darwin_total_bytes() -> int | None:
    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
        total = int(raw)
    except (OSError, ValueError, subprocess.CalledProcessError):
        logger.debug("Unable to read Darwin hw.memsize", exc_info=True)
        return None
    return total if total > 0 else None


def _linux_total_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            meminfo = handle.read()
    except OSError:
        return None
    for line in meminfo.splitlines():
        if not line.startswith("MemTotal:"):
            continue
        digits = "".join(ch for ch in line if ch.isdigit())
        if digits:
            return int(digits) * 1024
    return None
