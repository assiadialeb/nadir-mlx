"""Default mlx-lm prompt-cache sizing from host unified memory (MLX-125)."""

from __future__ import annotations

import logging
import platform
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_SIZE = 10


def host_total_bytes() -> int | None:
    """Return total unified/system RAM in bytes, or None if unknown."""
    if platform.system() == "Darwin":
        return _darwin_total_bytes()
    return _linux_total_bytes()


def default_prompt_cache_settings(
    total_bytes: int | None = None,
) -> dict[str, int]:
    """Return RAM-aware ``prompt_cache_size`` defaults (no byte cap).

    mlx-lm defaults to ``--prompt-cache-size 10`` with **no** ``--prompt-cache-bytes``
    ceiling (effectively unlimited). Nadir only sets the entry count by default so
    a single large agent prefix is not immediately LRU-evicted. Operators may still
    set ``prompt_cache_bytes`` explicitly in Advanced JSON when they want a hard cap.
    """
    total = total_bytes if total_bytes is not None else host_total_bytes()
    size = _DEFAULT_CACHE_SIZE
    if total is not None and total > 0:
        gib = total / (1024**3)
        if gib >= 64:
            size = 32
        elif gib >= 36:
            size = 24
        elif gib >= 24:
            size = 16
    return {"prompt_cache_size": size}


def resolve_prompt_cache_cli_flags(advanced: dict[str, Any] | None) -> dict[str, Any]:
    """Merge explicit advanced knobs with RAM-aware size defaults for TEXT launch.

    ``--prompt-cache-bytes`` is emitted only when the operator set
    ``prompt_cache_bytes`` explicitly — never from a RAM-fraction default.
    """
    advanced = advanced or {}
    defaults = default_prompt_cache_settings()
    size = advanced.get("prompt_cache_size", defaults["prompt_cache_size"])
    flags: dict[str, Any] = {}
    if size is not None:
        flags["prompt-cache-size"] = int(size)
    if "prompt_cache_bytes" in advanced and advanced["prompt_cache_bytes"] is not None:
        flags["prompt-cache-bytes"] = int(advanced["prompt_cache_bytes"])
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
