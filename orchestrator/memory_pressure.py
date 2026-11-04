"""Host memory pressure helpers for LRU eviction (MLX-112)."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from typing import Any

from django.conf import settings

from orchestrator.model_utils import get_model_folder_size_bytes

logger = logging.getLogger(__name__)

# Weights on disk underestimate resident MLX footprint; apply a conservative factor.
_DISK_TO_RUNTIME_FACTOR = 1.15
_DEFAULT_FOOTPRINT_BYTES = 2 * 1024**3


def lru_eviction_enabled() -> bool:
    raw = os.environ.get("NADIR_LRU_EVICTION_ENABLED")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(getattr(settings, "NADIR_LRU_EVICTION_ENABLED", True))


def memory_reserve_bytes() -> int:
    raw = os.environ.get("NADIR_MEMORY_RESERVE_GB")
    if raw:
        try:
            return max(0, int(float(raw) * 1024**3))
        except ValueError:
            pass
    gb = float(getattr(settings, "NADIR_MEMORY_RESERVE_GB", 4.0))
    return max(0, int(gb * 1024**3))


def estimate_instance_footprint_bytes(
    model_name: str,
    server_config: dict[str, Any] | None = None,
) -> int:
    """Estimate resident bytes for a model (disk size × factor, with floor)."""
    del server_config  # reserved for future quant-aware sizing
    try:
        disk = int(get_model_folder_size_bytes(model_name) or 0)
    except Exception:
        disk = 0
    if disk <= 0:
        return _DEFAULT_FOOTPRINT_BYTES
    return max(_DEFAULT_FOOTPRINT_BYTES, int(disk * _DISK_TO_RUNTIME_FACTOR))


def host_available_bytes() -> int | None:
    """Return approximate available unified/system memory, or None if unknown."""
    if platform.system() == "Darwin":
        return _darwin_available_bytes()
    return _linux_available_bytes()


def _darwin_available_bytes() -> int | None:
    try:
        page_size = int(
            subprocess.check_output(["pagesize"], text=True, encoding="utf-8").strip()
        )
        vm_stat = subprocess.check_output(
            ["vm_stat"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
    except (OSError, ValueError, subprocess.CalledProcessError):
        logger.debug("Unable to read Darwin vm_stat for memory pressure", exc_info=True)
        return None

    free_pages = 0
    for line in vm_stat.splitlines():
        key, _, value = line.partition(":")
        if key.strip() not in {"Pages free", "Pages inactive", "Pages speculative"}:
            continue
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            free_pages += int(digits)
    if free_pages <= 0:
        return None
    available = free_pages * page_size
    reserve = memory_reserve_bytes()
    return max(0, available - reserve)


def _linux_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            meminfo = handle.read()
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in meminfo.splitlines():
        key, _, rest = line.partition(":")
        digits = "".join(ch for ch in rest if ch.isdigit())
        if digits:
            values[key] = int(digits) * 1024
    available = values.get("MemAvailable") or values.get("MemFree")
    if available is None:
        return None
    return max(0, available - memory_reserve_bytes())
