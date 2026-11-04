"""Post-load warmup so Metal shader compile is paid before the first user request (MLX-126)."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from orchestrator.instance_health import probe_generation_health, probe_http_health
from orchestrator.lifecycle_selectors import warmup_on_load_enabled

if TYPE_CHECKING:
    from orchestrator.models import InferenceInstance

logger = logging.getLogger(__name__)

_warmup_lock = threading.Lock()
_warmup_in_flight: set[int] = set()
_warmup_done_ids: set[int] = set()


def clear_warmup_state_for_tests() -> None:
    """Reset in-process warmup tracking (unit tests only)."""
    with _warmup_lock:
        _warmup_in_flight.clear()
        _warmup_done_ids.clear()


def mark_instance_needs_warmup(instance_id: int) -> None:
    """Forget a prior warmup so the next ready transition runs again (after stop/start)."""
    with _warmup_lock:
        _warmup_done_ids.discard(instance_id)
        _warmup_in_flight.discard(instance_id)


def _claim_warmup(instance_id: int) -> bool:
    with _warmup_lock:
        if instance_id in _warmup_done_ids or instance_id in _warmup_in_flight:
            return False
        _warmup_in_flight.add(instance_id)
        return True


def _finish_warmup(instance_id: int, *, success: bool) -> None:
    with _warmup_lock:
        _warmup_in_flight.discard(instance_id)
        if success:
            _warmup_done_ids.add(instance_id)


def maybe_run_load_warmup(instance: InferenceInstance) -> float | None:
    """Run a minimal generation after load when ``ops.warmup_on_load`` is enabled.

    Returns elapsed milliseconds when a warmup ran, otherwise ``None``.
    Failures are logged and never raise — readiness already succeeded.
    """
    if not warmup_on_load_enabled(instance.server_config):
        return None
    if instance.status != "RUNNING":
        return None
    if not _claim_warmup(instance.pk):
        return None

    started = time.perf_counter()
    success = False
    try:
        if not probe_http_health(instance):
            logger.info(
                "Skipping warmup for %s on port %s: /health not ready yet",
                instance.model_name,
                instance.port,
            )
            return None
        success = probe_generation_health(instance)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if success:
            logger.info(
                "Warmup completed for %s on port %s in %.0f ms",
                instance.model_name,
                instance.port,
                elapsed_ms,
            )
            return elapsed_ms
        logger.warning(
            "Warmup generation failed for %s on port %s after %.0f ms",
            instance.model_name,
            instance.port,
            elapsed_ms,
        )
        return elapsed_ms
    except Exception:
        logger.exception(
            "Warmup raised for %s on port %s",
            instance.model_name,
            instance.port,
        )
        return (time.perf_counter() - started) * 1000.0
    finally:
        _finish_warmup(instance.pk, success=success)
