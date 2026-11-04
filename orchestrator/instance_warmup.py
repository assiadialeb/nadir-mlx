"""Post-load warmup so Metal shader compile is paid before the first user request (MLX-126)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from django.utils import timezone

from orchestrator.instance_health import probe_generation_health
from orchestrator.lifecycle_selectors import warmup_on_load_enabled

if TYPE_CHECKING:
    from orchestrator.models import InferenceInstance

logger = logging.getLogger(__name__)


def _record_warmup_activity(instance: InferenceInstance) -> None:
    """A successful warmup is real use: keep idle offload from firing right after load."""
    from orchestrator.models import InferenceInstance as InstanceModel

    now = timezone.now()
    InstanceModel.objects.filter(pk=instance.pk).update(last_used_at=now)
    instance.last_used_at = now


def maybe_run_load_warmup(instance: InferenceInstance) -> float | None:
    """Run a minimal generation after /health is up, before the row turns RUNNING.

    Called once per launch from ``server_manager`` while the slot lock is held,
    so no other process can warm the same instance concurrently.

    Returns elapsed milliseconds when a warmup ran, otherwise ``None``.
    Failures are logged and never raise — readiness already succeeded.
    """
    if not warmup_on_load_enabled(instance.server_config):
        return None

    started = time.perf_counter()
    try:
        success = probe_generation_health(instance)
    except Exception:
        logger.exception(
            "Warmup raised for %s on port %s",
            instance.model_name,
            instance.port,
        )
        return (time.perf_counter() - started) * 1000.0

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not success:
        logger.warning(
            "Warmup generation failed for %s on port %s after %.0f ms",
            instance.model_name,
            instance.port,
            elapsed_ms,
        )
        return elapsed_ms
    _record_warmup_activity(instance)
    logger.info(
        "Warmup completed for %s on port %s in %.0f ms",
        instance.model_name,
        instance.port,
        elapsed_ms,
    )
    return elapsed_ms
