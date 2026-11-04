"""Persist and prune gateway per-request timing rows."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import TYPE_CHECKING

from django.db import transaction

if TYPE_CHECKING:
    from orchestrator.gateway.request_timing import RequestTimingSnapshot

logger = logging.getLogger(__name__)

# Prune every N inserts so retain cleanup is not on the hot path each request.
_PRUNE_EVERY_N = 10
_inserts_since_prune = 0
_inserts_lock = threading.Lock()

_write_queue: queue.Queue = queue.Queue()
_writer_started = False
_writer_lock = threading.Lock()
_FLUSH_SENTINEL = object()


def schedule_gateway_request_timing(snapshot: RequestTimingSnapshot) -> None:
    """Persist timing without blocking the ASGI event loop.

    Sync contexts (unit tests, non-async callers) write immediately. Async
    contexts enqueue to a daemon writer thread to avoid
    ``SynchronousOnlyOperation`` under Django ASGI.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        record_gateway_request_timing(snapshot)
        return
    enqueue_gateway_request_timing(snapshot)


def enqueue_gateway_request_timing(snapshot: RequestTimingSnapshot) -> None:
    """Queue a timing row for the background writer (async-safe)."""
    _ensure_writer()
    _write_queue.put(snapshot)


def flush_gateway_timing_writes(timeout: float = 5.0) -> None:
    """Block until the telemetry writer queue is drained (tests / shutdown)."""
    _ensure_writer()
    finished = threading.Event()
    _write_queue.put((_FLUSH_SENTINEL, finished))
    if not finished.wait(timeout=timeout):
        logger.warning("Timed out flushing gateway telemetry write queue")


def record_gateway_request_timing(snapshot: RequestTimingSnapshot) -> None:
    """Insert one timing row; prune excess rows every ``_PRUNE_EVERY_N`` inserts."""
    from orchestrator.models import GatewayRequestMetric

    create_kwargs: dict = {
        "request_id": snapshot.request_id,
        "endpoint": snapshot.endpoint[:128],
        "alias": snapshot.alias[:255],
        "stream": snapshot.stream,
        "started_at": snapshot.started_at,
        "queue_wait_ms": snapshot.queue_wait_ms,
        "wake_ms": snapshot.wake_ms,
        "cold_start": snapshot.cold_start,
        "prefill_ms": snapshot.prefill_ms,
        "ttft_ms": snapshot.ttft_ms,
        "visible_ttft_ms": snapshot.visible_ttft_ms,
        "generation_ms": snapshot.generation_ms,
        "total_ms": snapshot.total_ms,
        "prompt_tokens": snapshot.prompt_tokens,
        "completion_tokens": snapshot.completion_tokens,
        "cached_prompt_tokens": snapshot.cached_prompt_tokens,
        "prefill_tokens_per_s": snapshot.prefill_tokens_per_s,
        "generation_tokens_per_s": snapshot.generation_tokens_per_s,
        "error_code": (snapshot.error_code or "")[:64],
    }
    if snapshot.instance_id is not None:
        create_kwargs["instance_id"] = snapshot.instance_id
    GatewayRequestMetric.objects.create(**create_kwargs)
    should_prune = False
    global _inserts_since_prune
    with _inserts_lock:
        _inserts_since_prune += 1
        if _inserts_since_prune >= _PRUNE_EVERY_N:
            _inserts_since_prune = 0
            should_prune = True
    if should_prune:
        prune_gateway_timing_rows()


def prune_gateway_timing_rows() -> None:
    """Delete timing rows beyond the configured retain window."""
    from orchestrator.gateway.request_timing import telemetry_retain_count
    from orchestrator.models import GatewayRequestMetric

    retain = telemetry_retain_count()
    with transaction.atomic():
        excess_ids = list(
            GatewayRequestMetric.objects.order_by("-started_at", "-id")
            .values_list("id", flat=True)[retain:]
        )
        if excess_ids:
            GatewayRequestMetric.objects.filter(id__in=excess_ids).delete()


def list_recent_gateway_timings(*, limit: int = 25, instance_id: int | None = None):
    """Return recent timing rows for the UI."""
    from orchestrator.models import GatewayRequestMetric

    queryset = GatewayRequestMetric.objects.all()
    if instance_id is not None:
        queryset = queryset.filter(instance_id=instance_id)
    return list(queryset.order_by("-started_at", "-id")[: max(1, limit)])


def prompt_cache_hit_rate(
    rows: list | None = None,
    *,
    limit: int = 25,
    instance_id: int | None = None,
) -> float | None:
    """Return cached_prompt_tokens / prompt_tokens over recent rows, or None."""
    if rows is None:
        rows = list_recent_gateway_timings(limit=limit, instance_id=instance_id)
    prompt_total = 0
    cached_total = 0
    for row in rows:
        prompt = getattr(row, "prompt_tokens", None)
        cached = getattr(row, "cached_prompt_tokens", None)
        if not isinstance(prompt, int) or prompt <= 0:
            continue
        if not isinstance(cached, int) or cached < 0:
            continue
        prompt_total += prompt
        cached_total += min(cached, prompt)
    if prompt_total <= 0:
        return None
    return cached_total / prompt_total


def _ensure_writer() -> None:
    global _writer_started
    with _writer_lock:
        if _writer_started:
            return
        thread = threading.Thread(
            target=_writer_loop,
            name="nadir-gateway-telemetry",
            daemon=True,
        )
        thread.start()
        _writer_started = True


def _writer_loop() -> None:
    while True:
        item = _write_queue.get()
        try:
            if isinstance(item, tuple) and item and item[0] is _FLUSH_SENTINEL:
                item[1].set()
                continue
            # Long-lived daemon thread: refresh Django DB connections after
            # Postgres restarts so inserts do not fail until gateway restart.
            from django.db import close_old_connections

            close_old_connections()
            record_gateway_request_timing(item)  # type: ignore[arg-type]
        except Exception:
            logger.warning("Failed to persist gateway request timing", exc_info=True)
        finally:
            _write_queue.task_done()
