"""Persist and prune gateway per-request timing rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.db import transaction

if TYPE_CHECKING:
    from orchestrator.gateway.request_timing import RequestTimingSnapshot


def record_gateway_request_timing(snapshot: RequestTimingSnapshot) -> None:
    """Insert one timing row and prune older rows beyond the retain window."""
    from orchestrator.gateway.request_timing import telemetry_retain_count
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
