"""Per-request latency collector for the Nadir Gateway (TTFT / prefill / cache)."""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from orchestrator.gateway.protocol.chat_sse import parse_chat_sse_line

logger = logging.getLogger(__name__)

_active_collector: ContextVar["RequestTimingCollector | None"] = ContextVar(
    "nadir_gateway_request_timing",
    default=None,
)


def telemetry_enabled() -> bool:
    from django.conf import settings

    from orchestrator.env_utils import env_bool

    return env_bool(
        "NADIR_GATEWAY_TELEMETRY_ENABLED",
        bool(getattr(settings, "NADIR_GATEWAY_TELEMETRY_ENABLED", True)),
    )


def telemetry_retain_count() -> int:
    from django.conf import settings

    from orchestrator.env_utils import env_int

    return max(
        1,
        env_int(
            "NADIR_GATEWAY_TELEMETRY_RETAIN",
            int(getattr(settings, "NADIR_GATEWAY_TELEMETRY_RETAIN", 100)),
        ),
    )


@dataclass
class RequestTimingSnapshot:
    """Immutable timing record ready for logging / persistence."""

    request_id: str
    endpoint: str
    alias: str
    instance_id: int | None
    stream: bool
    started_at: datetime
    queue_wait_ms: float | None = None
    wake_ms: float | None = None
    cold_start: bool = False
    prefill_ms: float | None = None
    ttft_ms: float | None = None
    visible_ttft_ms: float | None = None
    generation_ms: float | None = None
    total_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    prefill_tokens_per_s: float | None = None
    generation_tokens_per_s: float | None = None
    error_code: str | None = None

    def as_log_fields(self) -> dict[str, Any]:
        """Stable structured fields for logs / future Prometheus."""
        return {
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "alias": self.alias,
            "instance_id": self.instance_id,
            "stream": self.stream,
            "queue_wait_ms": self.queue_wait_ms,
            "wake_ms": self.wake_ms,
            "cold_start": self.cold_start,
            "prefill_ms": self.prefill_ms,
            "ttft_ms": self.ttft_ms,
            "visible_ttft_ms": self.visible_ttft_ms,
            "generation_ms": self.generation_ms,
            "total_ms": self.total_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "prefill_tokens_per_s": self.prefill_tokens_per_s,
            "generation_tokens_per_s": self.generation_tokens_per_s,
            "error_code": self.error_code,
        }


@dataclass
class RequestTimingCollector:
    """Mutable per-request timing state (one gateway call)."""

    endpoint: str
    stream: bool
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    alias: str = ""
    instance_id: int | None = None
    started_mono: float = field(default_factory=time.perf_counter)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    queue_enter_mono: float | None = None
    queue_acquired_mono: float | None = None
    wake_ms: float | None = None
    cold_start: bool = False
    upstream_open_mono: float | None = None
    first_any_token_mono: float | None = None
    first_visible_token_mono: float | None = None
    stream_end_mono: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    error_code: str | None = None
    _finalized: bool = False

    def mark_queue_enter(self) -> None:
        if self.queue_enter_mono is None:
            self.queue_enter_mono = time.perf_counter()

    def mark_queue_acquired(self) -> None:
        if self.queue_acquired_mono is None:
            self.queue_acquired_mono = time.perf_counter()

    def mark_wake(self, wake_ms: float, *, cold_start: bool) -> None:
        self.wake_ms = max(0.0, wake_ms)
        self.cold_start = cold_start

    def bind_target(self, *, alias: str, instance_id: int) -> None:
        self.alias = alias
        self.instance_id = instance_id

    def mark_upstream_open(self) -> None:
        if self.upstream_open_mono is None:
            self.upstream_open_mono = time.perf_counter()

    def observe_sse_line(self, line: bytes) -> None:
        """Update TTFT / usage milestones from one restored SSE line."""
        event = parse_chat_sse_line(line)
        if event is None:
            return
        if event.usage:
            self.apply_usage(event.usage)
        now = time.perf_counter()
        has_any = bool(event.text or event.reasoning or event.tool_call_deltas)
        if has_any and self.first_any_token_mono is None:
            self.first_any_token_mono = now
        if event.text and self.first_visible_token_mono is None:
            self.first_visible_token_mono = now
        if event.done or event.finish_reason:
            self.stream_end_mono = now

    def apply_usage(self, usage: dict[str, Any]) -> None:
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if isinstance(prompt, int):
            self.prompt_tokens = prompt
        if isinstance(completion, int):
            self.completion_tokens = completion
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
            if isinstance(cached, int):
                self.cached_prompt_tokens = cached

    def mark_json_complete(self, payload: Any) -> None:
        """Non-stream: treat response arrival as TTFT and apply usage."""
        now = time.perf_counter()
        if self.upstream_open_mono is None:
            self.upstream_open_mono = now
        if self.first_any_token_mono is None:
            self.first_any_token_mono = now
        if self.first_visible_token_mono is None:
            self.first_visible_token_mono = now
        self.stream_end_mono = now
        if isinstance(payload, dict):
            usage = payload.get("usage")
            if isinstance(usage, dict):
                self.apply_usage(usage)

    def set_error(self, code: str) -> None:
        self.error_code = code

    def snapshot(self) -> RequestTimingSnapshot:
        end = self.stream_end_mono or time.perf_counter()
        queue_ms = _delta_ms(self.queue_enter_mono, self.queue_acquired_mono)
        prefill_ms = _delta_ms(self.upstream_open_mono, self.first_any_token_mono)
        ttft_ms = _delta_ms(self.started_mono, self.first_any_token_mono)
        visible_ttft_ms = _delta_ms(self.started_mono, self.first_visible_token_mono)
        generation_ms = _delta_ms(self.first_any_token_mono, end)
        total_ms = _delta_ms(self.started_mono, end)
        prefill_tps = _rate(self.prompt_tokens, prefill_ms)
        generation_tps = _rate(self.completion_tokens, generation_ms)
        return RequestTimingSnapshot(
            request_id=self.request_id,
            endpoint=self.endpoint,
            alias=self.alias,
            instance_id=self.instance_id,
            stream=self.stream,
            started_at=self.started_at,
            queue_wait_ms=queue_ms,
            wake_ms=self.wake_ms,
            cold_start=self.cold_start,
            prefill_ms=prefill_ms,
            ttft_ms=ttft_ms,
            visible_ttft_ms=visible_ttft_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens,
            prefill_tokens_per_s=prefill_tps,
            generation_tokens_per_s=generation_tps,
            error_code=self.error_code,
        )

    def finalize(self) -> RequestTimingSnapshot | None:
        if self._finalized:
            return None
        self._finalized = True
        snap = self.snapshot()
        logger.info("gateway_request_timing %s", snap.as_log_fields())
        try:
            from orchestrator.gateway_telemetry_services import record_gateway_request_timing

            record_gateway_request_timing(snap)
        except Exception:
            logger.warning("Failed to persist gateway request timing", exc_info=True)
        return snap


def start_request_timing(*, endpoint: str, stream: bool) -> RequestTimingCollector | None:
    """Begin collecting timing for the current context; no-op when disabled."""
    if not telemetry_enabled():
        return None
    collector = RequestTimingCollector(endpoint=endpoint, stream=stream)
    collector._token = _active_collector.set(collector)  # type: ignore[attr-defined]
    return collector


def get_active_collector() -> RequestTimingCollector | None:
    return _active_collector.get()


def reset_request_timing(collector: RequestTimingCollector | None) -> None:
    if collector is None:
        return
    token: Token | None = getattr(collector, "_token", None)
    if token is not None:
        try:
            _active_collector.reset(token)
        except ValueError:
            # SSE finalizes in a different Context than start_request_timing.
            _active_collector.set(None)
    collector.finalize()


def _delta_ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return max(0.0, (end - start) * 1000.0)


def _rate(tokens: int | None, duration_ms: float | None) -> float | None:
    if tokens is None or duration_ms is None or duration_ms <= 0:
        return None
    return tokens / (duration_ms / 1000.0)
