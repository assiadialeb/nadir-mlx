"""Unit tests for gateway per-request timing collector (MLX-123)."""

from __future__ import annotations

import json
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase, override_settings

from orchestrator.gateway.request_timing import (
    RequestTimingCollector,
    reset_request_timing,
    start_request_timing,
)
from orchestrator.gateway_telemetry_services import (
    flush_gateway_timing_writes,
    list_recent_gateway_timings,
    record_gateway_request_timing,
)
from orchestrator.models import GatewayRequestMetric


def _sse_data(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n"


class RequestTimingCollectorTests(SimpleTestCase):
    def test_visible_ttft_ignores_reasoning_then_tracks_text(self) -> None:
        collector = RequestTimingCollector(endpoint="/v1/chat/completions", stream=True)
        collector.mark_upstream_open()
        collector.observe_sse_line(
            _sse_data(
                {
                    "choices": [
                        {"delta": {"reasoning_content": "thinking…"}, "index": 0}
                    ]
                }
            )
        )
        self.assertIsNotNone(collector.first_any_token_mono)
        self.assertIsNone(collector.first_visible_token_mono)
        collector.observe_sse_line(
            _sse_data({"choices": [{"delta": {"content": "Hello"}, "index": 0}]})
        )
        self.assertIsNotNone(collector.first_visible_token_mono)
        snap = collector.snapshot()
        self.assertIsNotNone(snap.ttft_ms)
        self.assertIsNotNone(snap.visible_ttft_ms)
        self.assertGreaterEqual(snap.visible_ttft_ms or 0, snap.ttft_ms or 0)

    def test_usage_extracts_cached_tokens(self) -> None:
        collector = RequestTimingCollector(endpoint="/v1/chat/completions", stream=True)
        collector.apply_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 80},
            }
        )
        snap = collector.snapshot()
        self.assertEqual(snap.prompt_tokens, 100)
        self.assertEqual(snap.cached_prompt_tokens, 80)
        self.assertEqual(snap.completion_tokens, 20)

    def test_log_fields_are_stable_keys(self) -> None:
        collector = RequestTimingCollector(endpoint="/v1/chat/completions", stream=False)
        fields = collector.snapshot().as_log_fields()
        for key in (
            "request_id",
            "queue_wait_ms",
            "wake_ms",
            "prefill_ms",
            "ttft_ms",
            "visible_ttft_ms",
            "cached_prompt_tokens",
            "generation_tokens_per_s",
        ):
            self.assertIn(key, fields)


class RequestTimingPersistenceTests(TestCase):
    @override_settings(NADIR_GATEWAY_TELEMETRY_RETAIN=2)
    def test_record_prunes_to_retain_window(self) -> None:
        from orchestrator.gateway_telemetry_services import prune_gateway_timing_rows

        with patch.dict("os.environ", {"NADIR_GATEWAY_TELEMETRY_RETAIN": "2"}):
            for index in range(4):
                collector = RequestTimingCollector(
                    endpoint="/v1/chat/completions",
                    stream=False,
                    alias=f"m{index}",
                )
                collector.mark_json_complete(
                    {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}
                )
                record_gateway_request_timing(collector.snapshot())
            # Inserts prune every N rows; force retain window for this assertion.
            prune_gateway_timing_rows()
        self.assertEqual(GatewayRequestMetric.objects.count(), 2)
        recent = list_recent_gateway_timings(limit=10)
        self.assertEqual(len(recent), 2)

    @override_settings(NADIR_GATEWAY_TELEMETRY_RETAIN=2)
    def test_record_prunes_automatically_every_n_inserts(self) -> None:
        import orchestrator.gateway_telemetry_services as telemetry

        with patch.object(telemetry, "_PRUNE_EVERY_N", 2):
            with patch.object(telemetry, "_inserts_since_prune", 0):
                with patch.dict("os.environ", {"NADIR_GATEWAY_TELEMETRY_RETAIN": "2"}):
                    for index in range(4):
                        collector = RequestTimingCollector(
                            endpoint="/v1/chat/completions",
                            stream=False,
                            alias=f"auto{index}",
                        )
                        collector.mark_json_complete(
                            {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}
                        )
                        record_gateway_request_timing(collector.snapshot())
        self.assertEqual(GatewayRequestMetric.objects.count(), 2)


class RequestTimingContextTests(SimpleTestCase):
    @override_settings(NADIR_GATEWAY_TELEMETRY_ENABLED=True)
    def test_start_and_reset_collector(self) -> None:
        with patch(
            "orchestrator.gateway_telemetry_services.record_gateway_request_timing"
        ) as mock_record:
            collector = start_request_timing(
                endpoint="/v1/chat/completions", stream=False
            )
            self.assertIsNotNone(collector)
            assert collector is not None
            collector.mark_json_complete({"usage": {"prompt_tokens": 3}})
            reset_request_timing(collector)
            mock_record.assert_called_once()

    @override_settings(NADIR_GATEWAY_TELEMETRY_ENABLED=True)
    def test_reset_survives_foreign_contextvar_token(self) -> None:
        """SSE may finalize in a different Context than start (Starlette stream)."""
        import contextvars

        with patch(
            "orchestrator.gateway_telemetry_services.record_gateway_request_timing"
        ) as mock_record:
            collector = start_request_timing(
                endpoint="/v1/chat/completions", stream=True
            )
            assert collector is not None

            def _finalize_elsewhere() -> None:
                reset_request_timing(collector)

            contextvars.copy_context().run(_finalize_elsewhere)
            mock_record.assert_called_once()
            # Idempotent: second finalize is a no-op.
            reset_request_timing(collector)
            mock_record.assert_called_once()


class RequestTimingAsyncPersistTests(IsolatedAsyncioTestCase, SimpleTestCase):
    """T1: finalize from an async context must not hit sync ORM on the event loop."""

    @override_settings(NADIR_GATEWAY_TELEMETRY_ENABLED=True)
    async def test_finalize_from_async_persists_via_writer_queue(self) -> None:
        with patch(
            "orchestrator.gateway_telemetry_services.record_gateway_request_timing"
        ) as mock_record:
            collector = start_request_timing(
                endpoint="/v1/chat/completions", stream=True
            )
            assert collector is not None
            collector.mark_json_complete(
                {"usage": {"prompt_tokens": 10, "completion_tokens": 2}}
            )
            reset_request_timing(collector)
            flush_gateway_timing_writes()
            mock_record.assert_called_once()
            snap = mock_record.call_args.args[0]
            self.assertEqual(snap.prompt_tokens, 10)
