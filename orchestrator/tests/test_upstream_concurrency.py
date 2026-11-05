"""Tests for gateway upstream concurrency limiting."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH, GatewayTarget
from orchestrator.gateway.upstream_concurrency import (
    UpstreamQueueTimeoutError,
    default_max_concurrent_upstream,
    global_max_concurrent_upstream,
    reset_upstream_coordinator_for_tests,
    resolve_max_concurrent_upstream,
    upstream_concurrency_slot,
)

TARGET = GatewayTarget(
    alias="qwen",
    instance_id=42,
    launch_mode="MULTIMODAL",
    host="127.0.0.1",
    port=11475,
    upstream_model="default_model",
    api_path=CHAT_COMPLETIONS_PATH,
    max_concurrent_upstream=None,
)


class UpstreamConcurrencyTests(SimpleTestCase):
    @override_settings(NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM=16)
    def test_default_limit_reads_settings_when_env_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM", None)
            self.assertEqual(default_max_concurrent_upstream(), 16)

    def test_zero_env_disables_global_limit(self) -> None:
        with patch.dict(os.environ, {"NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM": "0"}, clear=False):
            self.assertIsNone(default_max_concurrent_upstream())

    def test_instance_override_takes_priority(self) -> None:
        limited = GatewayTarget(
            alias="qwen",
            instance_id=7,
            launch_mode="TEXT",
            host="127.0.0.1",
            port=11400,
            upstream_model="default_model",
            api_path=CHAT_COMPLETIONS_PATH,
            max_concurrent_upstream=8,
        )
        self.assertEqual(resolve_max_concurrent_upstream(limited), 8)

    def test_instance_zero_disables_limit(self) -> None:
        unlimited = GatewayTarget(
            alias="qwen",
            instance_id=8,
            launch_mode="TEXT",
            host="127.0.0.1",
            port=11400,
            upstream_model="default_model",
            api_path=CHAT_COMPLETIONS_PATH,
            max_concurrent_upstream=0,
        )
        self.assertIsNone(resolve_max_concurrent_upstream(unlimited))

    def test_semaphore_limits_parallel_slots(self) -> None:
        limited_target = GatewayTarget(
            alias="qwen",
            instance_id=99,
            launch_mode="TEXT",
            host="127.0.0.1",
            port=11400,
            upstream_model="default_model",
            api_path=CHAT_COMPLETIONS_PATH,
            max_concurrent_upstream=1,
        )
        entered = 0
        max_seen = 0
        lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal entered, max_seen
            async with upstream_concurrency_slot(limited_target):
                async with lock:
                    entered += 1
                    max_seen = max(max_seen, entered)
                await asyncio.sleep(0.05)
                async with lock:
                    entered -= 1

        async def run() -> None:
            await asyncio.gather(*(worker() for _ in range(4)))

        asyncio.run(run())
        self.assertEqual(max_seen, 1)

    def test_queue_timeout_raises_when_slots_never_free(self) -> None:
        limited_target = GatewayTarget(
            alias="qwen",
            instance_id=100,
            launch_mode="TEXT",
            host="127.0.0.1",
            port=11400,
            upstream_model="default_model",
            api_path=CHAT_COMPLETIONS_PATH,
            max_concurrent_upstream=1,
        )

        async def hold_slot() -> None:
            async with upstream_concurrency_slot(limited_target):
                await asyncio.sleep(0.2)

        async def wait_forever() -> None:
            task = asyncio.create_task(hold_slot())
            await asyncio.sleep(0.01)
            with patch(
                "orchestrator.gateway.upstream_concurrency.queue_timeout_seconds",
                return_value=0.05,
            ):
                with self.assertRaises(UpstreamQueueTimeoutError):
                    async with upstream_concurrency_slot(limited_target):
                        await asyncio.sleep(0)
            await task

        asyncio.run(wait_forever())


def _text_target(instance_id: int, **overrides: object) -> GatewayTarget:
    fields: dict[str, object] = {
        "alias": f"model-{instance_id}",
        "instance_id": instance_id,
        "launch_mode": "TEXT",
        "host": "127.0.0.1",
        "port": 11400 + instance_id % 100,
        "upstream_model": "default_model",
        "api_path": CHAT_COMPLETIONS_PATH,
        "max_concurrent_upstream": 0,
    }
    fields.update(overrides)
    return GatewayTarget(**fields)


async def _measure_peak(targets: list[GatewayTarget]) -> int:
    active = 0
    peak = 0

    async def worker(target: GatewayTarget) -> None:
        nonlocal active, peak
        async with upstream_concurrency_slot(target):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1

    await asyncio.gather(*(worker(target) for target in targets))
    return peak


class GlobalUpstreamConcurrencyTests(SimpleTestCase):
    def setUp(self) -> None:
        reset_upstream_coordinator_for_tests()

    def test_global_cap_limits_requests_across_instances(self) -> None:
        targets = [_text_target(501), _text_target(502), _text_target(503)] * 2
        with patch.dict(os.environ, {"NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL": "2"}, clear=False):
            peak = asyncio.run(_measure_peak(targets))
        self.assertEqual(peak, 2)

    def test_zero_global_cap_is_unlimited(self) -> None:
        targets = [_text_target(511), _text_target(512), _text_target(513)]
        with patch.dict(os.environ, {"NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL": "0"}, clear=False):
            self.assertIsNone(global_max_concurrent_upstream())
            peak = asyncio.run(_measure_peak(targets))
        self.assertEqual(peak, 3)

    def test_decode_concurrency_bounds_instance_cap(self) -> None:
        bounded = _text_target(521, max_concurrent_upstream=16, decode_concurrency=4)
        unlimited = _text_target(522, max_concurrent_upstream=0, decode_concurrency=6)
        self.assertEqual(resolve_max_concurrent_upstream(bounded), 4)
        self.assertEqual(resolve_max_concurrent_upstream(unlimited), 6)

    def test_batch_request_waits_behind_interactive(self) -> None:
        from orchestrator.gateway.gpu_scheduler import (
            PRIORITY_BATCH,
            PRIORITY_INTERACTIVE,
            set_request_priority,
        )

        target = _text_target(531, max_concurrent_upstream=1)
        order: list[str] = []

        async def request(priority: str, label: str) -> None:
            set_request_priority(priority)
            async with upstream_concurrency_slot(target):
                order.append(label)
                await asyncio.sleep(0.01)

        async def run() -> None:
            holder = asyncio.create_task(request(PRIORITY_INTERACTIVE, "first"))
            await asyncio.sleep(0)
            batch = asyncio.create_task(request(PRIORITY_BATCH, "batch"))
            await asyncio.sleep(0)
            interactive = asyncio.create_task(request(PRIORITY_INTERACTIVE, "interactive"))
            await asyncio.gather(holder, batch, interactive)

        with patch.dict(
            os.environ,
            {
                "NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL": "0",
                "NADIR_GATEWAY_BATCH_MAX_SHARE": "1",
            },
            clear=False,
        ):
            asyncio.run(run())
        self.assertEqual(order, ["first", "interactive", "batch"])

    def test_full_instance_queue_does_not_block_other_instance(self) -> None:
        from orchestrator.gateway.gpu_scheduler import PRIORITY_INTERACTIVE, set_request_priority

        target_a = _text_target(601, max_concurrent_upstream=1)
        target_b = _text_target(602, max_concurrent_upstream=1)
        order: list[str] = []

        async def work(target: GatewayTarget, label: str, hold: float) -> None:
            set_request_priority(PRIORITY_INTERACTIVE)
            async with upstream_concurrency_slot(target):
                order.append(f"{label}-start")
                await asyncio.sleep(hold)
                order.append(f"{label}-end")

        async def run() -> None:
            with patch.dict(
                os.environ,
                {
                    "NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL": "2",
                    "NADIR_GATEWAY_BATCH_MAX_SHARE": "1",
                },
                clear=False,
            ):
                reset_upstream_coordinator_for_tests()
                await asyncio.gather(
                    work(target_a, "A1", 0.06),
                    work(target_a, "A2", 0.01),
                    work(target_a, "A3", 0.01),
                    work(target_b, "B1", 0.01),
                )

        asyncio.run(run())
        self.assertIn("B1-start", order)
        self.assertLess(order.index("B1-start"), order.index("A2-start"))
