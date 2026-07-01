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
