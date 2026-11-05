"""Tests for the priority-aware upstream slot scheduler (MLX-128)."""

from __future__ import annotations

import asyncio

from django.test import SimpleTestCase

from orchestrator.gateway.gpu_scheduler import (
    PRIORITY_BATCH,
    PRIORITY_INTERACTIVE,
    PrioritySlotScheduler,
    parse_request_priority,
)


async def _hold(scheduler: PrioritySlotScheduler, priority: str, order: list[str], label: str) -> None:
    await scheduler.acquire(priority, timeout_seconds=1.0)
    order.append(label)
    await asyncio.sleep(0.01)
    scheduler.release()


class PrioritySlotSchedulerTests(SimpleTestCase):
    def test_parse_request_priority_defaults_to_interactive(self) -> None:
        self.assertEqual(parse_request_priority(None), PRIORITY_INTERACTIVE)
        self.assertEqual(parse_request_priority("urgent"), PRIORITY_INTERACTIVE)
        self.assertEqual(parse_request_priority(" Batch "), PRIORITY_BATCH)

    def test_interactive_waiter_is_served_before_earlier_batch(self) -> None:
        async def run() -> list[str]:
            scheduler = PrioritySlotScheduler(1)
            order: list[str] = []
            await scheduler.acquire(PRIORITY_INTERACTIVE, timeout_seconds=1.0)
            batch = asyncio.create_task(_hold(scheduler, PRIORITY_BATCH, order, "batch"))
            await asyncio.sleep(0)
            interactive = asyncio.create_task(
                _hold(scheduler, PRIORITY_INTERACTIVE, order, "interactive"),
            )
            await asyncio.sleep(0)
            scheduler.release()
            await asyncio.gather(batch, interactive)
            return order

        self.assertEqual(asyncio.run(run()), ["interactive", "batch"])

    def test_timeout_removes_waiter_and_keeps_accounting(self) -> None:
        async def run() -> tuple[int, int]:
            scheduler = PrioritySlotScheduler(1)
            await scheduler.acquire(PRIORITY_INTERACTIVE, timeout_seconds=1.0)
            with self.assertRaises(TimeoutError):
                await scheduler.acquire(PRIORITY_BATCH, timeout_seconds=0.01)
            scheduler.release()
            return scheduler.in_use, scheduler.queued(PRIORITY_BATCH)

        self.assertEqual(asyncio.run(run()), (0, 0))

    def test_resize_grants_waiters_without_losing_in_flight(self) -> None:
        async def run() -> int:
            scheduler = PrioritySlotScheduler(1)
            await scheduler.acquire(PRIORITY_INTERACTIVE, timeout_seconds=1.0)
            waiter = asyncio.create_task(scheduler.acquire(PRIORITY_INTERACTIVE, timeout_seconds=1.0))
            await asyncio.sleep(0)
            scheduler.resize(2)
            await waiter
            return scheduler.in_use

        self.assertEqual(asyncio.run(run()), 2)

    def test_slot_granted_at_cancel_time_is_handed_on(self) -> None:
        async def run() -> int:
            scheduler = PrioritySlotScheduler(1)
            await scheduler.acquire(PRIORITY_INTERACTIVE, timeout_seconds=1.0)
            cancelled = asyncio.create_task(scheduler.acquire(PRIORITY_INTERACTIVE, timeout_seconds=1.0))
            await asyncio.sleep(0)
            scheduler.release()
            cancelled.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cancelled
            return scheduler.in_use

        self.assertEqual(asyncio.run(run()), 0)


class RequestPriorityMiddlewareTests(SimpleTestCase):
    def test_middleware_exposes_header_priority_to_handlers(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from orchestrator.gateway.gpu_scheduler import current_request_priority
        from orchestrator.gateway.middleware import RequestPriorityMiddleware

        app = FastAPI()
        app.add_middleware(RequestPriorityMiddleware)

        @app.get("/priority")
        async def priority() -> dict[str, str]:
            return {"priority": current_request_priority()}

        client = TestClient(app)
        self.assertEqual(client.get("/priority").json(), {"priority": PRIORITY_INTERACTIVE})
        batch = client.get("/priority", headers={"X-Nadir-Priority": "batch"})
        self.assertEqual(batch.json(), {"priority": PRIORITY_BATCH})
