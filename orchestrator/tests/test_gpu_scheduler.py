"""Tests for the priority-aware upstream slot scheduler (MLX-128)."""

from __future__ import annotations

import asyncio

from django.test import SimpleTestCase

from orchestrator.gateway.gpu_scheduler import (
    PRIORITY_BATCH,
    PRIORITY_INTERACTIVE,
    GatewayUpstreamScheduler,
    PrioritySlotScheduler,
    _UpstreamWaiter,
    parse_request_priority,
)


async def _hold(scheduler: PrioritySlotScheduler, priority: str, order: list[str], label: str) -> None:
    await scheduler.acquire(priority, timeout_seconds=1.0)
    order.append(label)
    await asyncio.sleep(0.01)
    scheduler.release(PRIORITY_INTERACTIVE)


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
            scheduler.release(PRIORITY_INTERACTIVE)
            await asyncio.gather(batch, interactive)
            return order

        self.assertEqual(asyncio.run(run()), ["interactive", "batch"])

    def test_timeout_removes_waiter_and_keeps_accounting(self) -> None:
        async def run() -> tuple[int, int]:
            scheduler = PrioritySlotScheduler(1)
            await scheduler.acquire(PRIORITY_INTERACTIVE, timeout_seconds=1.0)
            with self.assertRaises(TimeoutError):
                await scheduler.acquire(PRIORITY_BATCH, timeout_seconds=0.01)
            scheduler.release(PRIORITY_INTERACTIVE)
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

    def test_abandon_after_grant_releases_slot(self) -> None:
        async def run() -> int:
            scheduler = PrioritySlotScheduler(1)
            await scheduler.acquire(PRIORITY_INTERACTIVE, timeout_seconds=1.0)
            waiter = asyncio.get_running_loop().create_future()
            scheduler._waiters[PRIORITY_INTERACTIVE].append(waiter)
            scheduler.release(PRIORITY_INTERACTIVE)  # grants waiter
            self.assertTrue(waiter.done())
            scheduler._abandon(waiter, PRIORITY_INTERACTIVE)
            return scheduler.in_use

        self.assertEqual(asyncio.run(run()), 0)


    def test_batch_capacity_at_least_one_when_share_positive(self) -> None:
        scheduler = PrioritySlotScheduler(1, batch_max_share=0.5)
        self.assertEqual(scheduler.batch_capacity(), 1)

        async def run() -> None:
            await scheduler.acquire(PRIORITY_BATCH, timeout_seconds=0.5)
            scheduler.release(PRIORITY_BATCH)

        asyncio.run(run())

    def test_batch_share_reserves_capacity_for_interactive(self) -> None:
        async def run() -> str:
            scheduler = PrioritySlotScheduler(2, batch_max_share=0.5)
            await scheduler.acquire(PRIORITY_BATCH, timeout_seconds=1.0)
            # Second batch must wait: only 1 of 2 slots may be batch (50%).
            batch2 = asyncio.create_task(scheduler.acquire(PRIORITY_BATCH, timeout_seconds=1.0))
            await asyncio.sleep(0)
            await scheduler.acquire(PRIORITY_INTERACTIVE, timeout_seconds=1.0)
            self.assertFalse(batch2.done())
            scheduler.release(PRIORITY_INTERACTIVE)
            scheduler.release(PRIORITY_BATCH)
            await batch2
            scheduler.release(PRIORITY_BATCH)
            return "ok"

        self.assertEqual(asyncio.run(run()), "ok")



class GatewayUpstreamSchedulerTests(SimpleTestCase):
    def test_saturated_instance_does_not_block_other_model(self) -> None:
        async def run() -> list[str]:
            scheduler = GatewayUpstreamScheduler(2, batch_max_share=1.0)
            order: list[str] = []

            async def hold(instance_id: int, label: str, delay: float) -> None:
                await scheduler.acquire(
                    instance_id=instance_id,
                    instance_limit=1,
                    priority=PRIORITY_INTERACTIVE,
                    timeout_seconds=2.0,
                )
                order.append(f"{label}-start")
                await asyncio.sleep(delay)
                order.append(f"{label}-end")
                scheduler.release(
                    instance_id=instance_id,
                    instance_limit=1,
                    priority=PRIORITY_INTERACTIVE,
                )

            tasks = [
                asyncio.create_task(hold(1, "A1", 0.08)),
                asyncio.create_task(hold(1, "A2", 0.01)),
                asyncio.create_task(hold(1, "A3", 0.01)),
                asyncio.create_task(hold(2, "B1", 0.01)),
            ]
            await asyncio.gather(*tasks)
            return order

        order = asyncio.run(run())
        self.assertIn("B1-start", order)
        b_index = order.index("B1-start")
        a2_index = order.index("A2-start")
        self.assertLess(b_index, a2_index)

    def test_abandon_after_grant_wakes_next_waiter(self) -> None:
        """G2-c: slot freed on cancel must call _grant_waiters (not only _release)."""

        async def run() -> None:
            scheduler = GatewayUpstreamScheduler(1, batch_max_share=1.0)
            loop = asyncio.get_running_loop()
            await scheduler.acquire(
                instance_id=1,
                instance_limit=1,
                priority=PRIORITY_INTERACTIVE,
                timeout_seconds=1.0,
            )
            waiter_b = _UpstreamWaiter(
                instance_id=1,
                instance_limit=1,
                priority=PRIORITY_INTERACTIVE,
                future=loop.create_future(),
            )
            waiter_c = _UpstreamWaiter(
                instance_id=1,
                instance_limit=1,
                priority=PRIORITY_INTERACTIVE,
                future=loop.create_future(),
            )
            scheduler._interactive.append(waiter_b)
            scheduler._interactive.append(waiter_c)
            scheduler.release(
                instance_id=1,
                instance_limit=1,
                priority=PRIORITY_INTERACTIVE,
            )
            self.assertTrue(waiter_b.future.done())
            self.assertFalse(waiter_c.future.done())
            scheduler._release(
                instance_id=1,
                instance_limit=1,
                priority=PRIORITY_INTERACTIVE,
            )
            scheduler._grant_waiters()
            self.assertTrue(waiter_c.future.done())

        asyncio.run(run())


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
