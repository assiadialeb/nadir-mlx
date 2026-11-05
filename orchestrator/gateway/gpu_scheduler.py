"""Priority-aware upstream slot scheduling (interactive before batch)."""

from __future__ import annotations

import asyncio
from collections import deque
from contextvars import ContextVar
from typing import Literal

RequestPriority = Literal["interactive", "batch"]

PRIORITY_INTERACTIVE: RequestPriority = "interactive"
PRIORITY_BATCH: RequestPriority = "batch"
PRIORITY_HEADER = "x-nadir-priority"

_current_priority: ContextVar[RequestPriority] = ContextVar(
    "nadir_request_priority",
    default=PRIORITY_INTERACTIVE,
)


def parse_request_priority(raw_value: str | None) -> RequestPriority:
    """Map the ``X-Nadir-Priority`` header to a priority (unknown values = interactive)."""
    if (raw_value or "").strip().lower() == PRIORITY_BATCH:
        return PRIORITY_BATCH
    return PRIORITY_INTERACTIVE


def set_request_priority(priority: RequestPriority) -> None:
    _current_priority.set(priority)


def current_request_priority() -> RequestPriority:
    return _current_priority.get()


class PrioritySlotScheduler:
    """Counting semaphore that grants freed slots to interactive waiters first.

    Batch waiters are only served when no interactive request is queued, so
    background jobs never delay an agent or chat turn. The capacity can change
    at runtime without losing track of slots already in flight.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self.in_use = 0
        self._waiters: dict[RequestPriority, deque[asyncio.Future[None]]] = {
            PRIORITY_INTERACTIVE: deque(),
            PRIORITY_BATCH: deque(),
        }

    def resize(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._grant_free_slots()

    def queued(self, priority: RequestPriority) -> int:
        return sum(1 for waiter in self._waiters[priority] if not waiter.done())

    async def acquire(self, priority: RequestPriority, timeout_seconds: float) -> None:
        """Take a slot or wait for one; raise ``TimeoutError`` after ``timeout_seconds``."""
        if self._can_take_immediately(priority):
            self.in_use += 1
            return
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters[priority].append(waiter)
        try:
            async with asyncio.timeout(timeout_seconds):
                await waiter
        except BaseException:
            self._abandon(waiter, priority)
            raise

    def release(self) -> None:
        self.in_use = max(0, self.in_use - 1)
        self._grant_free_slots()

    def _can_take_immediately(self, priority: RequestPriority) -> bool:
        if self.in_use >= self.capacity:
            return False
        if self.queued(PRIORITY_INTERACTIVE):
            return False
        return priority == PRIORITY_INTERACTIVE or not self.queued(PRIORITY_BATCH)

    def _abandon(self, waiter: asyncio.Future[None], priority: RequestPriority) -> None:
        if waiter.done() and not waiter.cancelled():
            # Slot was granted just as the caller gave up: hand it on.
            self.release()
            return
        waiter.cancel()
        try:
            self._waiters[priority].remove(waiter)
        except ValueError:
            pass

    def _next_waiter(self) -> asyncio.Future[None] | None:
        for priority in (PRIORITY_INTERACTIVE, PRIORITY_BATCH):
            queue = self._waiters[priority]
            while queue:
                waiter = queue.popleft()
                if not waiter.done():
                    return waiter
        return None

    def _grant_free_slots(self) -> None:
        while self.in_use < self.capacity:
            waiter = self._next_waiter()
            if waiter is None:
                return
            self.in_use += 1
            waiter.set_result(None)
