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

    Batch waiters are served only when no interactive request is queued and the
    batch share of capacity is not exhausted (``batch_max_share``). Capacity can
    change at runtime without losing track of slots already in flight.
    """

    def __init__(self, capacity: int, *, batch_max_share: float = 1.0) -> None:
        self.capacity = max(1, int(capacity))
        self.batch_max_share = _clamp_share(batch_max_share)
        self.in_use = 0
        self._in_use_by: dict[RequestPriority, int] = {
            PRIORITY_INTERACTIVE: 0,
            PRIORITY_BATCH: 0,
        }
        self._waiters: dict[RequestPriority, deque[asyncio.Future[None]]] = {
            PRIORITY_INTERACTIVE: deque(),
            PRIORITY_BATCH: deque(),
        }

    def resize(self, capacity: int, *, batch_max_share: float | None = None) -> None:
        self.capacity = max(1, int(capacity))
        if batch_max_share is not None:
            self.batch_max_share = _clamp_share(batch_max_share)
        self._grant_free_slots()

    def queued(self, priority: RequestPriority) -> int:
        return sum(1 for waiter in self._waiters[priority] if not waiter.done())

    def batch_capacity(self) -> int:
        """Max slots batch traffic may hold (0 = batch always waits for free interactive)."""
        if self.batch_max_share >= 1.0:
            return self.capacity
        return max(0, int(self.capacity * self.batch_max_share))

    async def acquire(self, priority: RequestPriority, timeout_seconds: float) -> None:
        """Take a slot or wait for one; raise ``TimeoutError`` after ``timeout_seconds``."""
        if self._can_take_immediately(priority):
            self._take(priority)
            return
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters[priority].append(waiter)
        try:
            async with asyncio.timeout(timeout_seconds):
                await waiter
        except BaseException:
            self._abandon(waiter, priority)
            raise

    def release(self, priority: RequestPriority) -> None:
        if self._in_use_by[priority] > 0:
            self._in_use_by[priority] -= 1
        self.in_use = max(0, self.in_use - 1)
        self._grant_free_slots()

    def _take(self, priority: RequestPriority) -> None:
        self.in_use += 1
        self._in_use_by[priority] += 1

    def _can_take_immediately(self, priority: RequestPriority) -> bool:
        if self.in_use >= self.capacity:
            return False
        if self.queued(PRIORITY_INTERACTIVE):
            return False
        if priority == PRIORITY_INTERACTIVE:
            return True
        if self._in_use_by[PRIORITY_BATCH] >= self.batch_capacity():
            return False
        return not self.queued(PRIORITY_BATCH)

    def _abandon(self, waiter: asyncio.Future[None], priority: RequestPriority) -> None:
        if waiter.done() and not waiter.cancelled():
            # Slot was granted just as the caller gave up: hand it on.
            self.release(priority)
            return
        waiter.cancel()
        try:
            self._waiters[priority].remove(waiter)
        except ValueError:
            pass

    def _next_waiter(self) -> tuple[RequestPriority, asyncio.Future[None]] | None:
        for priority in (PRIORITY_INTERACTIVE, PRIORITY_BATCH):
            if priority == PRIORITY_BATCH and self._in_use_by[PRIORITY_BATCH] >= self.batch_capacity():
                continue
            queue = self._waiters[priority]
            while queue:
                waiter = queue.popleft()
                if not waiter.done():
                    return priority, waiter
        return None

    def _grant_free_slots(self) -> None:
        while self.in_use < self.capacity:
            nxt = self._next_waiter()
            if nxt is None:
                return
            priority, waiter = nxt
            self._take(priority)
            waiter.set_result(None)


def _clamp_share(value: float) -> float:
    try:
        share = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(1.0, max(0.0, share))
