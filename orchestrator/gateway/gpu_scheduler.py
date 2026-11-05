"""Priority-aware upstream slot scheduling (interactive before batch)."""

from __future__ import annotations

import asyncio
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field
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
    """Single-pool counting semaphore (used in unit tests and legacy paths).

    Batch waiters are served only when no interactive request is queued and the
    batch share of capacity is not exhausted (``batch_max_share``).
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
        """Max slots batch traffic may hold."""
        if self.batch_max_share <= 0.0:
            return 0
        if self.batch_max_share >= 1.0:
            return self.capacity
        return max(1, int(self.capacity * self.batch_max_share))

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


@dataclass
class _UpstreamWaiter:
    instance_id: int
    instance_limit: int | None
    priority: RequestPriority
    future: asyncio.Future[None] = field(repr=False)


class GatewayUpstreamScheduler:
    """Reserves global + per-instance capacity atomically for one upstream request.

    Waiters blocked on a full instance do not hold global slots. Interactive
    waiters are preferred over batch; within a class, skip waiters whose instance
    is full so other models can proceed.
    """

    def __init__(
        self,
        global_limit: int | None,
        *,
        batch_max_share: float = 0.5,
    ) -> None:
        self.global_limit = global_limit if global_limit is None else max(1, int(global_limit))
        self.batch_max_share = _clamp_share(batch_max_share)
        self.global_in_use = 0
        self.global_batch_in_use = 0
        self.instance_in_use: dict[int, int] = {}
        self._interactive: deque[_UpstreamWaiter] = deque()
        self._batch: deque[_UpstreamWaiter] = deque()
        self._lock = asyncio.Lock()

    def global_batch_capacity(self) -> int:
        if self.global_limit is None:
            return 0
        if self.batch_max_share <= 0.0:
            return 0
        if self.batch_max_share >= 1.0:
            return self.global_limit
        return max(1, int(self.global_limit * self.batch_max_share))

    async def acquire(
        self,
        *,
        instance_id: int,
        instance_limit: int | None,
        priority: RequestPriority,
        timeout_seconds: float,
    ) -> None:
        async with self._lock:
            if self._can_grant(instance_id, instance_limit, priority):
                self._grant(instance_id, instance_limit, priority)
                return
            waiter = _UpstreamWaiter(
                instance_id=instance_id,
                instance_limit=instance_limit,
                priority=priority,
                future=asyncio.get_running_loop().create_future(),
            )
            if priority == PRIORITY_INTERACTIVE:
                self._interactive.append(waiter)
            else:
                self._batch.append(waiter)
        try:
            async with asyncio.timeout(timeout_seconds):
                await waiter.future
        except BaseException:
            async with self._lock:
                self._remove_waiter(waiter)
                if waiter.future.done() and not waiter.future.cancelled():
                    self._release(instance_id, instance_limit, priority)
            raise

    async def release(
        self,
        *,
        instance_id: int,
        instance_limit: int | None,
        priority: RequestPriority,
    ) -> None:
        async with self._lock:
            self._release(instance_id, instance_limit, priority)
            self._grant_waiters()

    def _can_grant(
        self,
        instance_id: int,
        instance_limit: int | None,
        priority: RequestPriority,
    ) -> bool:
        if not self._global_has_room(priority):
            return False
        return self._instance_has_room(instance_id, instance_limit)

    def _global_has_room(self, priority: RequestPriority) -> bool:
        if self.global_limit is None:
            return True
        if self.global_in_use >= self.global_limit:
            return False
        if priority != PRIORITY_BATCH:
            return True
        return self.global_batch_in_use < self.global_batch_capacity()

    def _instance_has_room(self, instance_id: int, instance_limit: int | None) -> bool:
        if instance_limit is None:
            return True
        return self.instance_in_use.get(instance_id, 0) < instance_limit

    def _grant(
        self,
        instance_id: int,
        instance_limit: int | None,
        priority: RequestPriority,
    ) -> None:
        if self.global_limit is not None:
            self.global_in_use += 1
            if priority == PRIORITY_BATCH:
                self.global_batch_in_use += 1
        if instance_limit is not None:
            self.instance_in_use[instance_id] = self.instance_in_use.get(instance_id, 0) + 1

    def _release(
        self,
        instance_id: int,
        instance_limit: int | None,
        priority: RequestPriority,
    ) -> None:
        if self.global_limit is not None:
            self.global_in_use = max(0, self.global_in_use - 1)
            if priority == PRIORITY_BATCH:
                self.global_batch_in_use = max(0, self.global_batch_in_use - 1)
        if instance_limit is not None:
            used = max(0, self.instance_in_use.get(instance_id, 0) - 1)
            if used == 0:
                self.instance_in_use.pop(instance_id, None)
            else:
                self.instance_in_use[instance_id] = used

    def _remove_waiter(self, target: _UpstreamWaiter) -> None:
        for queue in (self._interactive, self._batch):
            try:
                queue.remove(target)
                return
            except ValueError:
                continue

    def _grant_waiters(self) -> None:
        while True:
            waiter = self._pick_grantable_waiter()
            if waiter is None:
                return
            self._grant(waiter.instance_id, waiter.instance_limit, waiter.priority)
            waiter.future.set_result(None)

    def _pick_grantable_waiter(self) -> _UpstreamWaiter | None:
        for queue in (self._interactive, self._batch):
            for waiter in list(queue):
                if not self._can_grant(
                    waiter.instance_id,
                    waiter.instance_limit,
                    waiter.priority,
                ):
                    continue
                queue.remove(waiter)
                return waiter
        return None


def _clamp_share(value: float) -> float:
    try:
        share = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(1.0, max(0.0, share))
