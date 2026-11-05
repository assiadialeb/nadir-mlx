"""Cross-process locks serializing start/stop/restart on one inference port slot.

UI (runserver), gateway and MCP are separate processes: ``threading.Lock`` cannot
serialize them. ``fcntl.flock`` on a per-port file works across processes and
SQLite/Postgres alike, and is released by the kernel if the holder dies.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import threading
import time
from pathlib import Path
from typing import Iterator

from django.conf import settings

DEFAULT_SLOT_LOCK_TIMEOUT_SECONDS = 600.0
# User actions wait briefly so they are not rejected by sub-millisecond lock
# probes (health refresh, watchdog, Servers page) on the same port.
USER_ACTION_SLOT_WAIT_SECONDS = 2.0
_POLL_INTERVAL_SECONDS = 0.1
_held = threading.local()


class InstanceSlotBusy(RuntimeError):
    """Another actor holds the start/stop lock for this port."""

    def __init__(self, port: int) -> None:
        super().__init__(
            f"A start or stop is already in progress for port {port}. Retry in a moment."
        )
        self.port = port


def runtime_state_dir() -> Path:
    """Return (and create) the directory holding lock files and runtime stamps."""
    path = Path(settings.NADIR_RUNTIME_STATE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slot_lock_path(port: int) -> Path:
    locks_dir = runtime_state_dir() / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    return locks_dir / f"port-{int(port)}.lock"


def _held_counts() -> dict[int, int]:
    counts = getattr(_held, "counts", None)
    if counts is None:
        counts = {}
        _held.counts = counts
    return counts


def _acquire(fd: int, port: int, *, blocking: bool, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if not blocking or time.monotonic() >= deadline:
                raise InstanceSlotBusy(port) from None
            time.sleep(_POLL_INTERVAL_SECONDS)


@contextlib.contextmanager
def instance_slot_lock(
    port: int,
    *,
    blocking: bool = True,
    timeout_seconds: float = DEFAULT_SLOT_LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    """Hold the exclusive start/stop lock for ``port`` (re-entrant per thread).

    Raises:
        InstanceSlotBusy: lock not acquired (non-blocking, or timeout reached).
    """
    counts = _held_counts()
    if counts.get(port):
        counts[port] += 1
        try:
            yield
        finally:
            counts[port] -= 1
        return

    fd = os.open(_slot_lock_path(port), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        _acquire(fd, port, blocking=blocking, timeout_seconds=timeout_seconds)
        counts[port] = 1
        try:
            yield
        finally:
            counts.pop(port, None)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class SlotLockHandoff:
    """Slot lock acquired (non-blocking) in one thread and released by another.

    Lets an HTTP request reserve the slot and validate synchronously, then hand
    the long transition to a worker thread without a window where a concurrent
    actor could slip in.
    """

    def __init__(self, port: int, *, timeout_seconds: float = 0.0) -> None:
        self.port = port
        self._fd: int | None = os.open(_slot_lock_path(port), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _acquire(
                self._fd,
                port,
                blocking=timeout_seconds > 0,
                timeout_seconds=timeout_seconds,
            )
        except InstanceSlotBusy:
            os.close(self._fd)
            self._fd = None
            raise

    @contextlib.contextmanager
    def held_by_current_thread(self) -> Iterator[None]:
        """Make nested ``instance_slot_lock(port)`` calls re-entrant in this thread."""
        counts = _held_counts()
        counts[self.port] = counts.get(self.port, 0) + 1
        try:
            yield
        finally:
            counts[self.port] -= 1
            if not counts[self.port]:
                counts.pop(self.port, None)

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


def is_slot_lock_held_elsewhere(port: int) -> bool:
    """Return True when another thread or process currently holds the slot lock."""
    if _held_counts().get(port):
        return False
    try:
        with instance_slot_lock(port, blocking=False):
            return False
    except InstanceSlotBusy:
        return True
