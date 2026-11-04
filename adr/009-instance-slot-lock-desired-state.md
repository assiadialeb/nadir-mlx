# ADR 009 — Instance slot lock & desired state

**Date:** 2026-11-04  
**Status:** Accepted  

## Context

Instances are started and stopped by about ten actors spread over three OS processes:

| Process | Actors |
|---------|--------|
| UI (`runserver`) | Start / Stop / Restart / Edit / Delete views |
| Gateway (`run_gateway`, background owner under `serve`) | Wake on request, speculative wake, idle offload, LRU eviction, watchdog auto-restart, startup reconcile, health reconcile |
| MCP (`run_mcp`) | `start_instance` / `stop_instance` tools |

Observed failures before this decision:

- Serialization used a `threading.Lock` per port, which does not cross process boundaries: a UI start and a gateway wake could spawn two launchers on the same port, the second killing the first.
- Intent was not persisted. A user stop of an `always_on` instance was undone by startup reconcile after a reboot, and by the watchdog if the stop raced a health check.
- The "stop in progress" flag lived in `server_config.ops` JSON, written with read-modify-write by several actors (lost updates), and stayed set forever if the stopping process died.
- Stored PIDs were trusted blindly. After a reboot a recycled PID (or any program listening on the port) could be signalled.
- The UI start blocked the HTTP request for the whole model load, and rows were marked `RUNNING` as soon as the process existed, before `/health` answered.
- Warmup claims and the gateway route cache were per-process memory, so the UI and MCP never invalidated the gateway's view.

## Decision

1. **Per-port slot lock (`orchestrator/instance_locks.py`).** `fcntl.flock` on `NADIR_RUNTIME_STATE_DIR/locks/port-<port>.lock`, re-entrant per thread. Every transition (`prepare_instance_launch`, `launch_prepared_instance`, `stop_instance`, `restart_instance`, `update_stopped_instance`, `delete_instance`) runs under it. User actions and background actors use non-blocking acquisition (`InstanceSlotBusy`) so no actor waits on a slot held by a symmetric actor (eviction vs. loader). `SlotLockHandoff` lets the UI request reserve the slot and validate synchronously, then hand the lock to a worker thread.
2. **`desired_state` column** (`running` / `stopped`). Written only by explicit intent: user/MCP start and gateway wake → `running`; user/MCP stop, idle offload, LRU eviction → `stopped`. Watchdog and startup reconcile call `restart_instance(automatic=True)`, which re-reads the row under the lock and returns untouched unless intent is `running`.
3. **`stopping_since` column** replaces the JSON flag. Readers treat it as stale after 60 s, so a crashed stopper never freezes a slot.
4. **PID identity.** A PID is trusted only if `ps -o command=` shows an `orchestrator.*` launcher with `--port <port>`. Stop targets are the launcher trees and listeners that belong to Nadir; foreign listeners are never killed.
5. **Two-phase start, LOADING until healthy.** `prepare_instance_launch` reserves the row (`LOADING`); `launch_prepared_instance` spawns, polls `/health` (`MLX_INSTANCE_READY_TIMEOUT_SECONDS`), runs warmup while still `LOADING`, re-checks intent, then marks `RUNNING`. UI Start / Restart run phase two in a worker thread.
6. **Cross-process route cache invalidation.** `clear_gateway_route_cache()` bumps the mtime of `route-cache.generation`; the gateway rebuilds its snapshot when the stamp changes.
7. **Hygiene.** Migration `0015` seeds `desired_state` from status, drops the legacy JSON flag and rebinds `0.0.0.0` hosts to `127.0.0.1`. Benchmarks record `runner_pid`; the watchdog fails runs whose runner died.

## Alternatives

| Option | Rejected because |
|--------|------------------|
| Database row lock (`SELECT … FOR UPDATE`) | Not available on SQLite (default local install); holding a transaction open during a multi-minute model load is fragile |
| Single coordinator process (all transitions via the gateway) | Requires a new internal API and makes the UI unusable when the gateway is down |
| Celery / Redis task queue | New infrastructure for a local-first single-host tool |
| Keep `threading.Lock` and route everything through one process | UI, gateway and MCP are separate processes by design (`manage.py serve`) |

## Consequences

- Start / stop are serialized per port across all processes; concurrent requests get an explicit toast instead of racing.
- Operator intent survives reboots: a stopped `always_on` model stays stopped until started again.
- The Servers page shows `Loading` until the model actually serves, and auto-refreshes while a transition runs.
- `NADIR_RUNTIME_STATE_DIR` must be shared when processes run in separate containers.
- `flock` is advisory and POSIX-only (macOS / Linux), which matches the supported platforms.
