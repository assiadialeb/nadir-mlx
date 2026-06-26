# ADR 006 — Instance wake & idle offload (Ollama-like lifecycle)

**Date:** 2026-06-23  
**Status:** Accepted  
**Epic:** MLX-38

## Context

[ADR 001](001-nadir-gateway.md) delivered **Nadir Gateway** (`:11380`) as the single OpenAI-compatible entrypoint. Today the gateway only proxies to instances in **RUNNING** status. A **STOPPED**, **LOADING**, or **FAILED** alias returns `503 model_unavailable` — there is no automatic wake.

On a RAM-constrained Mac Studio, operators want **Ollama-like** behaviour:

- **Wake on demand:** first request to a stopped alias starts the MLX process and waits until healthy.
- **Idle offload:** after configurable inactivity, stop the process and free unified memory.
- **Always-on option:** keep current behaviour for latency-critical models (default chat model).

Prerequisites from ADR 001 are satisfied: clients call one gateway URL; alias routing via `model` / `server_config.model_id` exists (MLX-19).

Existing building blocks:

| Component | Role |
|-----------|------|
| `orchestrator/server_manager.py` | `start_instance`, `stop_instance`, status transitions |
| `orchestrator/gateway/selectors.py` | `resolve_gateway_target`, `_UNAVAILABLE_STATUSES` |
| `orchestrator/instance_watchdog.py` | Periodic health + opt-in `auto_restart` |
| `orchestrator/server_config_schema.py` | Per-instance ops fields (`auto_restart`, …) |
| `InferenceInstance` | `status`, `stopped_at`, `health_status` |

## Decision

### 1. Lifecycle policy (per instance, in `server_config`)

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `lifecycle_mode` | `always_on` \| `on_demand` | `always_on` | Whether idle offload applies |
| `idle_minutes` | int (5–1440) | `30` | Stop after this many minutes without gateway traffic (`on_demand` only) |

Backward compatibility: missing fields ⇒ `always_on` (no behaviour change for existing instances).

### 2. Activity tracking

Add `InferenceInstance.last_used_at` (nullable `DateTimeField`, UTC). Updated by the gateway when a request is accepted for proxying (including streaming start). Idle watcher uses `last_used_at`; fallback to `stopped_at` / `created_at` when null.

### 3. Wake path (`ensure_instance_ready`)

New service `orchestrator/lifecycle_services.py`:

```
ensure_instance_ready(alias) -> GatewayTarget
```

Flow:

1. If alias is **RUNNING** and healthy ⇒ return target immediately.
2. If `lifecycle_mode != on_demand` and status is **STOPPED** ⇒ raise `model_unavailable` (no auto wake).
3. If **on_demand** and **STOPPED** ⇒ call `start_instance` for the existing row (reuse port/config), set **LOADING**.
4. **Singleflight:** concurrent wakes for the same alias share one in-process `asyncio.Lock` / thread lock keyed by alias (gateway worker).
5. Poll DB + `/health` on instance port until **RUNNING** or timeout (`NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS`, default `300`).
6. On timeout ⇒ **FAILED** or revert to **STOPPED**, return `503` with code `model_waking_timeout`.

Gateway integrates `ensure_instance_ready` **before** `resolve_gateway_target` on all `/v1/*` proxy routes.

Client-visible behaviour (Ollama parity): the HTTP connection stays open during cold start (within wake timeout), then proxies normally.

### 4. Idle offload watcher

New Django background loop `orchestrator/instance_idle_watcher.py` (same process model as `instance_watchdog`):

- Interval: `NADIR_IDLE_CHECK_INTERVAL_SECONDS` (default `60`).
- Candidate: `lifecycle_mode=on_demand`, `status=RUNNING`, `last_used_at + idle_minutes < now`.
- **Grace:** skip if active gateway singleflight lock held for alias.
- **Grace:** skip if `auto_restart` recovery in progress (reuse watchdog flags).
- Action: `stop_instance(instance)` — same path as manual stop (sets **STOPPED**, frees RAM).
- Global kill switch: `NADIR_IDLE_OFFLOAD_ENABLED` (default `true`).

Watchdog `auto_restart` remains independent: it recovers **DOWN** running instances; idle watcher stops **healthy but unused** instances.

### 5. Gateway API surface

| Topic | Behaviour |
|-------|-----------|
| `GET /v1/models` | Include **all** registered aliases; add extension `nadir.lifecycle_mode`, `nadir.status` (`stopped` / `loading` / `running` / `failed`) so clients can show sleeping models |
| Errors | `503 model_waking` while **LOADING** after wake triggered; `503 model_waking_timeout` on wake timeout |
| `Retry-After` | Optional header on `model_waking` when not holding connection (future); v1 holds connection |

### 6. Control plane UI

- Instance form: lifecycle mode select + idle minutes (shown when `on_demand`).
- List badges: **Sleeping** (STOPPED + on_demand), **Waking** (LOADING), **Ready** (RUNNING), unchanged for always_on.
- Manual Start/Stop unchanged; stopping an always_on instance stays operator-driven.

### 7. Operator guidance

Document in `docs/usage/instance-lifecycle.md`:

- Increase client request timeouts for cold starts (large VLMs may need 120–300s).
- Single `api_base` to gateway suffices; no per-model port when all aliases are on gateway.
- Recommend `on_demand` for TTS, STT, IMAGE, secondary rerankers; `always_on` for primary chat.

## Alternatives

| Option | Why rejected |
|--------|----------------|
| Wake inside Django HTTP (control plane) | Gateway is the hot path; adds latency and couples Gunicorn to subprocess launch |
| Gateway calls `subprocess` directly | Duplicates `server_manager`; breaks single source of truth for ports/PIDs |
| Unload in gateway process only | Gateway restart would lose idle timers; Django DB is source of truth |
| New status `WAKING` | Extra migration/UI; **LOADING** already exists and matches startup |
| Immediate 503 + client retry only | Poor DX vs Ollama; many clients do not retry cold models |
| Global idle timeout in settings only | Operators need per-model tuning (7B vs 27B VLM) |

## Consequences

**Positive**

- Frees tens of GB on idle secondary models without changing client URLs.
- Cluster clients can point all Mac backends to `:11380` with heterogeneous sleep policies.
- Reuses proven `start_instance` / `stop_instance` and watchdog patterns.

**Negative**

- First request after idle pays full model load time (30s–3min).
- Gateway worker blocks during wake (acceptable for default single-worker deployment; document for multi-worker).
- Race: idle stop vs incoming request — mitigated by singleflight + recheck `last_used_at` before stop.
- `GET /v1/models` payload grows (all aliases, not only RUNNING).

**Risks**

- Streaming requests must refresh `last_used_at` at start so long streams are not offloaded mid-flight.
- Route cache (MLX-31) must invalidate on status transitions.

## Out of scope (this epic)

- LRU eviction when waking would exceed RAM budget
- Priority preemption between models
- Gateway API key auth (unchanged)
- Multi-worker uvicorn coordination beyond DB-backed singleflight

## Implementation map (Gravity)

| Ticket | Deliverable |
|--------|-------------|
| MLX-39 | This ADR + `lifecycle_mode` / `idle_minutes` schema + validation + migration `last_used_at` |
| MLX-40 | `ensure_instance_ready` + singleflight + settings |
| MLX-41 | Gateway hook + route cache invalidation + `/v1/models` extensions |
| MLX-42 | `last_used_at` updates on proxy (incl. streaming) |
| MLX-43 | `instance_idle_watcher` + env toggles |
| MLX-44 | UI lifecycle fields + status badges |
| MLX-45 | Operator runbook + coverage matrix + client timeout guidance |
| MLX-46 | Unit + integration tests (wake, timeout, idle, always_on guard) |

## References

- [ADR 001 — Nadir Gateway](001-nadir-gateway.md)
- `orchestrator/server_manager.py`
- `orchestrator/gateway/selectors.py`
- `orchestrator/instance_watchdog.py`
- `docs/usage/instance-lifecycle.md`
- `docs/usage/nadir-gateway-coverage-matrix.md`
