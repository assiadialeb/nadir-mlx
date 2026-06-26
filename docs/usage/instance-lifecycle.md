# Instance lifecycle — wake on demand & idle offload

Operator guide for Ollama-like **sleep / wake** behaviour on Nadir (MLX Server).

!!! note "Prerequisite"
    Nadir Gateway (`:11380`) must be running. See [nadir-gateway.md](nadir-gateway.md).

**Epic:** MLX-38 · **ADR:** [006-instance-wake-idle-offload.md](../adr/006-instance-wake-idle-offload.md)

## Concepts

| Term | Meaning |
|------|---------|
| **always_on** | Instance stays running until manually stopped (default, pre-v0.3 behaviour) |
| **on_demand** | Instance may be **stopped** when idle; gateway **wakes** it on first request |
| **Idle offload** | Background watcher stops `on_demand` instances after `idle_minutes` without traffic |
| **Cold start** | Time to load weights into unified memory after wake |

## Per-instance configuration

Set in the instance **server config** (UI advanced / ops section):

| Key | Values | Default |
|-----|--------|---------|
| `lifecycle_mode` | `always_on`, `on_demand` | `always_on` |
| `idle_minutes` | 5–1440 | `30` |

Recommendations:

- Primary chat model → `always_on`
- TTS, STT, image, reranker, secondary chat → `on_demand` with `idle_minutes` 15–60

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` | `300` | Max wait for instance to become RUNNING after wake |
| `NADIR_IDLE_OFFLOAD_ENABLED` | `true` | Global switch for idle watcher |
| `NADIR_IDLE_CHECK_INTERVAL_SECONDS` | `60` | How often idle candidates are evaluated |

## Request flow

```mermaid
sequenceDiagram
    participant Client as Client
    participant GW as Nadir Gateway :11380
    participant LC as lifecycle_services
    participant SM as server_manager
    participant MLX as Instance :114xx

    Client->>GW: POST /v1/chat/completions model=alias
    GW->>LC: ensure_instance_ready(alias)
    alt STOPPED and on_demand
        LC->>SM: start_instance
        SM->>MLX: spawn process
        LC->>LC: wait health (singleflight)
    end
    LC-->>GW: GatewayTarget
    GW->>MLX: proxy request
    MLX-->>GW: response / stream
    GW-->>Client: response / stream
```

## Client timeouts

For **`on_demand`** instances, set client request timeouts ≥ `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` (default 300s). Large VLMs may need 180–300s on first request after sleep.

Clients only need the **gateway alias** in `model` and `api_base` `http://<host>:11380/v1` — no per-instance port.

## Gateway errors

| HTTP | Code | When |
|------|------|------|
| 503 | `model_waking` | Wake in progress (rare if connection is held) |
| 503 | `model_waking_timeout` | Wake exceeded timeout |
| 503 | `model_unavailable` | `always_on` but stopped, or FAILED |

## Verification

1. Create instance with `lifecycle_mode: on_demand`, stop it from UI → status **Stopped**.
2. `curl` chat completion via gateway with `model: <alias>` → succeeds after cold start.
3. Wait `idle_minutes` + one check interval → instance returns **Stopped**, RAM freed.
4. `GET /v1/models` lists alias with `nadir.status` / `nadir.lifecycle_mode` extensions.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Immediate 503 on stopped alias | `lifecycle_mode` still `always_on` |
| Timeout on first request | Increase client timeout and `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` |
| Instance never sleeps | Traffic still hitting instance port directly (bypass gateway) |
| Instance sleeps during long stream | Bug — report; `last_used_at` should update at stream start |

## Related

- [Coverage matrix](nadir-gateway-coverage-matrix.md)
- [ADR 001 — Gateway](../adr/001-nadir-gateway.md)
