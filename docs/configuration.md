# Configuration

Environment variables are loaded from `.env` at the project root (see `.env.example`).

## Django

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `DJANGO_SECRET_KEY` | auto `data/django_secret_key` | Production | Secret key; generated on first run if unset (mode 600) |
| `DJANGO_DEBUG` | `false` | No | Set `true` only for local development |
| `DJANGO_ALLOWED_HOSTS` | `127.0.0.1,localhost` | Production | Comma-separated hostnames |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | — | When `DEBUG=false` | Full URLs with scheme and port |
| `DJANGO_HTTP_PORT` | `8000` | No | Documented default for the control plane |

## Database

SQLite is used when no PostgreSQL host is configured.

| Variable | Default | Description |
|----------|---------|-------------|
| `NADIR_DATABASE_URL` | — | Optional single URL (`postgresql://...`) |
| `NADIR_DB_HOST` | — | When set, enables PostgreSQL |
| `NADIR_DB_PORT` | `5432` | PostgreSQL port |
| `NADIR_DB_NAME` | `nadir_db` | Database name |
| `NADIR_DB_USER` | `nadir_user` | Application user |
| `NADIR_DB_PASSWORD` | — | Database password |

## Inference instances

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_DEFAULT_SERVER_HOST` | `127.0.0.1` | Bind address for MLX backends |
| `MLX_HEALTH_INTERVAL_SECONDS` | `30` | Background health poll interval |
| `MLX_RESTART_BACKOFF_SECONDS` | `30` | Delay before watchdog restart |
| `MLX_INSTANCE_WATCHDOG_ENABLED` | `true` | Auto-restart crashed instances |
| `MLX_INSTANCE_STARTUP_RECONCILE_ENABLED` | `true` | Restore `always_on` instances after process/host restart |
| `MLX_STOP_PORT_RELEASE_TIMEOUT_SECONDS` | `12` | Max wait for port release after stop |
| `MLX_INSTANCE_READY_TIMEOUT_SECONDS` | `300` | Max wait for `/health` after spawn before the start is marked `FAILED` |
| `NADIR_RUNTIME_STATE_DIR` | `data/runtime` | Cross-process slot locks and route-cache generation stamp (share it between UI, gateway and MCP) |
| `MLX_DISABLE_INSTANCE_WATCHDOG` | — | Set `1` to disable watchdog (tests) |

## Nadir Gateway

| Variable | Default | Description |
|----------|---------|-------------|
| `NADIR_GATEWAY_HOST` | `127.0.0.1` | Gateway bind address |
| `NADIR_GATEWAY_PORT` | `11380` | Must stay **outside** `11400–11500` |
| `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS` | `300` | Upstream proxy timeout |
| `NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM` | `16` | Max parallel upstream requests per instance (`0` = unlimited) |
| `NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL` | `16` | Max parallel upstream requests across all instances (`0` = unlimited); instances on one Mac share GPU bandwidth |
| `NADIR_GATEWAY_QUEUE_TIMEOUT_SECONDS` | `300` | Max wait in gateway queue |
| `NADIR_GATEWAY_ROUTE_CACHE_TTL_SECONDS` | `20` | Alias / models cache TTL |
| `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` | `300` | Max wait for `on_demand` wake + health |
| `NADIR_GATEWAY_WAKE_POLL_INTERVAL_SECONDS` | `1` | Health poll during wake |
| `NADIR_GATEWAY_ACTIVITY_HEARTBEAT_SECONDS` | `15` | Mid-request `last_used_at` refresh during long JSON calls and SSE (incl. prefill) |
| `NADIR_IDLE_OFFLOAD_ENABLED` | `true` | Stop idle `on_demand` instances |
| `NADIR_IDLE_CHECK_INTERVAL_SECONDS` | `60` | Idle watcher interval |
| `NADIR_LRU_EVICTION_ENABLED` | `true` | Evict LRU `on_demand` instances before a load when memory is tight |
| `NADIR_LRU_EVICTION_GRACE_SECONDS` | `60` | Min idle age before LRU eviction; floored to `2 ×` activity heartbeat |
| `NADIR_MEMORY_RESERVE_GB` | `4` | Keep this much free memory (GiB) when evaluating LRU eviction |
| `NADIR_MEMORY_PLANNING_CONTEXT_TOKENS` | `8192` | Context length assumed resident when sizing KV for eviction (capped by `max_kv_size` / model max) |
| `NADIR_MEMORY_PLANNING_PROMPT_PREFIX_TOKENS` | `4096` | Assumed shared-prefix length per prompt-cache entry when `prompt_cache_bytes` is unset |
| `NADIR_GATEWAY_PUBLIC_BASE_URL` | `http://127.0.0.1:11380` | Base URL for generated image links |
| `NADIR_GATEWAY_API_KEY` | — | Required when binding outside loopback; `Authorization: Bearer` or `X-API-Key` |
| `NADIR_INFERENCE_API_KEY` | — | Optional auth for Nadir workers (image/stt/embedding/tts); gateway injects it upstream. mlx-lm/mlx-vlm remain loopback-only. |
| `NADIR_GATEWAY_CORS_ORIGINS` | — | Allowed browser `Origin` values; empty rejects any `Origin` (CLI/SDK without Origin still work) |
| `NADIR_GATEWAY_ALLOWED_HOSTS` | — | Extra `Host` values for TrustedHostMiddleware (beyond localhost / 127.0.0.1 / `[::1]`) |
| `NADIR_GATEWAY_MAX_BODY_MB` | `50` | Reject requests whose `Content-Length` exceeds this size |

Per-instance overrides (lifecycle, idle minutes, concurrency) are set in **server config** in the UI. See [Server config reference](usage/server-config-reference.md).

## Nadir MCP

| Variable | Default | Description |
|----------|---------|-------------|
| `NADIR_MCP_HOST` | `127.0.0.1` | Bind address for `manage.py run_mcp` |
| `NADIR_MCP_PORT` | `11381` | Bind port for `manage.py run_mcp` |
| `NADIR_MCP_PATH` | `/mcp` | URL path (Streamable HTTP) |
| `NADIR_MCP_API_KEY` | — | Auth for MCP; falls back to `NADIR_GATEWAY_API_KEY` |
| `NADIR_MCP_ENABLED` | `false` | Opt-in mount of `/mcp` on Django ASGI (not required for `run_mcp`) |
| `NADIR_MCP_ALLOWED_HOSTS` | — | Extra Host values (loopback always allowed) |
| `NADIR_MCP_CORS_ORIGINS` | — | Allowed browser Origins; empty rejects any `Origin` |

See [MCP server](usage/mcp.md).

## Benchmarks & images

| Variable | Default | Description |
|----------|---------|-------------|
| `NADIR_BENCHMARK_ENDPOINT_ENABLED` | `false` | Allow remote benchmark targets (anti-SSRF) |
| `NADIR_BENCHMARK_ENDPOINT_ALLOWED_HOSTS` | — | Host allowlist when enabled |
| `IMAGE_OUTPUT_TTL_SECONDS` | `3600` | TTL for gateway-served generated images |

## Server config (UI / JSON)

Advanced options per instance (thinking mode, tool parser, lifecycle) are documented in [Server config reference](usage/server-config-reference.md).

## Related

- [Installation](installation.md)
- [Nadir Gateway](usage/nadir-gateway.md)
- [Instance lifecycle](usage/instance-lifecycle.md)
