# Configuration

Environment variables are loaded from `.env` at the project root (see `.env.example`).

## Django

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `DJANGO_SECRET_KEY` | — | Production | Secret key for sessions and signing |
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
| `MLX_STOP_PORT_RELEASE_TIMEOUT_SECONDS` | `12` | Max wait for port release after stop |
| `MLX_DISABLE_INSTANCE_WATCHDOG` | — | Set `1` to disable watchdog (tests) |

## Nadir Gateway

| Variable | Default | Description |
|----------|---------|-------------|
| `NADIR_GATEWAY_HOST` | `127.0.0.1` | Gateway bind address |
| `NADIR_GATEWAY_PORT` | `11380` | Must stay **outside** `11400–11500` |
| `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS` | `300` | Upstream proxy timeout |
| `NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM` | `16` | Max parallel upstream requests per instance (`0` = unlimited) |
| `NADIR_GATEWAY_QUEUE_TIMEOUT_SECONDS` | `300` | Max wait in gateway queue |
| `NADIR_GATEWAY_ROUTE_CACHE_TTL_SECONDS` | `20` | Alias / models cache TTL |
| `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` | `300` | Max wait for `on_demand` wake + health |
| `NADIR_GATEWAY_WAKE_POLL_INTERVAL_SECONDS` | `1` | Health poll during wake |
| `NADIR_IDLE_OFFLOAD_ENABLED` | `true` | Stop idle `on_demand` instances |
| `NADIR_IDLE_CHECK_INTERVAL_SECONDS` | `60` | Idle watcher interval |
| `NADIR_GATEWAY_PUBLIC_BASE_URL` | `http://127.0.0.1:11380` | Base URL for generated image links |

Per-instance overrides (lifecycle, idle minutes, concurrency) are set in **server config** in the UI. See [Server config reference](usage/server-config-reference.md).

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
