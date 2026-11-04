# Nadir MCP server

HTTP **Streamable MCP** endpoint so IDE agents (Cursor, Claude Desktop, etc.) can list, start, and stop local MLX instances without leaving the chat.

Transport: Streamable HTTP (compatible with clients that still label the URL transport as “SSE”).

## Quick start

```bash
# Terminal 1 — Django control plane (existing)
python manage.py runserver

# Terminal 2 — gateway (existing)
python manage.py run_gateway

# Terminal 3 — MCP
python manage.py run_mcp
# http://127.0.0.1:11381/mcp
```

Optional API key (recommended; **required** for non-loopback binds):

```bash
export NADIR_MCP_API_KEY=change-me
# falls back to NADIR_GATEWAY_API_KEY when NADIR_MCP_API_KEY is unset
python manage.py run_mcp
```

## Tools

| Tool | Purpose |
|------|---------|
| `nadir_instances_list` | List instances (id, alias, status, port, mode) |
| `nadir_instance_get` | Fetch one instance by id |
| `nadir_instance_start` | Start / restart a registered instance |
| `nadir_instance_stop` | Stop an instance |
| `nadir_models_list` | List downloaded folders under `./models` |
| `nadir_gateway_health` | Probe gateway `/health` + advertised aliases |

## Cursor (`mcp.json`)

```json
{
  "mcpServers": {
    "nadir": {
      "url": "http://127.0.0.1:11381/mcp",
      "headers": {
        "Authorization": "Bearer change-me"
      }
    }
  }
}
```

Omit `headers` when no `NADIR_MCP_API_KEY` / `NADIR_GATEWAY_API_KEY` is set (loopback only).

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `NADIR_MCP_HOST` | `127.0.0.1` | Bind address for `run_mcp` |
| `NADIR_MCP_PORT` | `11381` | Bind port for `run_mcp` |
| `NADIR_MCP_PATH` | `/mcp` | URL path |
| `NADIR_MCP_API_KEY` | — | Auth for MCP (falls back to gateway key) |
| `NADIR_MCP_ENABLED` | `false` | Opt-in mount of `/mcp` on Django ASGI (`uvicorn`) |
| `NADIR_MCP_ALLOWED_HOSTS` | — | Extra Host values (loopback always allowed) |
| `NADIR_MCP_CORS_ORIGINS` | — | Allowed browser `Origin` values; empty rejects any `Origin` |

## Combined ASGI (optional)

Instead of `runserver` + `run_mcp`, serve Django and MCP on one port (loopback):

```bash
export NADIR_MCP_ENABLED=true
uvicorn mlx_orchestrator.asgi:application --host 127.0.0.1 --port 8000
```

MCP is then at `http://127.0.0.1:8000/mcp`. Do **not** enable this on a LAN bind without `NADIR_MCP_API_KEY`.

## Security

- Prefer loopback bind (`manage.py run_mcp` default).
- DNS-rebinding protection is **always on**: `Host` must be loopback (or `NADIR_MCP_ALLOWED_HOSTS`); any `Origin` not in `NADIR_MCP_CORS_ORIGINS` is rejected (403).
- Non-loopback bind without an API key is refused (`run_mcp` at startup; ASGI middleware returns 401).
- Start/stop are powerful — treat the MCP key like a local admin credential.

## Related

- [Integration recipes](integration-recipes.md)
- [Nadir Gateway](nadir-gateway.md)
- [Instance lifecycle](instance-lifecycle.md)
