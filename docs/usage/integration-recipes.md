# Integration recipes

Copy-paste configs for common clients against **Nadir Gateway**.

!!! note "Prerequisites"
    1. Control plane on `:8000` and gateway on `:11380` — see [Nadir Gateway](nadir-gateway.md).
    2. At least one instance with a known **gateway alias** (`GET http://127.0.0.1:11380/v1/models`).
    3. For `on_demand` aliases, client timeouts ≥ `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` (default **300s**).

## Shared settings

| Setting | Value |
|---------|--------|
| OpenAI **base URL** | `http://127.0.0.1:11380/v1` |
| Ollama **base URL** | `http://127.0.0.1:11380` (no `/v1`) |
| Anthropic **base URL** | `http://127.0.0.1:11380` (routes under `/v1/messages`) |
| **Model** | Gateway alias (exact string from the UI / `/v1/models`) |
| **API key** | Omit on loopback if unset; otherwise `Authorization: Bearer <NADIR_GATEWAY_API_KEY>` or `X-API-Key` |
| Docker on macOS | Use `http://host.docker.internal:11380/v1` instead of `127.0.0.1` |

!!! tip "Team LAN: keys, quotas, metrics"
    Nadir keeps a **single** optional gateway API key. For multi-key auth, per-user quotas, usage dashboards, and Prometheus scrapes, put **[LiteLLM](#litellm)** (or another proxy) **in front** of Nadir. Do not expect those features inside the Nadir UI.

---

## Open WebUI

Two options: OpenAI-compatible connection (recommended) or Ollama façade.

### OpenAI connection

1. **Admin** → **Settings** → **Connections** → add **OpenAI**
2. **API Base URL:** `http://host.docker.internal:11380/v1` (Open WebUI in Docker) or `http://127.0.0.1:11380/v1` (native)
3. **API Key:** `sk-local` (or your `NADIR_GATEWAY_API_KEY`)
4. Save, then pick the gateway alias as the model in chat

### Ollama connection

1. **Connections** → **Ollama**
2. **URL:** `http://host.docker.internal:11380` (no `/v1`)
3. Models come from `GET /api/tags` on the gateway

!!! warning "Timeouts"
    Raise Open WebUI request timeouts for cold starts (large VLMs often need 180–300s).

---

## Continue

Edit `~/.continue/config.yaml` (or the workspace config):

```yaml
name: Local Nadir
version: 1.0.0
schema: v1
models:
  - name: Nadir chat
    provider: openai
    model: <gateway-alias>
    apiBase: http://127.0.0.1:11380/v1
    apiKey: sk-local
  - name: Nadir embed
    provider: openai
    model: <embedding-alias>
    apiBase: http://127.0.0.1:11380/v1
    apiKey: sk-local
    roles:
      - embed
```

Replace aliases with values from `GET /v1/models`. The embedding model must be an **EMBEDDING** launch-mode instance.

Alternative: set `provider: ollama` with `apiBase: http://127.0.0.1:11380` (no `/v1`) if you prefer the Ollama façade.

---

## Cursor

1. **Cursor Settings** → **Models**
2. Enable **OpenAI API Key** / override base URL (wording varies by Cursor version)
3. **Override OpenAI Base URL:** `http://127.0.0.1:11380/v1`
4. **API Key:** `sk-local` or `NADIR_GATEWAY_API_KEY`
5. Add a custom model whose id is the **gateway alias**

For Agent / tools flows, use a TEXT alias with tool support configured — see [Chat tools model matrix](chat-tools-model-matrix.md).

!!! note "Cursor + Anthropic-shaped clients"
    If a Cursor feature expects Anthropic Messages, point it at the gateway host (`http://127.0.0.1:11380`) and use `/v1/messages` (already exposed). Prefer the OpenAI path when available.

---

## Claude Code

Claude Code speaks the **Anthropic Messages** API. Nadir exposes that façade on the gateway.

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:11380
export ANTHROPIC_API_KEY=sk-local   # or NADIR_GATEWAY_API_KEY
# Select / pass the gateway alias as the model name in Claude Code settings
```

Smoke test without Claude Code:

```bash
curl -s http://127.0.0.1:11380/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-local" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "<gateway-alias>",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "ping"}]
  }'
```

!!! tip "Thinking / tools"
    Gateway maps Anthropic `thinking` and `tool_use` from upstream reasoning / tools when the TEXT model supports them. See the [API coverage matrix](nadir-gateway-coverage-matrix.md).

---

## n8n

1. Add credentials **Header Auth** or use the OpenAI node custom base URL
2. **OpenAI** node → **Base URL:** `http://127.0.0.1:11380/v1`
3. **API Key:** `sk-local` (or your gateway key)
4. **Model:** gateway alias
5. Operation: **Message a Model** / Chat Completions

For embeddings, same base URL with an EMBEDDING alias. For long generations, increase the node timeout.

From an n8n Docker container on macOS, use `http://host.docker.internal:11380/v1`.

---

## LiteLLM

Use LiteLLM when you need **virtual keys**, **budgets/rate limits**, **spend logs**, or **Prometheus** in front of Nadir (Team LAN pattern). Nadir stays the MLX backend on loopback.

### Minimal `config.yaml`

```yaml
model_list:
  - model_name: nadir-chat
    litellm_params:
      model: openai/<gateway-alias>
      api_base: http://127.0.0.1:11380/v1
      api_key: sk-local
      timeout: 300

  - model_name: nadir-embed
    litellm_params:
      model: openai/<embedding-alias>
      api_base: http://127.0.0.1:11380/v1
      api_key: sk-local

litellm_settings:
  drop_params: true

general_settings:
  master_key: sk-litellm-master
```

Run:

```bash
pip install 'litellm[proxy]'
litellm --config config.yaml --port 4000
```

Clients then use `http://127.0.0.1:4000` with LiteLLM virtual keys. LiteLLM calls Nadir with the upstream `api_key` / loopback rules above.

### Recommended topology

```text
Team clients  →  LiteLLM :4000 (keys, quotas, /metrics)
                     ↓
              Nadir Gateway :11380 (loopback)
                     ↓
              MLX instances :11400–11500 (loopback)
```

Keep `NADIR_GATEWAY_HOST=127.0.0.1` and only expose LiteLLM on the LAN (with its own TLS/auth).

!!! tip "Prometheus"
    Scrape LiteLLM’s metrics endpoint — not Nadir. Nadir does not ship a first-class `/metrics` product surface.

---

## Cursor / Claude — Nadir MCP (instance control)

Agents can list/start/stop local instances via the **Streamable HTTP MCP** server:

```bash
python manage.py run_mcp   # http://127.0.0.1:11381/mcp
```

Cursor `mcp.json`:

```json
{
  "mcpServers": {
    "nadir": {
      "url": "http://127.0.0.1:11381/mcp",
      "headers": { "Authorization": "Bearer change-me" }
    }
  }
}
```

Full tool list and ASGI mount options: [MCP server](mcp.md).

---

## Quick verification

```bash
# OpenAI shape
curl -s http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<gateway-alias>","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'

# Ollama tags
curl -s http://127.0.0.1:11380/api/tags

# Models list
curl -s http://127.0.0.1:11380/v1/models
```

If auth is enabled:

```bash
curl -s http://127.0.0.1:11380/v1/models -H "Authorization: Bearer $NADIR_GATEWAY_API_KEY"
```

## Related

- [Nadir Gateway](nadir-gateway.md) — routes, security, errors
- [Instance lifecycle](instance-lifecycle.md) — wake / idle / LRU
- [API coverage matrix](nadir-gateway-coverage-matrix.md)
- [Chat tools model matrix](chat-tools-model-matrix.md)
