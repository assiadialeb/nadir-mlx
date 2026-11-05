# Nadir Gateway

Single OpenAI-compatible entrypoint on each Mac Studio: **`http://127.0.0.1:11380/v1`**.

Clients (Open WebUI, curl, OpenAI SDKs, custom scripts) send the **gateway alias** in the `model` field. The gateway resolves the alias to an MLX instance and proxies to the correct local backend. **`on_demand`** instances are woken automatically on first request; **`always_on`** instances must already be running.

!!! note "Control plane vs data plane"
    - **Django `:8000`** — download models, start/stop instances, benchmarks, UI.
    - **Nadir Gateway `:11380`** — inference only (`/v1/*` OpenAI + `/api/*` Ollama façade).
    - **MLX instances `:11400–11500`** — not exposed to cluster clients; reached via the gateway.

!!! tip "Ollama-compatible clients (Open WebUI, Continue, Raycast)"
    Point the client **Ollama base URL** at the gateway root: `http://127.0.0.1:11380` (not native Ollama `:11434`). Use the gateway **alias** as the model name. Probes (`GET /`, `/api/version`, `/api/tags`, `/api/ps`, `/api/show`) and chat/generate/embed are served on this port.

!!! note "Lifecycle modes"
    - **`always_on`** (default) — instance must be **Running** before inference; gateway returns `503 model_unavailable` if stopped.
    - **`on_demand`** — gateway **wakes** a stopped instance on first request (cold start). Set client request timeouts ≥ `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` (default 300s). See [instance-lifecycle.md](instance-lifecycle.md).

## Quick start

### 1. Start Django and the gateway

Recommended (one terminal):

```bash
source venv/bin/activate
python manage.py serve
# optional: python manage.py serve --with-mcp
```

Or use **two terminals** (or `tmux` / `screen`):

```bash
# Terminal 1 — control plane
source venv/bin/activate
python manage.py runserver
```

```bash
# Terminal 2 — data plane
source venv/bin/activate
python manage.py run_gateway
# equivalent: python -m orchestrator.gateway
```

Health check:

```bash
curl http://127.0.0.1:11380/health
```

### 2. Configure inference instances

For each model you want to expose:

1. Open **http://127.0.0.1:8000** → **Servers**
2. Pick launch mode (Text, Embeddings, Image, …) and model
3. Note the **Gateway alias** (defaults to the model folder name; editable before start)
4. Set **lifecycle** in server config: `always_on` (stay loaded) or `on_demand` (idle offload)
5. For `always_on`, wait until status is **Running** before the first request. For `on_demand`, you may leave the instance **Stopped** — the gateway wakes it on first traffic.

### 3. Discover aliases

```bash
curl http://127.0.0.1:11380/v1/models
```

Example response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "gemma-4-12B-it-4bit",
      "object": "model",
      "created": 1782083900,
      "owned_by": "nadir",
      "metadata": { "launch_mode": "TEXT" }
    }
  ]
}
```

Only **RUNNING** instances appear. Internal ports are never returned.

## Gateway routes by launch mode

| Launch mode | Gateway route | Notes |
|-------------|---------------|-------|
| **TEXT** | `POST /v1/chat/completions`, `POST /v1/completions` | default chat |
| **MULTIMODAL** | `POST /v1/chat/completions` | vision in messages |
| **EMBEDDING** | `POST /v1/embeddings` | batch input |
| **RERANKER** | `POST /v1/rerank` | OpenAI-compatible rerank |
| **IMAGE** | `POST /v1/images/generations` | `b64_json` or URL |
| **TTS** | `POST /v1/audio/speech` | binary audio response |
| **STT** | `POST /v1/audio/transcriptions`, `POST /v1/audio/translations` | multipart upload |

If you call a route with an alias whose launch mode does not match (e.g. chat on an IMAGE alias), the gateway returns **400** `unsupported_endpoint`.

## Client configuration

| Setting | Value |
|---------|--------|
| **API Base** | `http://127.0.0.1:11380/v1` (host) or `http://host.docker.internal:11380/v1` (client in Docker on macOS) |
| **API Key** | Optional on loopback; required when `NADIR_GATEWAY_API_KEY` is set (`Authorization: Bearer` or `X-API-Key`). Placeholders like `sk-local` work only if no key is configured. |
| **Model** | Gateway alias exactly as shown in the UI or `GET /v1/models` |

Copy-paste setups for Open WebUI, Continue, Cursor, Claude Code, n8n, and LiteLLM (team keys / metrics): **[Integration recipes](integration-recipes.md)**.

Example with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11380/v1",
    api_key="sk-local",
)

response = client.chat.completions.create(
    model="gemma-4-12B-it-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=64,
)
print(response.choices[0].message.content)
```

For **`on_demand`** models, set client timeouts high enough for cold starts (180–300s for large VLMs).

### Multi-Mac / cluster

Each Mac Studio runs its own gateway on `:11380`. Point clients at the appropriate host (VPN IP or internal DNS) and use the gateway alias registered on that machine.

## curl examples (via gateway)

Replace `<alias>` with the value from the UI or `GET /v1/models`.

### Chat (TEXT / MULTIMODAL)

```bash
curl http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

Streaming:

```bash
curl http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "stream": true,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Vision (MULTIMODAL alias, base64 image):

```bash
curl http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<vlm-alias>",
    "max_tokens": 128,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64>"}}
      ]
    }]
  }'
```

See [gateway-runbooks/vlm.md](gateway-runbooks/vlm.md) for full E2E steps.

### Embeddings (EMBEDDING)

Supports `encoding_format` (`float` default, `base64`) and optional `dimensions` (truncate to first N floats).

```bash
curl http://127.0.0.1:11380/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "input": ["Local embeddings on Apple Silicon"],
    "encoding_format": "base64",
    "dimensions": 256
  }'
```

### Rerank (RERANKER)

```bash
curl http://127.0.0.1:11380/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "query": "What is Python?",
    "documents": [
      "Python is a programming language.",
      "The weather is sunny today."
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

### Image generation (IMAGE)

```bash
curl http://127.0.0.1:11380/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "prompt": "A red apple on a wooden table",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

### Text-to-speech (TTS)

```bash
curl http://127.0.0.1:11380/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "input": "Hello from Nadir gateway.",
    "voice": "alloy"
  }' \
  --output speech.wav
```

OpenAI voice names (`alloy`, `nova`, …) are mapped to Kokoro voices server-side.

### Speech-to-text (STT)

```bash
curl http://127.0.0.1:11380/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=<alias>" \
  -F "response_format=json"
```

Subtitles (`srt` / `vtt`) and translation (audio → English):

```bash
curl http://127.0.0.1:11380/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=<alias>" \
  -F "response_format=srt"

curl http://127.0.0.1:11380/v1/audio/translations \
  -F "file=@sample-fr.wav" \
  -F "model=<alias>" \
  -F "response_format=json"
```

!!! note "Input formats"
    WAV and MP3 decode in memory. M4A, FLAC, OGG, Opus, and WebM require **ffmpeg** on the MLX host. Realtime WebSocket STT is not supported in v1.

## Prompt cache (prefix KV / TTFT)

mlx-lm keeps an LRU of **prefix KV caches** (`--prompt-cache-size` / `--prompt-cache-bytes`). Agent loops that send the same system prompt + tool definitions every turn reuse that prefix and cut **TTFT**.

| Concern | Guidance |
|---------|----------|
| Instance knobs | TEXT Advanced JSON: `prompt_cache_size` (RAM-aware default entry count). Optional `prompt_cache_bytes` only when you want a hard cap — omit it for agent workloads. See [server-config-reference](server-config-reference.md). |
| Tool order | Clients must keep the **tools array order** identical across turns. |
| Façades | Anthropic / OpenAI / Responses / Ollama canonicalize nested JSON object keys (schemas + tool arguments) so translation does not reshuffle the shared prefix. |
| Observability | mlx-lm reports `usage.prompt_tokens_details.cached_tokens`. Gateway telemetry (MLX-123) stores it; Servers UI shows per-request Cache and a recent **hit rate**. |
| Anthropic clients | `/v1/messages` maps `cached_tokens` → `usage.cache_read_input_tokens` and sets `input_tokens` to non-cached prompt tokens (Claude Code / SDK parity). |
| Manual cache replay | `NADIR_GATEWAY_MODEL=<alias> python scripts/gateway_anthropic_cache_replay.py` — three stable-prefix turns; expect high `cache_read_input_tokens` on turns 2–3. |

!!! warning "What breaks the cache"
    Per-turn noise in the shared prefix (timestamps in the system prompt, reordered tools, unstable JSON key order in tool schemas) forces a full prefill. Keep system+tools byte-stable; put turn-specific context only in user/assistant messages.

## Thinking / reasoning control (TTFT)

mlx-lm accepts per-request `chat_template_kwargs` (notably `enable_thinking`). Nadir maps client fields so agents can disable long “thinking” traces that inflate **visible TTFT**:

| Client API | Client field | Upstream `chat_template_kwargs` |
|------------|--------------|----------------------------------|
| OpenAI chat `/v1/chat/completions` | `reasoning_effort`: `none` / `minimal` | `enable_thinking: false` |
| OpenAI chat | `reasoning_effort`: `low` / `medium` / `high` | `enable_thinking: true` + `thinking_budget` |
| Responses `/v1/responses` | `reasoning.effort` or `reasoning_effort` | same mapping |
| Anthropic `/v1/messages` | `thinking` absent or `type: disabled` | `enable_thinking: false` |
| Anthropic | `thinking.type: enabled` (+ optional `budget_tokens`) | `enable_thinking: true` (+ budget) |
| Ollama `/api/chat` | `think` absent or `false` | `enable_thinking: false` |
| Ollama `/api/chat` | `think: true` | `enable_thinking: true` (façade also emits `thinking`) |

Clients may still send `chat_template_kwargs` directly; façade mappings override colliding keys.

## Security

The gateway is designed for **local-first** use. Browser CSRF and DNS-rebinding protections are always on.

| Control | Behaviour |
|---------|-----------|
| **Trusted Host** | `Host` must be `localhost`, `127.0.0.1`, `[::1]`, or a value from `NADIR_GATEWAY_ALLOWED_HOSTS` / `NADIR_GATEWAY_PUBLIC_BASE_URL`. Unknown hosts → **400**. |
| **Origin guard** | If the request has an `Origin` header, it must appear in `NADIR_GATEWAY_CORS_ORIGINS`. Unknown origin → **403**. CLI/SDK clients that omit `Origin` are unaffected. |
| **JSON Content-Type** | JSON POSTs must use `application/json` (or omit Content-Type). `text/plain` and other types → **415** (blocks browser “simple” cross-site POSTs). |
| **API key** | Optional on loopback. **Required** when `NADIR_GATEWAY_HOST` is not loopback — startup refuses to bind otherwise. |
| **Body size** | Requests with `Content-Length` above `NADIR_GATEWAY_MAX_BODY_MB` (default 50) → **413**. |
| **OpenAPI `/docs`** | Exposed only when the gateway binds to loopback. |
| **Image files** | `GET /v1/images/files/{uuid}` is public (unguessable ids) so `<img src>` works without an API key. |

!!! warning "LAN exposure"
    Prefer `NADIR_GATEWAY_HOST=127.0.0.1`. If you bind to a LAN address, set `NADIR_GATEWAY_API_KEY` and add the host to `NADIR_GATEWAY_ALLOWED_HOSTS`. mlx-lm / mlx-vlm still have **no auth** — keep instance bind on `127.0.0.1` and expose clients only through the gateway.

!!! tip "Inference worker key"
    Set `NADIR_INFERENCE_API_KEY` to protect Nadir-owned workers (image / STT / embedding / TTS). The gateway injects `Authorization: Bearer …` on upstream calls and health probes. mlx-lm and mlx-vlm are protected only by loopback bind.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NADIR_GATEWAY_HOST` | `127.0.0.1` | Gateway bind address |
| `NADIR_GATEWAY_PORT` | `11380` | Must stay **outside** `11400–11500` |
| `NADIR_GATEWAY_API_KEY` | — | Required for non-loopback binds (`Authorization: Bearer` or `X-API-Key`) |
| `NADIR_INFERENCE_API_KEY` | — | Optional; injected upstream to Nadir workers |
| `NADIR_GATEWAY_CORS_ORIGINS` | — | Allowed browser origins; empty rejects any `Origin` |
| `NADIR_GATEWAY_ALLOWED_HOSTS` | — | Extra `Host` values for TrustedHostMiddleware |
| `NADIR_GATEWAY_MAX_BODY_MB` | `50` | Max request body via Content-Length |
| `NADIR_GATEWAY_PUBLIC_BASE_URL` | `http://127.0.0.1:11380` | Base URL for generated image links |
| `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS` | `300` | Upstream proxy timeout |
| `NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM` | `16` | Max parallel upstream requests per instance (`0` = unlimited) |
| `NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL` | `8` | Cross-instance GPU cap (`0` = unlimited). Default 8 on 48 GB class Macs so interactive traffic queues ahead of batch under load |
| `NADIR_GATEWAY_BATCH_MAX_SHARE` | `0.5` | Max share of the **global** pool for `X-Nadir-Priority: batch` (interactive keeps the rest). Per-instance caps are not batch-limited unless the global cap binds first. |
| `NADIR_BATCH_FILE_MAX_MB` | `200` | Max upload for `POST /v1/files` (multipart JSONL) |
| `NADIR_BATCH_FILES_DIR` | `data/batch_files` | Local blob store for batch input/output |
| `NADIR_BATCH_LINE_TIMEOUT_SECONDS` | `600` | Per-line timeout when the batch worker calls `/v1/chat/completions` |
| `NADIR_GATEWAY_BATCH_WORKER_ENABLED` | `true` | Daemon batch worker inside `run_gateway` / `serve` |
| `NADIR_GATEWAY_BATCH_POLL_SECONDS` | `15` | Poll interval for validating batch jobs |
| `NADIR_BATCH_RETENTION_DAYS` | `30` | Purge terminal batches and blobs (worker cycle) |
| `NADIR_GATEWAY_QUEUE_TIMEOUT_SECONDS` | `300` | Max wait in gateway queue when all slots are busy |
| `NADIR_GATEWAY_ROUTE_CACHE_TTL_SECONDS` | `20` | In-memory alias / models cache TTL |
| `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` | `300` | Max wait for `on_demand` wake + health |
| `NADIR_GATEWAY_WAKE_POLL_INTERVAL_SECONDS` | `1` | Health poll interval during wake |
| `NADIR_GATEWAY_ACTIVITY_HEARTBEAT_SECONDS` | `15` | Mid-request `last_used_at` refresh (SSE + long JSON) |
| `NADIR_GATEWAY_TELEMETRY_ENABLED` | `true` | Per-request TTFT / prefill / cache metrics (logs + Servers UI) for OpenAI chat, Anthropic, Responses, and Ollama |
| `NADIR_GATEWAY_TELEMETRY_FORCE_USAGE` | `true` | Force upstream `stream_options.include_usage` for token/cache counts; usage-only SSE chunks are **stripped** from OpenAI clients that did not opt in (regression C1) |
| `NADIR_GATEWAY_TELEMETRY_RETAIN` | `100` | Max recent timing rows kept in the local DB (pruned in batches) |
| `NADIR_IDLE_OFFLOAD_ENABLED` | `true` | Stop idle `on_demand` instances in background |
| `NADIR_IDLE_CHECK_INTERVAL_SECONDS` | `60` | Idle watcher evaluation interval |
| `NADIR_LRU_EVICTION_ENABLED` | `true` | Evict LRU `on_demand` under memory pressure before load |
| `NADIR_LRU_EVICTION_GRACE_SECONDS` | `60` | Min idle age before eviction (floored to `2 ×` heartbeat) |
| `NADIR_MEMORY_RESERVE_GB` | `4` | Free-memory headroom when deciding to evict |

!!! tip "Upstream concurrency queue"
    When more clients hit the gateway than MLX can serve in parallel, excess requests **wait in a queue** (like Ollama) instead of failing immediately. Tune `NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM` for your hardware, or set **Max concurrent upstream requests** per instance in the server form. Use `0` globally or per instance to disable the cap (legacy pass-through).

!!! note "GPU scheduling and priority"
    Every upstream request acquires **global** and **instance** capacity in one step (`NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL`, default **8**), under one `NADIR_GATEWAY_QUEUE_TIMEOUT_SECONDS` budget. A waiter blocked on a full instance never holds a global slot, so other models keep moving.

    Freed slots go to **interactive** requests first. Batch traffic is further capped by `NADIR_GATEWAY_BATCH_MAX_SHARE` (default **0.5**) on the **global** pool only — a single instance with a low `decode_concurrency` can still use all of its own slots for batch when the global cap is not saturated. Background clients (report generation, bulk embeddings, evaluations) should send `X-Nadir-Priority: batch`. Any other value, or no header, is interactive.

    !!! tip "Proxies and LiteLLM"
        Many OpenAI-compatible proxies (including LiteLLM) **do not forward custom headers** by default. If `X-Nadir-Priority` is stripped, every request is treated as interactive — safe, but batch isolation will not apply until the proxy is configured to pass the header through.

    ```bash
    curl -s http://127.0.0.1:11380/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "X-Nadir-Priority: batch" \
      -d '{"model": "qwen-chat", "messages": [{"role": "user", "content": "Summarize"}]}'
    ```

    For TEXT instances the per-instance cap is also bounded by `decode_concurrency` (Advanced JSON) so requests never queue inside mlx-lm, out of reach of the priority order.

!!! tip "Route cache"
    The gateway caches alias → instance resolution and `GET /v1/models` in memory for `NADIR_GATEWAY_ROUTE_CACHE_TTL_SECONDS` (default 20s). After starting or stopping an instance, new routes may take up to one TTL window to appear. Lower the TTL in dev if you need faster feedback.

See [instance-lifecycle.md](instance-lifecycle.md) for wake, idle offload, LRU eviction, and activity heartbeats. Full variable list: [Configuration](../configuration.md).

## Error codes

Gateway and proxy errors use an OpenAI-style envelope:

```json
{
  "error": {
    "message": "Human-readable description",
    "type": "<code>",
    "code": "<code>"
  }
}
```

`type` and `code` are identical for routing errors. Upstream proxy errors may use upstream `type` with gateway `code` where applicable.

### Routing and lifecycle (`GatewayRouteError`)

| HTTP | `code` | When |
|------|--------|------|
| 400 | `invalid_model` | Missing or empty `model` field |
| 400 | `unsupported_endpoint` | Route does not match instance launch mode (e.g. chat on IMAGE alias) |
| 404 | `model_not_found` | No instance registered for this alias |
| 503 | `model_unavailable` | Instance STOPPED, LOADING, FAILED, or not ready (`always_on`) |
| 503 | `model_waking_timeout` | `on_demand` instance did not become healthy within `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` |
| 500 | `unsupported_launch_mode` | Internal: launch mode missing from route table |

### Proxy and queue (`http_proxy` / `mode_proxy`)

| HTTP | `code` | When |
|------|--------|------|
| 502 | `bad_gateway` | Connection error to MLX instance |
| 503 | `upstream_queue_timeout` | Waited longer than `NADIR_GATEWAY_QUEUE_TIMEOUT_SECONDS` for a concurrency slot |
| 504 | `gateway_timeout` | Upstream request exceeded `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS` |

Upstream MLX errors are forwarded when possible (`type` may be `upstream_error`).

### Request validation and security

| HTTP | `type` / `code` | When |
|------|-----------------|------|
| 400 | `invalid_request` | Malformed JSON body (not an object or parse error) |
| 400 | — | Invalid `Host` (TrustedHostMiddleware / DNS-rebinding guard) |
| 401 | `invalid_api_key` | Missing/invalid API key when `NADIR_GATEWAY_API_KEY` is set |
| 403 | `forbidden` | `Origin` present but not in `NADIR_GATEWAY_CORS_ORIGINS` |
| 413 | `payload_too_large` | `Content-Length` exceeds `NADIR_GATEWAY_MAX_BODY_MB` |
| 415 | `unsupported_media_type` | JSON route with non-JSON Content-Type (e.g. `text/plain`) |

### Contract coverage

OpenAPI contract tests assert error shape for `model_not_found` (404) and `model_unavailable` (503). Run locally:

```bash
pytest -m contract orchestrator/tests/contracts -q
```

## Local batch jobs (OpenAI-compatible)

Privacy-first overnight eval: upload JSONL, create a batch, let the gateway worker drain validating jobs (or run the management command manually).

1. `POST /v1/files` with `purpose=batch` and a JSONL file (streamed to disk; OpenAI line shape: `custom_id`, `url`, `body`).
2. `POST /v1/batches` with `input_file_id` and `endpoint: /v1/chat/completions`.
3. With `run_gateway` / `serve`, the **batch worker** polls every `NADIR_GATEWAY_BATCH_POLL_SECONDS` (default 15s) and holds the same lock as `process_gateway_batches`. On the **first** cycle after gateway start, any leftover `in_progress` job is reconciled to `failed` (orphan after crash/restart). Optional manual run: `python manage.py process_gateway_batches --max-jobs 1` (`--reconcile-stale` when the auto worker is off).
4. Poll `GET /v1/batches/{id}` — `request_counts` updates while `in_progress`; output/error JSONL files grow line-by-line (OpenAI line `id`, list-shaped `errors`, separate `error_file`).
5. Download `GET /v1/files/{output_file_id}/content` and optional `error_file_id`.

Fan-out uses `X-Nadir-Priority: batch` and shares the **global** concurrency scheduler with interactive traffic (`NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL`, `NADIR_GATEWAY_BATCH_MAX_SHARE`). Cancel sets `cancelling` and waits for in-flight lines (up to `NADIR_BATCH_LINE_TIMEOUT_SECONDS` each). Terminal jobs older than `NADIR_BATCH_RETENTION_DAYS` are purged by the worker; **shared** input/output blobs are kept until no batch job references them.

Cancel while `validating` stops immediately. Cancel while `in_progress` sets `cancelling`; the worker records any line that already finished, then stops scheduling new lines. In-flight upstream calls still run until they complete or hit `NADIR_BATCH_LINE_TIMEOUT_SECONDS` (default 600s).

E2E validation: [Batch runbook](gateway-runbooks/batch.md). Copy-paste client setup: [Integration recipes — Overnight bulk jobs](integration-recipes.md#overnight-bulk-jobs-openai-batch-api).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| `{"detail":"Not Found"}` on `/v1/chat/completions` | Old gateway process | Restart: `python manage.py run_gateway` |
| `404 model_not_found` | Unknown alias or typo | Check alias in UI; `GET /v1/models` |
| `503 model_unavailable` | `always_on` instance stopped / loading / failed | Start server in UI; wait for **Running** |
| `503 model_waking_timeout` | `on_demand` cold start exceeded wake timeout | Increase `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` and client timeout |
| `400 unsupported_endpoint` | Wrong route for launch mode | Use embeddings route for EMBEDDING alias, etc. |
| `502 bad_gateway` | MLX process down or port closed | Check instance status and logs |
| `503 upstream_queue_timeout` | Too many parallel gateway requests | Raise the per-instance or global concurrency cap, or the queue timeout; `batch` requests wait behind every queued interactive one |
| `504 gateway_timeout` | Inference slower than proxy timeout | Increase `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS` |
| `400` on every request with custom Host | Trusted Host reject | Add host to `NADIR_GATEWAY_ALLOWED_HOSTS` or use `localhost` / `127.0.0.1` |
| `403` from a browser app | Origin not allowlisted | Add the page origin to `NADIR_GATEWAY_CORS_ORIGINS` |
| `415` from a browser form/fetch | Wrong Content-Type | Send `Content-Type: application/json` |
| Empty `/v1/models` | No instances registered | Create at least one server (stopped `on_demand` aliases still appear) |
| Gateway refuses to start on LAN IP | No API key | Set `NADIR_GATEWAY_API_KEY` or bind to `127.0.0.1` |
| Batch `failed` — input missing on disk | Retention purge vs shared file (fixed in v1.1+) or manual delete | Re-upload JSONL; avoid deleting `input_file_id` while a job still references it |
| Batch stuck `in_progress` after gateway restart | Orphan job before auto-reconcile | Restart gateway (first worker cycle reconciles) or `process_gateway_batches --reconcile-stale` |

## Direct instance ports (debugging)

You can still call `http://127.0.0.1:<114xx>/v1` for debugging. **Prefer the gateway** for production clients so aliases stay stable and ports stay private.

## Per-mode runbooks (E2E validation)

| Launch mode | Runbook |
|-------------|---------|
| TEXT / MULTIMODAL | [gateway-runbooks/chat.md](gateway-runbooks/chat.md) |
| MULTIMODAL (vision) | [gateway-runbooks/vlm.md](gateway-runbooks/vlm.md) |
| MULTIMODAL (Gemma 4 MTP) | [gateway-runbooks/gemma4-mtp.md](gateway-runbooks/gemma4-mtp.md) |
| EMBEDDING | [gateway-runbooks/embedding.md](gateway-runbooks/embedding.md) |
| RERANKER | [gateway-runbooks/reranker.md](gateway-runbooks/reranker.md) |
| Local RAG (embed + rerank + chat) | [gateway-runbooks/rag.md](gateway-runbooks/rag.md) |
| IMAGE | [gateway-runbooks/image.md](gateway-runbooks/image.md) |
| TTS | [gateway-runbooks/tts.md](gateway-runbooks/tts.md) |
| STT | [gateway-runbooks/stt.md](gateway-runbooks/stt.md) |
| Batch (OpenAI Files + Batches) | [gateway-runbooks/batch.md](gateway-runbooks/batch.md) |

See also the [API coverage matrix](nadir-gateway-coverage-matrix.md) for gaps vs the OpenAI API (streaming, audio formats, etc.).
