# Nadir Gateway

Single OpenAI-compatible entrypoint on each Mac Studio: **`http://127.0.0.1:11380/v1`**.

Clients (Open WebUI, curl, OpenAI SDKs, custom scripts) send the **gateway alias** in the `model` field. The gateway resolves the alias to an MLX instance and proxies to the correct local backend. **`on_demand`** instances are woken automatically on first request; **`always_on`** instances must already be running.

!!! note "Control plane vs data plane"
    - **Django `:8000`** — download models, start/stop instances, benchmarks, UI.
    - **Nadir Gateway `:11380`** — inference only (`/v1/*`).
    - **MLX instances `:11400–11500`** — not exposed to cluster clients; reached via the gateway.

!!! note "Lifecycle modes"
    - **`always_on`** (default) — instance must be **Running** before inference; gateway returns `503 model_unavailable` if stopped.
    - **`on_demand`** — gateway **wakes** a stopped instance on first request (cold start). Set client request timeouts ≥ `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` (default 300s). See [instance-lifecycle.md](instance-lifecycle.md).

## Quick start

### 1. Start Django and the gateway

Use **two terminals** (or `tmux` / `screen`):

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
| **API Key** | Any non-empty string (`sk-local`) — gateway does not validate keys today |
| **Model** | Gateway alias exactly as shown in the UI or `GET /v1/models` |

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

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NADIR_GATEWAY_HOST` | `127.0.0.1` | Gateway bind address |
| `NADIR_GATEWAY_PORT` | `11380` | Must stay **outside** `11400–11500` |
| `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS` | `300` | Upstream proxy timeout |
| `NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM` | `16` | Max parallel upstream requests per instance (`0` = unlimited) |
| `NADIR_GATEWAY_QUEUE_TIMEOUT_SECONDS` | `300` | Max wait in gateway queue when all slots are busy |
| `NADIR_GATEWAY_ROUTE_CACHE_TTL_SECONDS` | `20` | In-memory alias / models cache TTL |
| `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` | `300` | Max wait for `on_demand` wake + health |
| `NADIR_GATEWAY_WAKE_POLL_INTERVAL_SECONDS` | `1` | Health poll interval during wake |
| `NADIR_IDLE_OFFLOAD_ENABLED` | `true` | Stop idle `on_demand` instances in background |
| `NADIR_IDLE_CHECK_INTERVAL_SECONDS` | `60` | Idle watcher evaluation interval |

!!! tip "Upstream concurrency queue"
    When more clients hit the gateway than MLX can serve in parallel, excess requests **wait in a queue** (like Ollama) instead of failing immediately. Tune `NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM` for your hardware, or set **Max concurrent upstream requests** per instance in the server form. Use `0` globally or per instance to disable the cap (legacy pass-through).

!!! tip "Route cache"
    The gateway caches alias → instance resolution and `GET /v1/models` in memory for `NADIR_GATEWAY_ROUTE_CACHE_TTL_SECONDS` (default 20s). After starting or stopping an instance, new routes may take up to one TTL window to appear. Lower the TTL in dev if you need faster feedback.

See [instance-lifecycle.md](instance-lifecycle.md) for wake and idle offload behaviour.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| `{"detail":"Not Found"}` on `/v1/chat/completions` | Old gateway process | Restart: `python manage.py run_gateway` |
| `404 model_not_found` | Unknown alias or typo | Check alias in UI; `GET /v1/models` |
| `503 model_unavailable` | `always_on` instance stopped / loading / failed | Start server in UI; wait for **Running** |
| `503 model_waking_timeout` | `on_demand` cold start exceeded wake timeout | Increase `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` and client timeout |
| `400 unsupported_endpoint` | Wrong route for launch mode | Use embeddings route for EMBEDDING alias, etc. |
| Empty `/v1/models` | No instances registered | Create at least one server (stopped `on_demand` aliases still appear) |

## Direct instance ports (debugging)

You can still call `http://127.0.0.1:<114xx>/v1` for debugging. **Prefer the gateway** for production clients so aliases stay stable and ports stay private.

## Per-mode runbooks (E2E validation)

| Launch mode | Runbook |
|-------------|---------|
| TEXT / MULTIMODAL | [gateway-runbooks/chat.md](gateway-runbooks/chat.md) |
| MULTIMODAL (vision) | [gateway-runbooks/vlm.md](gateway-runbooks/vlm.md) |
| EMBEDDING | [gateway-runbooks/embedding.md](gateway-runbooks/embedding.md) |
| RERANKER | [gateway-runbooks/reranker.md](gateway-runbooks/reranker.md) |
| IMAGE | [gateway-runbooks/image.md](gateway-runbooks/image.md) |
| TTS | [gateway-runbooks/tts.md](gateway-runbooks/tts.md) |
| STT | [gateway-runbooks/stt.md](gateway-runbooks/stt.md) |

See also the [API coverage matrix](nadir-gateway-coverage-matrix.md) for gaps vs the OpenAI API (streaming, audio formats, etc.).
