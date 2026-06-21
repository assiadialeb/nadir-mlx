# Nadir Gateway & LiteLLM integration

Single OpenAI-compatible entrypoint on each Mac Studio: **`http://127.0.0.1:11380/v1`**.

Clients (LiteLLM, Open WebUI, curl, SDKs) send the **gateway alias** in the `model` field. The gateway resolves the alias to a **RUNNING** MLX instance and proxies to the correct local backend.

!!! note "Control plane vs data plane"
    - **Django `:8000`** — download models, start/stop instances, benchmarks, UI.
    - **Nadir Gateway `:11380`** — inference only (`/v1/*`).
    - **MLX instances `:11400–11500`** — not exposed to cluster clients; reached via the gateway.

!!! warning "Instances must be RUNNING"
    The gateway does **not** wake stopped instances or load weights on demand (planned for a later sprint). Start each server from the UI (**Serveurs**) before calling the gateway.

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

### 2. Start inference instances

For each model you want to expose:

1. Open **http://127.0.0.1:8000** → **Serveurs**
2. Pick launch mode (Texte, Embeddings, Image, …) and model
3. Note the **Alias gateway** (defaults to the model folder name; editable before start)
4. Wait until status is **Running**

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

| Launch mode | Gateway route | LiteLLM `model_info.mode` (typical) |
|-------------|---------------|-------------------------------------|
| **TEXT** | `POST /v1/chat/completions`, `POST /v1/completions` | chat (default) |
| **MULTIMODAL** | `POST /v1/chat/completions` | chat |
| **EMBEDDING** | `POST /v1/embeddings` | `embedding` |
| **RERANKER** | `POST /v1/rerank` | rerank (provider-specific) |
| **IMAGE** | `POST /v1/images/generations` | image generation |
| **TTS** | `POST /v1/audio/speech` | `audio_speech` |
| **STT** | `POST /v1/audio/transcriptions` | transcription (multipart) |

If you call a route with an alias whose launch mode does not match (e.g. chat on an IMAGE alias), the gateway returns **400** `unsupported_endpoint`.

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

### Embeddings (EMBEDDING)

```bash
curl http://127.0.0.1:11380/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "input": ["Local embeddings on Apple Silicon"]
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

## LiteLLM configuration

Use **one API base** for all local MLX models on a Mac:

| Setting | Value |
|---------|--------|
| **API Base** | `http://127.0.0.1:11380/v1` (host) or `http://host.docker.internal:11380/v1` (LiteLLM in Docker on macOS) |
| **API Key** | Any non-empty string (`sk-local`) — gateway does not validate keys today |
| **Model** | Gateway alias (`openai/<alias>` in LiteLLM) |

### `config.yaml` — multiple modes, one gateway

```yaml
model_list:
  # TEXT — chat
  - model_name: gemma-chat
    litellm_params:
      model: openai/gemma-4-12B-it-4bit
      api_base: http://host.docker.internal:11380/v1
      api_key: sk-local

  # EMBEDDING
  - model_name: local-embed
    litellm_params:
      model: openai/nomic-embed-text
      api_base: http://host.docker.internal:11380/v1
      api_key: sk-local
    model_info:
      mode: embedding

  # RERANKER — OpenAI-compatible /v1/rerank on gateway
  - model_name: local-rerank
    litellm_params:
      model: openai/jina-reranker
      api_base: http://host.docker.internal:11380/v1
      api_key: sk-local
    model_info:
      mode: rerank

  # IMAGE
  - model_name: flux-local
    litellm_params:
      model: openai/flux-schnell
      api_base: http://host.docker.internal:11380/v1
      api_key: sk-local

  # TTS — mode must be audio_speech, not chat
  - model_name: kokoro-tts
    litellm_params:
      model: openai/kokoro-82m
      api_base: http://host.docker.internal:11380/v1
      api_key: sk-local
    model_info:
      mode: audio_speech

  # STT — transcription via OpenAI-compatible API
  - model_name: whisper-local
    litellm_params:
      model: openai/whisper-large-v3
      api_base: http://host.docker.internal:11380/v1
      api_key: sk-local
    model_info:
      mode: audio_transcription
```

Replace alias strings (`gemma-4-12B-it-4bit`, `nomic-embed-text`, …) with the **exact gateway aliases** from the MLX Server UI.

### LiteLLM UI checklist (per model)

1. **Provider**: OpenAI (OpenAI-compatible)
2. **API Base**: `http://host.docker.internal:11380/v1` — must end with `/v1`
3. **LiteLLM model name**: your proxy name (e.g. `gemma-chat`)
4. **Upstream model**: `openai/<gateway-alias>`
5. **Mode**: match the table above (`chat`, `embedding`, `audio_speech`, …)

### Cluster / multi-Mac

LiteLLM sits in front of several Mac Studios. Each Mac runs its own gateway on `:11380`. Register one LiteLLM model entry per `(mac, alias)` with the appropriate `api_base` (VPN IP or internal DNS).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NADIR_GATEWAY_HOST` | `127.0.0.1` | Gateway bind address |
| `NADIR_GATEWAY_PORT` | `11380` | Must stay **outside** `11400–11500` |
| `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS` | `300` | Upstream proxy timeout |

See also [ADR 001 — Nadir Gateway](../adr/001-nadir-gateway.md).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| `{"detail":"Not Found"}` on `/v1/chat/completions` | Old gateway process | Restart: `python manage.py run_gateway` |
| `404 model_not_found` | Unknown alias or typo | Check alias in UI; `GET /v1/models` |
| `503 model_unavailable` | Instance stopped / loading / failed | Start server in UI; wait for **Running** |
| `400 unsupported_endpoint` | Wrong route for launch mode | Use embeddings route for EMBEDDING alias, etc. |
| Empty `/v1/models` | No RUNNING instances | Start at least one server |

## Direct instance ports (legacy)

You can still call `http://127.0.0.1:<114xx>/v1` for debugging. **Prefer the gateway** for LiteLLM and production clients so aliases stay stable and ports stay private.

## Per-mode runbooks (E2E validation)

| Launch mode | Runbook |
|-------------|---------|
| EMBEDDING | [gateway-runbooks/embedding.md](gateway-runbooks/embedding.md) |
| RERANKER | [gateway-runbooks/reranker.md](gateway-runbooks/reranker.md) |
| IMAGE | [gateway-runbooks/image.md](gateway-runbooks/image.md) |
| TTS | *(MLX-29)* |
| STT | *(MLX-30)* |
