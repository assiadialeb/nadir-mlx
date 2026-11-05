# Nadir Gateway — API coverage matrix

Status of capabilities exposed via the gateway (`:11380/v1`) and gaps vs OpenAI API expectations.

!!! note "Living document"
    Use this page when planning new gateway or upstream features. Update it when acceptance criteria change.

Last updated: November 2026 — batch v2 (MLX-133), Anthropic cache usage (MLX-134), TTFT telemetry + global scheduler (MLX-122 / MLX-123).

## Cross-cutting (all modes)

| Topic | Status |
|-------|--------|
| Alias → RUNNING instance routing | ✅ |
| Aggregated `GET /v1/models` | ✅ |
| `GET /v1/models/{id}` | ✅ |
| In-memory alias cache (avoid DB on every hit) | ✅ (`NADIR_GATEWAY_ROUTE_CACHE_TTL_SECONDS`, default 20s) |
| Wake / idle stop for instances | ✅ [runbook](instance-lifecycle.md) |
| Activity heartbeat + LRU eviction grace | ✅ mid-request `last_used_at` (~15s); grace ≥ `2 ×` heartbeat |
| Global upstream cap + batch share | ✅ `NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL` + `X-Nadir-Priority: batch` / `NADIR_GATEWAY_BATCH_MAX_SHARE` |
| Gateway TTFT / cache telemetry | ✅ `NADIR_GATEWAY_TELEMETRY_*`; Servers UI + DB retain; usage-only SSE stripped when client did not opt in |
| Anthropic prompt cache usage fields | ✅ `cache_read_input_tokens`; `input_tokens` = non-cached (JSON + stream `message_delta`) |
| API key auth on gateway | ✅ optional on loopback; required for non-loopback binds via `NADIR_GATEWAY_API_KEY` |
| Trusted Host / Origin guard | ✅ `TrustedHostMiddleware` + reject unknown `Origin`; JSON `Content-Type` required (or absent) |
| Body size limit | ✅ `NADIR_GATEWAY_MAX_BODY_MB` (default 50) |
| Inference worker key forward | ✅ `NADIR_INFERENCE_API_KEY` injected upstream (Nadir workers); mlx-lm/vlm loopback-only |
| `X-Request-Id` on responses | ✅ |
| CORS | ⚠️ opt-in via `NADIR_GATEWAY_CORS_ORIGINS` (no credentials; `*` collapsed safely) |
| Shared upstream httpx client | ✅ |
| `POST /v1/responses` (stateless façade) | ✅ tools/`tool_choice`, stream lifecycle events; no `previous_response_id` / store |
| `POST /v1/messages` (Anthropic) | ✅ JSON + SSE stream; tools / `tool_use`; thinking from `reasoning`; Anthropic error envelope |
| `POST /v1/messages/count_tokens` | ✅ best-effort estimate (no tokenizer load) |
| Ollama façade | ✅ `/`, `/api/version`, `/api/tags`, `/api/ps`, `/api/show`, `/api/chat`, `/api/generate`, `/api/embed` + legacy `/api/embeddings` |
| Multi-worker uvicorn | ❌ single process by default |
| `/v1/files` (`purpose=batch`) | ✅ streaming upload / list / delete / content; `NADIR_BATCH_FILE_MAX_MB` (default 200) |
| `/v1/batches` (chat JSONL) | ✅ create / get / list / cancel; auto worker in `run_gateway` + `process_gateway_batches`; incremental `request_counts` + append output/error JSONL |
| Batch retention purge | ✅ terminal jobs + unreferenced blobs after `NADIR_BATCH_RETENTION_DAYS`; shared `input_file_id` safe across jobs |
| Stale `in_progress` after restart | ✅ first worker cycle reconciles to `failed` (fail-closed); CLI `--reconcile-stale` |
| OpenAI Batch parity | ✅ line `id`, `errors` list shape, separate `error_file`; ⏸️ resume/checkpoint, embeddings batch, webhooks, orphan uploads — [ADR 008](https://github.com/assiadialeb/nadir-mlx/blob/main/adr/008-files-batches-spike.md) |

## TEXT

| Capability | Status |
|------------|--------|
| `POST /v1/chat/completions` | ✅ |
| `POST /v1/completions` (legacy) | ✅ TEXT only |
| **SSE streaming** (`stream: true`) | ✅ gateway + upstream |
| **Tools / function calling** | ⚠️ gateway relay ✅; mlx-lm model-dependent ([matrix](chat-tools-model-matrix.md)) |
| **`response_format` json_object / json_schema** | ⚠️ relay ✅; enforcement upstream best-effort |
| `logprobs`, `n>1` | ⚠️ mlx-lm limits |
| `/v1/completions` on VLM alias | ❌ 400 (by design) |

**Main gaps:** strict JSON schema enforcement, logprobs — not gateway routing.

## MULTIMODAL (VLM)

| Capability | Status |
|------------|--------|
| Chat + **streaming** via `/v1/chat/completions` | ✅ |
| Multimodal messages (`image_url`, base64, local path) | ✅ ([runbook](gateway-runbooks/vlm.md), [matrix](vlm-vision-model-matrix.md)) |
| Multi-image per message | ⚠️ mlx-vlm may keep last image only |
| `/v1/completions` | ❌ 400 |

**Main gap:** multi-image parity upstream; gateway relay is complete.

## EMBEDDING

| Capability | Status |
|------------|--------|
| `POST /v1/embeddings` string + batch | ✅ |
| **Streaming** | ❌ |
| `encoding_format: base64` | ✅ (float32 little-endian) |
| `dimensions` (OpenAI truncation) | ✅ (first N dims) |
| `user`, rate/token limits | ⚠️ partial |

## RERANKER

| Capability | Status |
|------------|--------|
| `POST /v1/rerank` (Jina-like) | ✅ |
| **`model` required** on gateway | ✅ (optional upstream) |
| `return_documents`, `top_n` | ✅ |
| **Streaming** | ❌ |
| Cohere / other API shapes | ❌ |

## IMAGE

| Capability | Status |
|------------|--------|
| `POST /v1/images/generations` | ✅ |
| `b64_json` | ✅ |
| `response_format: url` | ✅ (local gateway URL, no CDN) |
| `GET /v1/images/files/{id}` | ✅ public (unguessable UUID; for `<img src>` without API key) |
| **Streaming** | ❌ |
| edits / variations / inpainting | ⚠️ routes exist; upstream returns **501** Not Implemented |
| Long generation timeout | ⚠️ `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS` (default 300s) |

## TTS (Kokoro)

| Capability | Status |
|------------|--------|
| `POST /v1/audio/speech` | ✅ |
| Formats **wav, mp3** | ✅ |
| OpenAI formats **opus, aac, flac, pcm** | ✅ (opus/aac/flac/pcm; ffmpeg required except wav/pcm) |
| **Audio streaming** | ✅ chunked relay (gateway + optional `stream: true` upstream) |
| OpenAI voice → Kokoro remap | ✅ upstream |
| `instructions` (GPT-4o mini TTS) | ❌ |

## STT (Whisper)

| Capability | Status |
|------------|--------|
| `POST /v1/audio/transcriptions` multipart | ✅ |
| `response_format`: json, text, verbose_json, **srt**, **vtt** | ✅ |
| Input **WAV / MP3** | ✅ |
| **M4A, FLAC, OGG, Opus, WebM** | ✅ with ffmpeg (documented) |
| **Streaming / realtime** | ❌ not supported in v1 |
| `/v1/audio/translations` | ✅ (Whisper translate → English) |
| Segments + optional `word_timestamps` | ✅ |
| `prompt`, `temperature` (Whisper) | ✅ forwarded to mlx-audio |

## Streaming summary

| Mode | Streaming |
|------|-----------|
| TEXT / VLM chat | ✅ SSE |
| TTS | ✅ chunked binary |
| Embeddings, rerank, image, STT | ❌ |

## Client integration QA priorities

**Ready for integration QA:**

- Chat + stream
- Embeddings batch
- Rerank
- Image `b64_json`
- TTS wav/mp3
- STT multipart WAV

**Likely mismatch points:**

1. STT M4A without ffmpeg on the host
2. Image when client expects a **URL**
3. **`on_demand` cold start** — client `timeout` must be ≥ `NADIR_GATEWAY_WAKE_TIMEOUT_SECONDS` (see [instance-lifecycle.md](instance-lifecycle.md))
4. Rerank / embedding — route must match launch mode (`/v1/rerank`, `/v1/embeddings`)
5. VLM with images — use base64 or local paths ([vlm-vision-model-matrix.md](vlm-vision-model-matrix.md))
6. Chat **tools** on models without `tool_parser_type` — see [chat-tools-model-matrix.md](chat-tools-model-matrix.md)

## References

- Integration guide: [nadir-gateway.md](nadir-gateway.md)
- E2E runbooks: see [Nadir Gateway — Per-mode runbooks](nadir-gateway.md#per-mode-runbooks-e2e-validation) (incl. [batch.md](gateway-runbooks/batch.md))
