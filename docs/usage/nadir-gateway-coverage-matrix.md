# Nadir Gateway — API coverage matrix

Status of capabilities exposed via the gateway (`:11380/v1`) and gaps vs OpenAI / LiteLLM expectations.

!!! note "Living document"
    Use this page when planning new gateway or upstream features. Update it when acceptance criteria change.

Last updated: June 2026 — MLX-37 (VLM vision E2E) done.

## Cross-cutting (all modes)

| Topic | Status |
|-------|--------|
| Alias → RUNNING instance routing | ✅ |
| Aggregated `GET /v1/models` | ✅ |
| In-memory alias cache (avoid DB on every hit) | ✅ MLX-31 (`NADIR_GATEWAY_ROUTE_CACHE_TTL_SECONDS`, default 20s) |
| Wake / idle stop for instances | ❌ next sprint |
| API key auth on gateway | ❌ (LiteLLM can enforce upstream) |
| Multi-worker uvicorn | ❌ single process by default |

## TEXT

| Capability | Status |
|------------|--------|
| `POST /v1/chat/completions` | ✅ |
| `POST /v1/completions` (legacy) | ✅ TEXT only |
| **SSE streaming** (`stream: true`) | ✅ gateway + upstream |
| **Tools / function calling** | ⚠️ MLX-36 — gateway relay ✅; mlx-lm model-dependent ([matrix](chat-tools-model-matrix.md)) |
| **`response_format` json_object / json_schema** | ⚠️ MLX-36 — relay ✅; enforcement upstream best-effort |
| `logprobs`, `n>1` | ⚠️ mlx-lm limits |
| `/v1/completions` on VLM alias | ❌ 400 (by design) |

**Main gaps:** strict JSON schema enforcement, logprobs — not gateway routing.

## MULTIMODAL (VLM)

| Capability | Status |
|------------|--------|
| Chat + **streaming** via `/v1/chat/completions` | ✅ |
| Multimodal messages (`image_url`, base64, local path) | ✅ MLX-37 ([runbook](gateway-runbooks/vlm.md), [matrix](vlm-vision-model-matrix.md)) |
| Multi-image per message | ⚠️ mlx-vlm may keep last image only |
| `/v1/completions` | ❌ 400 |

**Main gap:** multi-image parity upstream; gateway relay is complete.

## EMBEDDING

| Capability | Status |
|------------|--------|
| `POST /v1/embeddings` string + batch | ✅ |
| **Streaming** | ❌ |
| `encoding_format: base64` | ✅ MLX-35 (float32 little-endian) |
| `dimensions` (OpenAI truncation) | ✅ MLX-35 (first N dims) |
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
| `response_format: url` | ✅ MLX-34 (local gateway URL, no CDN) |
| **Streaming** | ❌ |
| edits / variations / inpainting | ❌ v1 — [ADR 003](../adr/003-image-edits-variations.md) (501) |
| Long generation timeout | ⚠️ `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS` (default 300s) |

## TTS (Kokoro)

| Capability | Status |
|------------|--------|
| `POST /v1/audio/speech` | ✅ |
| Formats **wav, mp3** | ✅ |
| OpenAI formats **opus, aac, flac, pcm** | ✅ MLX-32 (opus/aac/flac/pcm; ffmpeg required except wav/pcm) |
| **Audio streaming** | ✅ MLX-32 chunked relay (gateway + optional `stream: true` upstream) |
| OpenAI voice → Kokoro remap | ✅ upstream |
| `instructions` (GPT-4o mini TTS) | ❌ |

## STT (Whisper)

| Capability | Status |
|------------|--------|
| `POST /v1/audio/transcriptions` multipart | ✅ |
| `response_format`: json, text, verbose_json, **srt**, **vtt** | ✅ MLX-33 |
| Input **WAV / MP3** | ✅ |
| **M4A, FLAC, OGG, Opus, WebM** | ✅ with ffmpeg (documented) |
| **Streaming / realtime** | ❌ no-go v1 — [ADR 002](../adr/002-stt-realtime-spike.md) |
| `/v1/audio/translations` | ✅ MLX-33 (Whisper translate → English) |
| Segments + optional `word_timestamps` | ✅ MLX-33 |
| `prompt`, `temperature` (Whisper) | ✅ forwarded to mlx-audio |

## Streaming summary

| Mode | Streaming |
|------|-----------|
| TEXT / VLM chat | ✅ SSE |
| TTS | ✅ chunked binary (MLX-32) |
| Embeddings, rerank, image, STT | ❌ |

## LiteLLM QA priorities

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
4. Rerank / embedding depending on LiteLLM version and `model_info.mode`
5. VLM with images — use base64 or local paths ([vlm-vision-model-matrix.md](vlm-vision-model-matrix.md))
6. Chat **tools** on models without `tool_parser_type` — see [chat-tools-model-matrix.md](chat-tools-model-matrix.md)

## References

- Epic: MLX-17
- Route cache: MLX-31
- STT realtime spike: [ADR 002](../adr/002-stt-realtime-spike.md) (MLX-33)
- Chat tools / JSON: [ADR 004](../adr/004-chat-tools-structured-output.md) (MLX-36)
- VLM vision: [ADR 005](../adr/005-vlm-vision-gateway.md) (MLX-37)
- Integration guide: [nadir-gateway-litellm.md](nadir-gateway-litellm.md)
- E2E runbooks: [gateway-runbooks/](gateway-runbooks/)
- ADR: [001-nadir-gateway.md](../adr/001-nadir-gateway.md)
