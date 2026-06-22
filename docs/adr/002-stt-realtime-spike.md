# ADR 002: STT realtime / streaming — spike outcome

**Date:** 2026-06-22  
**Status:** Accepted (no-go for v1)

## Context

MLX-33 asked whether Nadir Gateway and the local Whisper stack can support **OpenAI-style realtime STT** (WebSocket `/v1/realtime`, partial transcripts while audio is still arriving) in addition to batch `POST /v1/audio/transcriptions`.

Today:

- Gateway proxies **multipart file upload** to upstream STT instances (`launch_mode: STT`).
- Upstream uses **mlx-audio Whisper** with batch `model.generate()` (full waveform in memory).
- mlx-audio also exposes `generate_streaming()` (AlignAtt, ~1s partial latency) but with a **Python iterator protocol**, not the OpenAI Realtime API.

## Options considered

| Option | Description | Verdict |
|--------|-------------|---------|
| A | OpenAI Realtime API (WebSocket) on gateway | **Rejected v1** — new protocol, session auth, audio buffer lifecycle; no mlx-audio equivalent |
| B | HTTP chunked partial JSON/SSE from gateway | **Deferred** — possible wrapper around `generate_streaming()`, but no LiteLLM/OpenAI client expects this shape today |
| C | Batch multipart only + richer `response_format` (srt/vtt) + `/v1/audio/translations` | **Accepted v1** (MLX-33) |

## Decision

**v1 ships batch STT only** with:

- `POST /v1/audio/transcriptions` — `json`, `text`, `verbose_json`, `srt`, `vtt`
- `POST /v1/audio/translations` — Whisper `task=translate` (English output)
- Segment timestamps from mlx-audio (`return_timestamps=True`)
- Optional `word_timestamps`, `prompt`, `temperature` forwarded to mlx-audio

**Realtime STT is out of scope** until a dedicated ticket defines:

1. Target client protocol (OpenAI Realtime vs custom SSE)
2. Gateway WebSocket worker or long-lived HTTP stream
3. Back-pressure and max session duration on Apple Silicon

## Input formats (documented)

| Format | Decoder | Notes |
|--------|---------|-------|
| WAV | miniaudio | Recommended for tests |
| MP3 | miniaudio | In-memory decode |
| M4A / AAC | ffmpeg | `brew install ffmpeg` |
| FLAC | miniaudio or ffmpeg | Depends on container |
| OGG / Opus / WebM | ffmpeg | Same requirement |

## Consequences

- LiteLLM `audio_transcription` batch flows work through `:11380` without protocol changes.
- Clients needing **live captions** must poll batch endpoints or wait for a future realtime ADR implementation.
- mlx-audio `generate_streaming()` remains available for a follow-up **MLX-3x** ticket if we define a non-OpenAI streaming contract.

## References

- `orchestrator/stt_server.py`
- `orchestrator/stt_response_formats.py`
- mlx-audio Whisper `generate_streaming()` (AlignAtt)
- MLX-33
