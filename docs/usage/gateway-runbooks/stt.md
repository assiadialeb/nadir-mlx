# Runbook — Gateway STT

Validate `POST /v1/audio/transcriptions` and `POST /v1/audio/translations` (multipart) through Nadir Gateway (`:11380`) for a **RUNNING** Whisper STT instance.

## Prerequisites

- Gateway alias e.g. **`whispers`** (`launch_mode: STT`)
- Instance on port **11445**
- Sample audio file (WAV recommended)
- **ffmpeg** on the host for M4A / FLAC / OGG / Opus / WebM uploads (`brew install ffmpeg`)

!!! note "Restart gateway"
    Multipart relay requires gateway code that detects Starlette `UploadFile` correctly. After updating Nadir MLX, run `python manage.py run_gateway` again.

## 1. Discovery

```bash
curl -s http://127.0.0.1:11380/v1/models | python3 -m json.tool
```

## 2. Create a test WAV (via TTS)

```bash
curl -s -o /tmp/nadir-stt-sample.wav http://127.0.0.1:11380/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "The quick brown fox jumps over the lazy dog", "response_format": "wav"}'
```

## 3. Transcribe via gateway (JSON)

```bash
curl -s -w "\nHTTP:%{http_code}\n" http://127.0.0.1:11380/v1/audio/transcriptions \
  -F "file=@/tmp/nadir-stt-sample.wav" \
  -F "model=whispers" \
  -F "response_format=json"
```

**Expected:** HTTP **200**, JSON `{"text": "…"}` (quality depends on model / audio).

Plain text output:

```bash
curl -s http://127.0.0.1:11380/v1/audio/transcriptions \
  -F "file=@/tmp/nadir-stt-sample.wav" \
  -F "model=whispers" \
  -F "response_format=text"
```

## 4. Subtitles (SRT / VTT)

```bash
curl -s http://127.0.0.1:11380/v1/audio/transcriptions \
  -F "file=@/tmp/nadir-stt-sample.wav" \
  -F "model=whispers" \
  -F "response_format=srt"

curl -s http://127.0.0.1:11380/v1/audio/transcriptions \
  -F "file=@/tmp/nadir-stt-sample.wav" \
  -F "model=whispers" \
  -F "response_format=vtt"
```

Verbose JSON with segment timestamps:

```bash
curl -s http://127.0.0.1:11380/v1/audio/transcriptions \
  -F "file=@/tmp/nadir-stt-sample.wav" \
  -F "model=whispers" \
  -F "response_format=verbose_json"
```

Optional word-level timestamps (mlx-audio alignment):

```bash
curl -s http://127.0.0.1:11380/v1/audio/transcriptions \
  -F "file=@/tmp/nadir-stt-sample.wav" \
  -F "model=whispers" \
  -F "response_format=verbose_json" \
  -F "word_timestamps=true"
```

## 5. Translate to English

Whisper `task=translate` via OpenAI-compatible translations route:

```bash
curl -s http://127.0.0.1:11380/v1/audio/translations \
  -F "file=@/tmp/nadir-stt-sample-fr.wav" \
  -F "model=whispers" \
  -F "response_format=json"
```

Output is English text regardless of source language.

## 6. Input formats

| Format | Requirement |
|--------|-------------|
| WAV, MP3 | Works out of the box (miniaudio) |
| M4A, AAC, OGG, Opus, WebM, FLAC | Requires **ffmpeg** on the MLX host |

## 7. Direct upstream (debug)

```bash
curl -s http://127.0.0.1:11445/v1/audio/transcriptions \
  -F "file=@/tmp/nadir-stt-sample.wav" \
  -F "model=whispers" \
  -F "response_format=json"
```

Gateway should return the same shape when routing works.

## 8. Wrong route

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "whispers", "messages": [{"role": "user", "content": "Hi"}]}'
```

**Expected:** HTTP **400**.

## Realtime STT (SSE)

OpenAI Realtime / WebSocket STT is **not supported** in v1. Partial transcripts are available via **Server-Sent Events** on a dedicated route:

```bash
curl -N http://127.0.0.1:11380/v1/audio/transcriptions/stream \
  -F "file=@/tmp/nadir-stt-sample.wav" \
  -F "model=whispers"
```

**Event contract:**

| Event | Payload |
|-------|---------|
| `transcript` | `{"object":"stt.transcript.delta\|completed","text":"…","is_final":false\|true,"task":"transcribe"}` |
| `done` | `{"object":"stt.transcript.done"}` |
| `error` | `{"object":"stt.transcript.error","message":"…"}` |

When mlx-audio exposes `generate_streaming`, deltas arrive as chunks are decoded; otherwise the server emits one completed transcript after batch `generate()`.

Gateway and upstream STT share the same multipart fields (`language`, `chunk_duration`, `word_timestamps`, `prompt`, `temperature`). `response_format` is ignored on the stream route (JSON-shaped events only).

Direct upstream:

```bash
curl -N http://127.0.0.1:11445/v1/audio/transcriptions/stream \
  -F "file=@/tmp/nadir-stt-sample.wav" \
  -F "model=whispers"
```

## Batch-only note

For SRT/VTT/plain text, keep using `POST /v1/audio/transcriptions` (non-stream).

## Troubleshooting

| HTTP | Cause | Action |
|------|--------|--------|
| 422 on `file` | Old gateway multipart bug | Restart gateway with latest code |
| 400 | Empty file | Check `-F file=@path` |
| 400 | Unsupported `response_format` | Use `json`, `text`, `verbose_json`, `srt`, or `vtt` |
| 400 | ffmpeg missing | Install ffmpeg or upload WAV/MP3 |
| 503 | STT not running | Start instance in UI |
