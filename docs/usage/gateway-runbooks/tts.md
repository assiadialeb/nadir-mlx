# Runbook — Gateway TTS (MLX-29 / MLX-32)

Validate `POST /v1/audio/speech` through Nadir Gateway (`:11380`) for a **RUNNING** Kokoro TTS instance.

## Prerequisites

- Gateway alias e.g. **`kokoro`** (`launch_mode: TTS` in `GET /v1/models`)
- Instance on port **11444** (or any port — routing is by alias)
- **ffmpeg** on the host for `mp3`, `opus`, `aac`, and `flac` (`brew install ffmpeg`)

## 1. Discovery

```bash
curl -s http://127.0.0.1:11380/v1/models | python3 -m json.tool
```

## 2. Generate WAV (binary response)

```bash
tmp=$(mktemp /tmp/nadir-tts-XXXX.wav)
code=$(curl -s -o "$tmp" -w "%{http_code}" http://127.0.0.1:11380/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello from Nadir gateway",
    "response_format": "wav"
  }')
echo "HTTP:$code size:$(wc -c < "$tmp")"
file "$tmp"
```

**Expected:** HTTP **200**, `RIFF … WAVE audio`, size > 1 KB.

## 3. MP3 variant

```bash
curl -s -o /tmp/nadir-tts-test.mp3 http://127.0.0.1:11380/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Test", "response_format": "mp3"}'
file /tmp/nadir-tts-test.mp3
```

## 4. Opus and AAC (OpenAI-compatible)

```bash
curl -s -o /tmp/nadir-tts-test.opus http://127.0.0.1:11380/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Test", "response_format": "opus"}'
file /tmp/nadir-tts-test.opus

curl -s -o /tmp/nadir-tts-test.aac http://127.0.0.1:11380/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Test", "response_format": "aac"}'
file /tmp/nadir-tts-test.aac
```

**Expected:** HTTP **200**, valid Opus / AAC containers. Unsupported formats (e.g. `webm`) return HTTP **400** with a clear message.

## 5. Chunked streaming relay

The gateway streams the upstream audio body with chunked transfer (no full buffering). OpenAI-compatible clients consume this as a byte stream.

Optional upstream flag `stream: true` on the MLX TTS server also chunks the encoded file after synthesis.

## 6. OpenAI voice mapping (optional)

Kokoro accepts OpenAI-style voices (`alloy`, `nova`, …) and remaps to Kokoro voice IDs:

```bash
curl -s -o /tmp/nadir-tts-alloy.wav http://127.0.0.1:11380/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Bonjour", "voice": "nova", "response_format": "wav"}'
```

## 7. Wrong route

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "messages": [{"role": "user", "content": "Hi"}]}'
```

**Expected:** HTTP **400**.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| JSON instead of audio | Wrong route or upstream error — check Content-Type |
| 400 unsupported format | Use `wav`, `mp3`, `opus`, `aac`, `flac`, or `pcm` |
| 400 ffmpeg / AAC | Install ffmpeg on the MLX host |
| 503 | Start TTS instance in UI |
| Empty WAV | Check `input` is non-empty |
