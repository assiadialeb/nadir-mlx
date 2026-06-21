# Runbook — Gateway STT (MLX-30)

Validate `POST /v1/audio/transcriptions` (multipart) through Nadir Gateway (`:11380`) for a **RUNNING** Whisper STT instance.

## Prerequisites

- Gateway alias e.g. **`whispers`** (`launch_mode: STT`)
- Instance on port **11445**
- Sample audio file (WAV recommended)

!!! note "Restart gateway"
    Multipart relay requires gateway code that detects Starlette `UploadFile` correctly. After updating mlx-server, run `python manage.py run_gateway` again.

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

## 3. Transcribe via gateway

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

## 4. Direct upstream (debug)

```bash
curl -s http://127.0.0.1:11445/v1/audio/transcriptions \
  -F "file=@/tmp/nadir-stt-sample.wav" \
  -F "model=whispers" \
  -F "response_format=json"
```

Gateway should return the same shape when routing works.

## 5. Wrong route

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "whispers", "messages": [{"role": "user", "content": "Hi"}]}'
```

**Expected:** HTTP **400**.

## 6. LiteLLM

```yaml
model_list:
  - model_name: local-whisper
    litellm_params:
      model: openai/whispers
      api_base: http://host.docker.internal:11380/v1
      api_key: sk-local
    model_info:
      mode: audio_transcription
```

## Troubleshooting

| HTTP | Cause | Action |
|------|--------|--------|
| 422 on `file` | Old gateway multipart bug | Restart gateway with latest code |
| 400 | Empty file | Check `-F file=@path` |
| 503 | STT not running | Start instance in UI |

Supported formats follow upstream mlx-audio (WAV, MP3; M4A if ffmpeg available).
