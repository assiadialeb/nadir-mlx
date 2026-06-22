# Runbook — Gateway VLM (vision E2E)

Validate multimodal `POST /v1/chat/completions` with **images in messages** through Nadir Gateway (`:11380`).

!!! note "Gateway vs mlx-vlm"
    The gateway **relays** the full JSON body (including base64 image data). Vision encoding and inference run on the **MULTIMODAL** instance (mlx-vlm).

## Prerequisites

- Django + gateway running
- **MULTIMODAL** instance **Running** (e.g. Gemma 3/4 VLM, Qwen-VL) with a known alias
- Alias listed in `GET /v1/models` with `"metadata": {"launch_mode": "MULTIMODAL"}`

## 1. Discovery

```bash
curl -s http://127.0.0.1:11380/v1/models | python3 -m json.tool
```

## 2. Text-only sanity check (same route)

```bash
curl -s http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<vlm-alias>",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 16
  }' | python3 -m json.tool
```

## 3. Vision — base64 inline (recommended, air-gapped)

Build a data URI from a local PNG/JPEG:

```bash
IMAGE_B64=$(python3 - <<'PY'
import base64
from pathlib import Path
path = Path("<path-to-image.png>")
print(base64.b64encode(path.read_bytes()).decode("ascii"))
PY
)

curl -s http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$(python3 - <<PY
import json
payload = {
    "model": "<vlm-alias>",
    "max_tokens": 128,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in one sentence."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,${IMAGE_B64}"}
            }
        ]
    }]
}
print(json.dumps(payload))
PY
)" | python3 -m json.tool
```

**Expected:** HTTP 200, `choices[0].message.content` describes the image plausibly.

## 4. Vision — local file path (mlx-vlm native)

When the mlx-vlm process can read the filesystem (same Mac user):

```bash
curl -s http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<vlm-alias>",
    "max_tokens": 128,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "/absolute/path/to/image.jpg"}}
      ]
    }]
  }' | python3 -m json.tool
```

## 5. Vision — gateway-hosted PNG (MLX-34)

If you generated an image with `response_format: url`, reuse the local gateway URL:

```bash
# After POST /v1/images/generations with response_format url → capture file URL
curl -s http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<vlm-alias>",
    "max_tokens": 128,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe the generated image."},
        {"type": "image_url", "image_url": {"url": "http://127.0.0.1:11380/v1/images/files/<file_id>"}}
      ]
    }]
  }' | python3 -m json.tool
```

!!! warning "Air-gapped policy"
    External `https://` image URLs require outbound network on the Mac. Prefer base64 or local paths in isolated environments.

## 6. Streaming vision

```bash
curl -N http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<vlm-alias>",
    "stream": true,
    "max_tokens": 128,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe briefly."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64>"}}
      ]
    }]
  }'
```

**Expected:** `text/event-stream` chunks. First-token latency is higher than text-only chat.

## 7. LiteLLM vision

```yaml
model_list:
  - model_name: local-vlm
    litellm_params:
      model: openai/<vlm-alias>
      api_base: http://host.docker.internal:11380/v1
      api_key: sk-local
    model_info:
      mode: chat
```

```python
import base64
import litellm

with open("<path-to-image.png>", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode("ascii")

response = litellm.completion(
    model="local-vlm",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ],
        }
    ],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

## 8. Negative tests

```bash
# Legacy completions on VLM alias → 400
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:11380/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<vlm-alias>", "prompt": "Hi"}'
```

**Expected:** HTTP **400** (`unsupported_endpoint`).

## Troubleshooting

| Symptom | Cause | Action |
|---------|--------|--------|
| 404 | Unknown alias | Check `GET /v1/models` |
| 503 | VLM instance stopped | Start MULTIMODAL server in UI |
| 502 / 504 | Upstream timeout | Increase `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS`; shrink image |
| 200 but ignores image | Text-only payload | Use `content` array with `image_url` block |
| Wrong image in multi-image message | mlx-vlm limitation | Send one image per request ([matrix](../vlm-vision-model-matrix.md)) |
| Garbage output | MLX / model mismatch | Update `mlx`, `mlx-vlm`; check instance logs |
| `(1, 1, 1), \|u1` on tiny PNG | Image too small for mlx-vlm / PIL | Use ≥32×32 RGB; real photo or script in [matrix](../vlm-vision-model-matrix.md) |

## References

- [vlm-vision-model-matrix.md](../vlm-vision-model-matrix.md)
- [ADR 005](../../adr/005-vlm-vision-gateway.md)
- [gateway-runbooks/chat.md](chat.md) (tools / JSON on VLM)
