# VLM vision — model matrix & limits

Qualification for OpenAI-style vision messages via Nadir Gateway → mlx-vlm.

Last updated: June 2026.

## Gateway relay

| Feature | Status |
|---------|--------|
| `messages[].content[]` with `image_url` | ✅ deep-copy relay |
| Large base64 data URIs | ✅ not truncated by gateway |
| `stream: true` with vision | ✅ SSE relay |
| Rewrite `model` only | ✅ upstream alias / folder name |

## Image input formats (mlx-vlm)

| Format | Example | Air-gapped |
|--------|---------|------------|
| Base64 data URI | `data:image/png;base64,...` | ✅ recommended |
| Local absolute path | `/Users/.../photo.jpg` | ✅ if file readable by instance |
| Gateway image file URL | `http://127.0.0.1:11380/v1/images/files/{id}` | ✅ local loopback |
| External HTTPS URL | `https://example.com/a.jpg` | ❌ needs network |

Supported MIME types depend on mlx-vlm / Pillow — **JPEG** and **PNG** are the primary QA path.

## Validated model families (Nadir MLX registry)

| Family | Example weights | Vision QA | Notes |
|--------|-----------------|-----------|-------|
| **Gemma 3 VLM** | `gemma-3-4b-it-qat-4bit`, `gemma-3-12b-it-qat-4bit` | ✅ | Default Mac Studio VLM path |
| **Gemma 4 VLM** | `gemma-4-e2b-it-4bit`, `gemma-4-12B-it-8bit` | ✅ | Unified config patched in `model_utils` |
| **Qwen-VL** (when installed) | `mlx-community/Qwen3-VL-*` | ✅ | Strong vision; check RAM |

TEXT-only weights (Llama, Qwen2.5 instruct without vision) must **not** be launched as MULTIMODAL.

## Known limitations

| Topic | Detail |
|-------|--------|
| **Multi-image per message** | Some mlx-vlm builds process only the **last** `image_url` block — prefer one image per request |
| **Tiny images (1×1 px)** | ❌ Gemma 4 / PIL may reject `(1,1,1)` uint8 — use ≥32×32 RGB PNG/JPEG |
| **Payload size** | Very large base64 JSON increases latency; resize images client-side when possible |
| **Timeout** | Default gateway proxy timeout **300s** (`NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS`) |
| **Tools + vision** | Both relayed; combined agent flows depend on mlx-vlm + model ([chat-tools-model-matrix.md](chat-tools-model-matrix.md)) |

## Quick qualification script

Use a **small but valid RGB image** (at least ~32×32 px). The 1×1 PNG from early drafts **fails** on Gemma 4 VLM with:

`Cannot handle this data type: (1, 1, 1), |u1`

```bash
# Generate a 64×64 red PNG and call the gateway
python3 - <<'PY' | curl -s http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @- | python3 -m json.tool
import base64, io, json
from PIL import Image
img = Image.new("RGB", (64, 64), color=(220, 40, 40))
buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
print(json.dumps({
    "model": "<vlm-alias>",
    "max_tokens": 32,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What color is dominant? One word."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }],
}))
PY
```

Pass criteria: HTTP 200 and a color-related answer (model may say *red*, *black*, etc. depending on preprocessing).

## References

- [gateway-runbooks/vlm.md](gateway-runbooks/vlm.md)
- mlx-vlm issue [#1084](https://github.com/Blaizzy/mlx-vlm/issues/1084) (multi-image)
