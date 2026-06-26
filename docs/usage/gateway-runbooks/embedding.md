# Runbook — Gateway EMBEDDING (MLX-26)

Validate `POST /v1/embeddings` through Nadir Gateway (`:11380`) for a **RUNNING** embedding instance.

## Prerequisites

- Django + gateway running (`python manage.py runserver`, `python manage.py run_gateway`)
- EMBEDDING instance **Running** with a known gateway alias (e.g. `nomic-embed-text`)
- Alias listed in `GET /v1/models` with `"metadata": {"launch_mode": "EMBEDDING"}`

## 1. Discovery

```bash
curl -s http://127.0.0.1:11380/v1/models | python3 -m json.tool
```

Confirm your embedding alias appears.

## 2. Single string input

```bash
curl -s http://127.0.0.1:11380/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "input": "Local embeddings on Apple Silicon"
  }' | python3 -m json.tool
```

**Expected:** HTTP 200, `"object": "list"`, `"data"[0].embedding` is a non-empty float array.

## 3. Batch input

```bash
curl -s http://127.0.0.1:11380/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "input": ["first sentence", "second sentence"]
  }' | python3 -m json.tool
```

**Expected:** Two entries in `data`, indices 0 and 1.

## 4. Base64 encoding

```bash
curl -s http://127.0.0.1:11380/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "input": "Encode me as base64",
    "encoding_format": "base64"
  }' | python3 -m json.tool
```

**Expected:** HTTP 200, `"data"[0].embedding` is a **base64 string** (not a float array).

Decode locally (Python):

```python
import base64, struct

encoded = "<paste embedding string>"
packed = base64.b64decode(encoded)
vector = struct.unpack(f"<{len(packed) // 4}f", packed)
print(vector)
```

## 5. Dimension truncation

Matryoshka-style truncation keeps the first `dimensions` floats (OpenAI-compatible):

```bash
curl -s http://127.0.0.1:11380/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "input": "Shorter vector please",
    "dimensions": 256
  }' | python3 -m json.tool
```

**Expected:** HTTP 200, embedding length **256** (or HTTP **400** if the model outputs fewer dimensions).

Combine with base64:

```bash
curl -s http://127.0.0.1:11380/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "input": "Compact base64",
    "encoding_format": "base64",
    "dimensions": 128
  }' | python3 -m json.tool
```

## 6. Wrong route (negative test)

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

**Expected:** HTTP **400** (`unsupported_endpoint`).

## Troubleshooting

| HTTP | Cause | Action |
|------|--------|--------|
| 404 | Unknown alias | Check UI alias / `GET /v1/models` |
| 503 | Instance not RUNNING | Start embedding server in UI |
| 400 | Chat route used | Use `/v1/embeddings` only |
| 400 | Invalid `dimensions` or `encoding_format` | Use positive `dimensions` ≤ vector size; `float` or `base64` only |
| 502 | Upstream down | Check instance logs in UI |
