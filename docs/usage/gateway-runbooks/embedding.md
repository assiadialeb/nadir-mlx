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

## 4. Wrong route (negative test)

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

**Expected:** HTTP **400** (`unsupported_endpoint`).

## 5. LiteLLM

Add to `config.yaml` (Docker → `host.docker.internal`):

```yaml
model_list:
  - model_name: local-embed
    litellm_params:
      model: openai/<alias>
      api_base: http://host.docker.internal:11380/v1
      api_key: sk-local
    model_info:
      mode: embedding
```

Test:

```python
import litellm

response = litellm.embedding(
    model="local-embed",
    input=["hello world"],
)
assert len(response.data[0]["embedding"]) > 0
```

## Troubleshooting

| HTTP | Cause | Action |
|------|--------|--------|
| 404 | Unknown alias | Check UI alias / `GET /v1/models` |
| 503 | Instance not RUNNING | Start embedding server in UI |
| 400 | Chat route used | Use `/v1/embeddings` only |
| 502 | Upstream down | Check instance logs in UI |
