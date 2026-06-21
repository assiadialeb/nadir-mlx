# Runbook — Gateway IMAGE (MLX-28)

Validate `POST /v1/images/generations` through Nadir Gateway (`:11380`) for a **RUNNING** mflux instance (FLUX, Z-Image, etc.).

## Prerequisites

- Django + gateway running
- IMAGE instance **Running** with gateway alias (e.g. `Flux-1`)
- Confirm exact alias via `GET /v1/models` — aliases are **case-sensitive**

## Timeouts

Image generation can exceed chat latency. Default gateway timeout:

```bash
NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS=300   # 5 minutes (default in .env)
```

For slow `quality` presets or large sizes, increase before restarting the gateway:

```bash
export NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS=600
python manage.py run_gateway
```

Upstream mflux only supports `response_format: b64_json` (not `url`).

## 1. Discovery

```bash
curl -s http://127.0.0.1:11380/v1/models | python3 -m json.tool
```

**Expected:** alias with `"metadata": {"launch_mode": "IMAGE"}`.

Optional — read upstream defaults (direct instance port, for tuning):

```bash
curl -s http://127.0.0.1:11400/v1/image/defaults | python3 -m json.tool
```

## 2. Fast generation (smoke test)

Use `quality: fast` for a quicker first run:

```bash
tmp=$(mktemp)
code=$(curl -s -o "$tmp" -w "%{http_code}" http://127.0.0.1:11380/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Flux-1",
    "prompt": "A red apple on a wooden table, studio lighting",
    "quality": "fast",
    "response_format": "b64_json",
    "n": 1
  }')
python3 - "$tmp" "$code" <<'PY'
import base64, json, sys
path, http_code = sys.argv[1], sys.argv[2]
payload = json.loads(open(path, encoding="utf-8").read())
if http_code != "200" or "data" not in payload:
    print(json.dumps(payload, indent=2), file=sys.stderr)
    sys.exit(1)
b64 = payload["data"][0]["b64_json"]
open("/tmp/nadir-gateway-flux-test.png", "wb").write(base64.b64decode(b64))
print("OK — saved /tmp/nadir-gateway-flux-test.png", len(b64), "b64 chars")
PY
rm -f "$tmp"
```

**Expected:** HTTP 200 within timeout, non-empty `b64_json`, PNG decodable.

## 3. Wrong route (negative test)

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Flux-1",
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

**Expected:** HTTP **400** (`unsupported_endpoint`).

## 4. LiteLLM

```yaml
model_list:
  - model_name: local-flux
    litellm_params:
      model: openai/Flux-1
      api_base: http://host.docker.internal:11380/v1
      api_key: sk-local
    model_info:
      mode: image_generation
```

Python sanity check (from LiteLLM venv):

```python
import litellm

response = litellm.image_generation(
    model="local-flux",
    prompt="A blue butterfly on a flower",
    quality="fast",
)
print(response)
```

!!! tip "Latency"
    First generation after model load may take 1–3 minutes even in `fast` mode. Increase `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS` if you see `504 gateway_timeout`.

## Troubleshooting

| HTTP | Cause | Action |
|------|--------|--------|
| 404 | Wrong alias casing | Use exact id from `GET /v1/models` |
| 504 | Gateway timeout | Raise `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS`, use `quality: fast` |
| 400 | `url` response_format | Use `b64_json` only |
| 400 | Empty prompt | Provide non-empty `prompt` |
| 503 | Instance not RUNNING | Start IMAGE server in UI |
