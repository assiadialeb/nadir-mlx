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

Upstream supports `response_format: b64_json` (default) and **`url`** (local PNG served by the gateway).

For `url`, files are stored under `data/generated_images/` and exposed at:

`GET http://127.0.0.1:11380/v1/images/files/{id}`

Configure the public base if clients reach the gateway via another host:

```bash
NADIR_GATEWAY_PUBLIC_BASE_URL=http://127.0.0.1:11380
IMAGE_OUTPUT_TTL_SECONDS=3600
```

`POST /v1/images/edits` and `/v1/images/variations` return **501** in v1 (mflux txt2img only). See [ADR 003](../../adr/003-image-edits-variations.md).

## 1. Discovery

```bash
curl -s http://127.0.0.1:11380/v1/models | python3 -m json.tool
```

**Expected:** alias with `"metadata": {"launch_mode": "IMAGE"}`.

Optional — read upstream defaults (direct instance port, for tuning):

```bash
curl -s http://127.0.0.1:11400/v1/image/defaults | python3 -m json.tool
```

## 3. URL response format

```bash
curl -s http://127.0.0.1:11380/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Flux-1",
    "prompt": "A red apple on a wooden table",
    "quality": "fast",
    "response_format": "url",
    "n": 1
  }' | python3 -m json.tool
```

Fetch the PNG (same host, no external CDN):

```bash
curl -s -o /tmp/nadir-gateway-flux-url.png "http://127.0.0.1:11380/v1/images/files/<file_id>"
file /tmp/nadir-gateway-flux-url.png
```

## 4. Fast generation (smoke test, b64_json)

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

## 5. Wrong route (negative test)

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Flux-1",
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

**Expected:** HTTP **400** (`unsupported_endpoint`).

!!! tip "Latency"
    First generation after model load may take 1–3 minutes even in `fast` mode. Increase `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS` if you see `504 gateway_timeout`.

## Troubleshooting

| HTTP | Cause | Action |
|------|--------|--------|
| 404 | Wrong alias casing | Use exact id from `GET /v1/models` |
| 504 | Gateway timeout | Raise `NADIR_GATEWAY_PROXY_TIMEOUT_SECONDS`, use `quality: fast` |
| 400 | `url` misconfigured | Set `NADIR_GATEWAY_PUBLIC_BASE_URL` if clients use another host |
| 404 | Image file expired | Regenerate; default TTL 3600s (`IMAGE_OUTPUT_TTL_SECONDS`) |
| 501 | edits / variations | Not supported v1 — use generations |
| 400 | Empty prompt | Provide non-empty `prompt` |
| 503 | Instance not RUNNING | Start IMAGE server in UI |
